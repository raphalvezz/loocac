"""
Interface única de política de preço.

Toda abordagem implementa:
    treinar(dados, semente)  -> aprende com o dataset logado (no-op nas heurísticas)
    precos(estados)          -> um preço por linha de `estados`

`estados` é um DataFrame com as colunas do estado (Regiao, Plataforma, Tier,
Orcamento, categóricas fixas e memória da assinatura). Só as políticas
privilegiadas (oráculos) recebem também a coluna 'Cenario', que identifica o
mundo verdadeiro; o avaliador a remove para as demais.

As políticas não privilegiadas conhecem o catálogo de faixas por chave de
estado (ver chave_estado), que é o suporte dos dados logados para aquele estado.
"""

import os
import tempfile

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import simulador as sim

CATEGORICAS = ['Regiao', 'Plataforma', 'Tier', 'Idade', 'Genero', 'Conteudo',
               'Tipo_Produto', 'Modelo_Cobranca', 'Complexidade_Oferta']
MEMORIA = ['dias_desde_ultima_interacao', 'clv_estimate_percentile',
           'avg_price_offered_segment_90d', 'price_volatility_30d']
ALVO = {'venda_unica': 'Lucro_Real', 'assinatura': 'LTV_Real'}
FRACAO_TREINO = 0.8  # mesmo split 80/20 dos notebooks (dados embaralhados)
COLUNAS_ESTADO = CATEGORICAS + ['Orcamento'] + MEMORIA


def por_estado_unico(funcao, estados):
    """Aplica funcao(estados_unicos) -> preços e devolve um preço por linha de `estados`.

    Um lote de decisões repete os mesmos estados (a grade tem 144); prever só os
    distintos dá o mesmo resultado com muito menos custo. Só para políticas
    determinísticas por estado.
    """
    colunas = [c for c in COLUNAS_ESTADO + ['Cenario'] if c in estados.columns]
    codigos, _ = pd.factorize(pd.MultiIndex.from_frame(estados[colunas].astype(str)))
    primeiras = pd.Series(range(len(codigos))).groupby(codigos).first().to_numpy()
    return np.asarray(funcao(estados.iloc[primeiras].reset_index(drop=True)))[codigos]


# ---------------------------------------------------------------------------
# O que o estado permite saber
# ---------------------------------------------------------------------------
def chave_estado(modelo, tier, orcamento, preco_medio_90d=None):
    """Parte do estado que distingue cenários.

    Venda única: (Tier, Orçamento). Os cenários 3/4 e 5/6 coincidem aqui.
    Assinatura: o estado também tem avg_price_offered_segment_90d (= preço mínimo
    do cenário nos dados), que separa 3/4 e 5/6.
    """
    if modelo == 'assinatura':
        return (tier, float(orcamento), float(preco_medio_90d))
    return (tier, float(orcamento))


def chave_cenario(modelo, cenario):
    return chave_estado(modelo, cenario['Tier'], cenario['Budget'],
                        sim.memoria_assinatura(cenario)['avg_price_offered_segment_90d'])


def chaves(modelo, estados):
    memoria = estados['avg_price_offered_segment_90d'] if modelo == 'assinatura' else [None] * len(estados)
    return [chave_estado(modelo, t, o, m) for t, o, m in zip(estados['Tier'], estados['Orcamento'], memoria)]


def catalogo_faixas(modelo, cenarios):
    """Chave de estado -> faixa de preço observável (união das faixas dos cenários com a mesma chave)."""
    grupos = {}
    for c in cenarios:
        grupos.setdefault(chave_cenario(modelo, c), []).append(c)
    return {k: (min(c['Price_Min'] for c in g), max(c['Price_Max'] for c in g)) for k, g in grupos.items()}


class Politica:
    nome = 'base'
    treinavel = False      # aprende com os dados (varia com a semente)
    privilegiada = False   # usa informação que o estado não tem
    gpu = 'não'
    explicabilidade = ''

    def __init__(self, modelo, cenarios):
        self.modelo = modelo
        self.cenarios = cenarios
        self.catalogo = catalogo_faixas(modelo, cenarios)

    def faixas(self, estados):
        return [self.catalogo[k] for k in chaves(self.modelo, estados)]

    def treinar(self, dados, semente):
        pass

    def precos(self, estados):
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Heurísticas
# ---------------------------------------------------------------------------
class Aleatoria(Politica):
    nome = 'aleatorio'
    explicabilidade = 'trivial: sorteio uniforme na faixa (a política que coletou os dados)'

    def treinar(self, dados, semente):
        self.rng = np.random.default_rng(semente)

    def precos(self, estados):
        return np.array([self.rng.uniform(lo, hi) for lo, hi in self.faixas(estados)])


class MeioFaixa(Politica):
    nome = 'meio_faixa'
    explicabilidade = 'total: regra fixa (ponto médio da faixa)'

    def precos(self, estados):
        return np.array([(lo + hi) / 2 for lo, hi in self.faixas(estados)])


class MaximoFaixa(Politica):
    nome = 'maximo_faixa'
    explicabilidade = 'total: regra fixa (teto da faixa)'

    def precos(self, estados):
        return np.array([hi for _, hi in self.faixas(estados)])


# ---------------------------------------------------------------------------
# Regressor de lucro + busca em grade (SL e bandido contextual)
# ---------------------------------------------------------------------------
class RegressorGrade(Politica):
    """Aprende lucro(estado, preço) com LightGBM e, para cada estado, escolhe o
    preço de maior lucro previsto numa grade dentro da faixa observável.

    Offline e sem exploração, um bandido contextual guloso É isto; SL e bandido
    (bandido.py) diferem só na configuração (alvo, escala das features,
    hiperparâmetros).
    """
    treinavel = True
    explicabilidade = 'alta: curva de lucro prevista por preço; importância/SHAP por feature'
    parametros = {}
    escala_log = False  # alvo e features numéricas em log

    def __init__(self, modelo, cenarios, n_grade=200):
        super().__init__(modelo, cenarios)
        self.n_grade = n_grade

    def nomes_features(self):
        nomes = list(self.ohe.get_feature_names_out()) + ['Orcamento']
        if self.modelo == 'assinatura':
            nomes += MEMORIA
        return nomes + ['Preco']

    def _X(self, estados, preco):
        f = np.log if self.escala_log else np.asarray
        # Mesmo estado que o CQL vê: na assinatura, inclui as features de memória
        partes = [self.ohe.transform(estados[CATEGORICAS]),
                  f(estados[['Orcamento']].to_numpy(dtype=float))]
        if self.modelo == 'assinatura':
            partes.append(estados[MEMORIA].to_numpy(dtype=float))
        partes.append(f(np.asarray(preco, dtype=float)).reshape(-1, 1))
        return np.column_stack(partes)

    def treinar(self, dados, semente, fracao_treino=FRACAO_TREINO):
        import lightgbm as lgb
        treino = dados.iloc[: int(len(dados) * fracao_treino)]
        self.ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit(treino[CATEGORICAS])
        self.lgb = lgb.LGBMRegressor(**self.parametros, random_state=semente, verbose=-1)
        alvo = treino[ALVO[self.modelo]].to_numpy()
        if self.escala_log:
            alvo = np.log(np.maximum(alvo, 1e-6))
        self.lgb.fit(self._X(treino, treino['Preco_Amostra'].to_numpy()), alvo)
        return self

    def prever_lucro(self, estados, precos):
        """Lucro previsto (US$) para cada par (estado, preço)."""
        pred = self.lgb.predict(self._X(estados, precos))
        return np.exp(pred) if self.escala_log else pred

    def precos(self, estados):
        return por_estado_unico(self._precos, estados)

    def _precos(self, estados):
        n = self.n_grade
        # Grade só dentro da faixa observável: fora dela não há dados e as árvores extrapolam
        grades = np.stack([np.geomspace(lo, hi, n) for lo, hi in self.faixas(estados)])
        rep = estados.loc[estados.index.repeat(n)].reset_index(drop=True)
        pred = self.lgb.predict(self._X(rep, grades.ravel())).reshape(len(estados), n)
        return grades[np.arange(len(estados)), pred.argmax(axis=1)]


class SL(RegressorGrade):
    """O regressor do SL_FINAL usado como política.

    Mesmos hiperparâmetros e alvo em US$ do notebook. Em relação à versão
    original (só categóricas), ganha Orçamento e Preço como features: sem o
    preço não há como escolher preço, e sem o orçamento o modelo não separa
    cenários do mesmo Tier.
    """
    nome = 'sl'
    parametros = {'n_estimators': 200, 'learning_rate': 0.1, 'num_leaves': 31}
    escala_log = False


# ---------------------------------------------------------------------------
# CQL (d3rlpy): mesma configuração dos notebooks
# ---------------------------------------------------------------------------
class PreprocessadorEstado:
    """OHE das categóricas + padronização do orçamento (+ memória na assinatura)."""

    def __init__(self, modelo, ohe=None, scaler_estado=None, scaler_memoria=None):
        self.modelo, self.ohe, self.scaler_estado, self.scaler_memoria = modelo, ohe, scaler_estado, scaler_memoria

    def fit(self, dados):
        self.ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit(dados[CATEGORICAS])
        self.scaler_estado = StandardScaler().fit(dados[['Orcamento']])
        if self.modelo == 'assinatura':
            self.scaler_memoria = StandardScaler().fit(dados[MEMORIA])
        return self

    def transform(self, estados):
        partes = [self.ohe.transform(estados[list(self.ohe.feature_names_in_)]),
                  self.scaler_estado.transform(estados[list(self.scaler_estado.feature_names_in_)])]
        if self.modelo == 'assinatura':
            partes.append(self.scaler_memoria.transform(estados[list(self.scaler_memoria.feature_names_in_)]))
        return np.concatenate(partes, axis=1).astype(np.float32)


class CQL(Politica):
    nome = 'cql'
    treinavel = True
    gpu = 'opcional (acelera o treino)'
    explicabilidade = 'baixa: rede neural; expõe só a ação e os quantis de lucro do crítico'

    def __init__(self, modelo, cenarios, n_passos=50000, passos_por_epoca=1000, paciencia=20,
                 avaliador=None):
        super().__init__(modelo, cenarios)
        self.faixas_tier = sim.faixas_tier(cenarios)
        self.n_passos, self.passos_por_epoca, self.paciencia = n_passos, passos_por_epoca, paciencia
        # Função (algo -> score) para escolher o checkpoint; ver avaliar_politicas.metrica_selecao
        self.avaliador = avaliador

    @classmethod
    def de_artefatos(cls, modelo, cenarios, caminho_modelo, ohe, scaler_estado, scaler_memoria=None):
        """Carrega o agente salvo pelos notebooks (.d3) com os encoders do Generator."""
        import d3rlpy
        pol = cls(modelo, cenarios)
        pol.algo = d3rlpy.load_learnable(caminho_modelo, device='cpu')
        pol.prep = PreprocessadorEstado(modelo, ohe, scaler_estado, scaler_memoria)
        return pol

    def treinar(self, dados, semente):
        import d3rlpy
        from d3rlpy.algos import CQLConfig
        from d3rlpy.dataset import Episode, FIFOBuffer, ReplayBuffer
        from d3rlpy.models import QRQFunctionFactory

        d3rlpy.seed(semente)
        treino = dados.iloc[: int(len(dados) * FRACAO_TREINO)]
        self.prep = PreprocessadorEstado(self.modelo).fit(treino)
        scaler_rec = StandardScaler().fit(treino[[ALVO[self.modelo]]])
        acoes = np.array([sim.preco_para_acao(p, t, self.faixas_tier)
                          for p, t in zip(treino['Preco_Amostra'], treino['Tier'])]).reshape(-1, 1)
        episodio = Episode(self.prep.transform(treino), acoes.astype(np.float32),
                           scaler_rec.transform(treino[[ALVO[self.modelo]]]).astype(np.float32), True)
        buffer = ReplayBuffer(FIFOBuffer(limit=len(treino)), episodes=[episodio])

        self.algo = CQLConfig(
            batch_size=256, gamma=0.0,  # bandido contextual de um passo
            observation_scaler=None, action_scaler=None, reward_scaler=None,
            alpha_learning_rate=1e-4, actor_learning_rate=1e-4, critic_learning_rate=3e-4,
            conservative_weight=5.0, q_func_factory=QRQFunctionFactory(n_quantiles=64),
        ).create(device='cuda:0' if _cuda() else 'cpu')

        melhor, sem_melhora = -np.inf, 0
        with tempfile.TemporaryDirectory() as tmp:
            caminho = os.path.join(tmp, 'melhor.pt')
            for _, _ in self.algo.fitter(buffer, n_steps=self.n_passos,
                                         n_steps_per_epoch=self.passos_por_epoca,
                                         logger_adapter=d3rlpy.logging.NoopAdapterFactory(),
                                         show_progress=False):
                score = self.avaliador(self) if self.avaliador else 0.0
                if score > melhor:
                    melhor, sem_melhora = score, 0
                    self.algo.save_model(caminho)
                else:
                    sem_melhora += 1
                    if sem_melhora >= self.paciencia:
                        break
            self.algo.load_model(caminho)
        self.score_selecao = melhor

    def precos(self, estados):
        acoes = self.algo.predict(self.prep.transform(estados)).reshape(-1)
        return np.array([float(sim.acao_para_preco(a, t, self.faixas_tier))
                         for a, t in zip(acoes, estados['Tier'])])


def _cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Referências (privilegiadas)
# ---------------------------------------------------------------------------
class OraculoObservavel(Politica):
    """Melhor preço possível sabendo só o estado: maximiza a MÉTRICA (% do lucro
    ótimo), em média entre os cenários que o estado não distingue (pesos iguais,
    como nos dados). É o teto real de qualquer política nessa métrica.

    Maximizar o lucro médio em US$ não serve: o cenário de lucro maior domina e o
    resultado deixa de ser teto para a média de % do ótimo."""
    nome = 'oraculo_obs'
    privilegiada = True  # conhece a função de lucro do simulador
    explicabilidade = 'referência (usa o simulador)'

    def precos(self, estados):
        return por_estado_unico(self._precos, estados)

    def _precos(self, estados, n=2001):
        out = []
        for chave, r in zip(chaves(self.modelo, estados), estados.itertuples()):
            grupo = [c for c in self.cenarios if chave_cenario(self.modelo, c) == chave]
            grade = np.geomspace(*self.catalogo[chave], n)
            media = np.mean([sim.lucro_esperado(grade, c, r.Regiao, r.Plataforma, self.modelo)
                             / sim.preco_otimo(c, r.Regiao, r.Plataforma, self.modelo)[1]
                             for c in grupo], axis=0)
            out.append(float(grade[np.argmax(media)]))
        return np.array(out)


class Oraculo(Politica):
    """Ótimo do simulador conhecendo o cenário verdadeiro (referência dos percentuais)."""
    nome = 'oraculo'
    privilegiada = True
    explicabilidade = 'referência (usa o simulador e o cenário verdadeiro)'

    def precos(self, estados):
        return por_estado_unico(lambda e: [sim.preco_otimo(self.cenarios[r.Cenario], r.Regiao, r.Plataforma,
                                                            self.modelo)[0] for r in e.itertuples()], estados)

