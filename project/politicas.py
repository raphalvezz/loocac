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
    # Todas as colunas: no ambiente sequencial o estado inclui reputação e período
    codigos, _ = pd.factorize(pd.MultiIndex.from_frame(estados.astype(str)))
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
        return [self.faixa_da_chave(k) for k in chaves(self.modelo, estados)]

    def faixa_da_chave(self, chave):
        """Faixa observável; para estados fora do catálogo (ex.: orçamento digitado na
        tela que não é de nenhum cenário, ou memória da assinatura com valor médio),
        usa os cenários mais próximos do mesmo Tier."""
        if chave in self.catalogo:
            return self.catalogo[chave]
        tier, orcamento = chave[0], chave[1]
        mesmo_tier = [k for k in self.catalogo if k[0] == tier]
        if not mesmo_tier:
            raise KeyError(f"Tier fora do catálogo: {tier}")
        orc_proximo = min({k[1] for k in mesmo_tier}, key=lambda o: abs(np.log(o) - np.log(orcamento)))
        faixas = [self.catalogo[k] for k in mesmo_tier if k[1] == orc_proximo]
        return min(f[0] for f in faixas), max(f[1] for f in faixas)

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

    def __init__(self, modelo, cenarios, n_grade=200, extras=()):
        super().__init__(modelo, cenarios)
        self.n_grade = n_grade
        self.extras = tuple(extras)  # colunas numéricas extras do estado (ex.: reputação)

    def nomes_features(self):
        nomes = list(self.ohe.get_feature_names_out()) + ['Orcamento']
        if self.modelo == 'assinatura':
            nomes += MEMORIA
        return nomes + list(getattr(self, 'extras', ())) + ['Preco']

    def _X(self, estados, preco):
        f = np.log if self.escala_log else np.asarray
        # Mesmo estado que o CQL vê: na assinatura, inclui as features de memória
        partes = [self.ohe.transform(estados[CATEGORICAS]),
                  f(estados[['Orcamento']].to_numpy(dtype=float))]
        if self.modelo == 'assinatura':
            partes.append(estados[MEMORIA].to_numpy(dtype=float))
        if getattr(self, 'extras', ()):
            partes.append(estados[list(self.extras)].to_numpy(dtype=float))
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

    def __init__(self, modelo, ohe=None, scaler_estado=None, scaler_memoria=None, extras=()):
        self.modelo, self.ohe, self.scaler_estado, self.scaler_memoria = modelo, ohe, scaler_estado, scaler_memoria
        self.extras = tuple(extras)  # colunas numéricas já em escala ~1 (ex.: reputação, período)

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
        if getattr(self, 'extras', ()):
            partes.append(estados[list(self.extras)].to_numpy(dtype=float))
        return np.concatenate(partes, axis=1).astype(np.float32)


class AgenteD3(Politica):
    """Base dos agentes d3rlpy (CQL e BC): ação [-1, 1] mapeada para uma faixa de
    preço (log min-max), pré-processamento do estado e salvar/carregar.

    escala_acao  'estado' ação [-1, 1] = faixa observável do estado, a mesma em que o
                          bandido busca o preço (PADRÃO; ver protocolo, seção 12)
                 'tier'   ação [-1, 1] = faixa do Tier do estado (versão dos notebooks)
                 'global' ação [-1, 1] = faixa de todos os preços
    """
    treinavel = True
    gpu = 'opcional (acelera o treino)'

    def __init__(self, modelo, cenarios, escala_acao='estado', extras=()):
        super().__init__(modelo, cenarios)
        self.extras = tuple(extras)
        self.faixas_tier = sim.faixas_tier(cenarios)
        if escala_acao == 'global':
            lo = min(f[0] for f in self.faixas_tier.values())
            hi = max(f[1] for f in self.faixas_tier.values())
            self.faixas_tier = {t: [lo, hi] for t in self.faixas_tier}
        elif escala_acao not in ('tier', 'estado'):
            raise ValueError(escala_acao)
        self.escala_acao = escala_acao
        self.algo = None

    def faixas_acao(self, estados):
        """(lo, hi) de preço que a ação [-1, 1] cobre, por linha de `estados`."""
        if self.escala_acao == 'estado':
            return np.array(self.faixas(estados), dtype=float)
        return np.array([self.faixas_tier[t] for t in estados['Tier']], dtype=float)

    def preco_para_acao(self, precos, estados):
        lo, hi = np.log(self.faixas_acao(estados)).T
        return 2 * (np.log(np.asarray(precos, dtype=float)) - lo) / (hi - lo) - 1

    def acao_para_preco(self, acoes, estados):
        lo, hi = np.log(self.faixas_acao(estados)).T
        return np.exp(lo + (np.clip(acoes, -1.0, 1.0) + 1) / 2 * (hi - lo))

    def _buffer(self, treino, recompensas):
        from d3rlpy.dataset import Episode, FIFOBuffer, ReplayBuffer
        self.prep = PreprocessadorEstado(self.modelo, extras=self.extras).fit(treino)
        obs = self.prep.transform(treino)
        acoes = self.preco_para_acao(treino['Preco_Amostra'].to_numpy(), treino).reshape(-1, 1).astype(np.float32)
        recompensas = np.asarray(recompensas, dtype=np.float32).reshape(-1, 1)
        if 'episodio' not in treino.columns:
            # Decisões independentes: um único episódio (com gamma=0 a ordem não importa)
            episodios = [Episode(obs, acoes, recompensas, True)]
        else:
            # Ambiente sequencial: um episódio por campanha, na ordem dos períodos
            episodios = []
            for idx in treino.groupby('episodio', sort=False).indices.values():
                idx = idx[np.argsort(treino['passo'].to_numpy()[idx])]
                episodios.append(Episode(obs[idx], acoes[idx], recompensas[idx], True))
        return ReplayBuffer(FIFOBuffer(limit=len(treino)), episodes=episodios)

    def precos(self, estados):
        acoes = self.algo.predict(self.prep.transform(estados)).reshape(-1)
        return self.acao_para_preco(acoes, estados)

    def salvar(self, prefixo):
        """Grava <prefixo>.d3 (rede, d3rlpy) e <prefixo>.joblib (encoders e configuração)."""
        import joblib
        algo, avaliador = self.algo, getattr(self, 'avaliador', None)
        algo.save(f'{prefixo}.d3')
        self.algo, self.avaliador = None, None  # funções/rede não vão no joblib
        try:
            joblib.dump(self, f'{prefixo}.joblib')
        finally:
            self.algo, self.avaliador = algo, avaliador

    @classmethod
    def carregar(cls, prefixo):
        import d3rlpy
        import joblib
        politica = joblib.load(f'{prefixo}.joblib')
        politica.algo = d3rlpy.load_learnable(f'{prefixo}.d3', device='cpu')
        return politica


class CQL(AgenteD3):
    """CQL (d3rlpy). Esta classe (com treinar_cql.py) é a versão CANÔNICA do
    agente; os notebooks de RL só documentam a versão anterior.

    Opções usadas nas ablações (ablacoes_cql.py); padrões:
      gamma               0.0 = bandido contextual de um passo
      peso_conservador    conservative_weight do CQL
      escala_acao         ver AgenteD3
      escala_recompensa   'global' padronização única (lucros de centenas a milhões)
                          'log'    log(lucro) padronizado
                          'estado' lucro / lucro médio da chave de estado - 1; com
                                   gamma=0 não muda o melhor preço de cada estado
    """
    nome = 'cql'
    explicabilidade = 'baixa: rede neural; expõe só a ação e os quantis de lucro do crítico'

    def __init__(self, modelo, cenarios, n_passos=50000, passos_por_epoca=1000, paciencia=20,
                 avaliador=None, gamma=0.0, peso_conservador=5.0, escala_acao='estado',
                 escala_recompensa='global', extras=()):
        super().__init__(modelo, cenarios, escala_acao, extras)
        if escala_recompensa not in ('global', 'log', 'estado'):
            raise ValueError(escala_recompensa)
        self.gamma, self.peso_conservador = gamma, peso_conservador
        self.escala_recompensa = escala_recompensa
        self.n_passos, self.passos_por_epoca, self.paciencia = n_passos, passos_por_epoca, paciencia
        # Função (algo -> score) para escolher o checkpoint; ver avaliar_politicas.metrica_selecao
        self.avaliador = avaliador

    @classmethod
    def de_artefatos(cls, modelo, cenarios, caminho_modelo, ohe, scaler_estado, scaler_memoria=None):
        """Carrega um agente salvo pelos notebooks (.d3, ação por Tier) com os encoders do Generator."""
        import d3rlpy
        pol = cls(modelo, cenarios, escala_acao='tier')
        pol.algo = d3rlpy.load_learnable(caminho_modelo, device='cpu')
        pol.prep = PreprocessadorEstado(modelo, ohe, scaler_estado, scaler_memoria)
        return pol

    def treinar(self, dados, semente):
        import d3rlpy
        from d3rlpy.algos import CQLConfig
        from d3rlpy.models import QRQFunctionFactory

        d3rlpy.seed(semente)
        treino = dados.iloc[: int(len(dados) * FRACAO_TREINO)]
        buffer = self._buffer(treino, self._recompensas(treino))

        self.algo = CQLConfig(
            batch_size=256, gamma=self.gamma,
            observation_scaler=None, action_scaler=None, reward_scaler=None,
            alpha_learning_rate=1e-4, actor_learning_rate=1e-4, critic_learning_rate=3e-4,
            conservative_weight=self.peso_conservador, q_func_factory=QRQFunctionFactory(n_quantiles=64),
        ).create(device='cuda:0' if _cuda() else 'cpu')

        melhor, sem_melhora = -np.inf, 0
        with tempfile.TemporaryDirectory() as tmp:
            caminho = os.path.join(tmp, 'melhor.pt')
            for _, _ in self.algo.fitter(buffer, n_steps=self.n_passos,
                                         n_steps_per_epoch=min(self.passos_por_epoca, self.n_passos),
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
        return self

    def _recompensas(self, treino):
        lucro = treino[ALVO[self.modelo]].to_numpy(dtype=float)
        if self.escala_recompensa == 'log':
            lucro = np.log(np.maximum(lucro, 1e-6))
            self.escala_rec = (float(lucro.mean()), float(lucro.std()))
            return ((lucro - self.escala_rec[0]) / self.escala_rec[1]).reshape(-1, 1)
        elif self.escala_recompensa == 'estado':
            grupo = [str(k) for k in chaves(self.modelo, treino)]
            media = pd.Series(lucro).groupby(grupo).transform('mean').to_numpy()
            return (lucro / media - 1).reshape(-1, 1)
        # Guarda a escala para converter os quantis do crítico de volta a US$ (quantis_reais)
        self.escala_rec = (float(lucro.mean()), float(lucro.std()))
        return ((lucro - self.escala_rec[0]) / self.escala_rec[1]).reshape(-1, 1)

    def quantis(self, estados, precos):
        """Quantis do crítico QR (escala da recompensa de treino) para (estado, preço)."""
        import torch
        q = self.algo.impl._q_func_forwarder._forwarders[0]._q_func
        q.eval()
        acoes = self.preco_para_acao(precos, estados).reshape(-1, 1)
        with torch.no_grad():
            out = q(torch.tensor(self.prep.transform(estados), dtype=torch.float32),
                    torch.tensor(acoes, dtype=torch.float32))
        return out.quantiles.cpu().numpy().reshape(len(estados), -1)


    def quantis_reais(self, estados, precos):
        """Quantis do crítico convertidos para US$ (lucro ou LTV)."""
        q = self.quantis(estados, precos)
        media, desvio = self.escala_rec
        if self.escala_recompensa == 'global':
            return q * desvio + media
        if self.escala_recompensa == 'log':
            return np.exp(q * desvio + media)
        raise NotImplementedError("escala_recompensa='estado' não tem conversão única para US$")


class BC(AgenteD3):
    """Behavior cloning: imita os preços da política que gerou os dados.

    Com coleta uniforme, aprende o preço médio (em ação) de cada estado. É a
    referência mínima de um método offline: o RL só se justifica se superar a
    imitação do histórico.
    """
    nome = 'bc'
    explicabilidade = 'baixa: rede neural que reproduz os preços do histórico'

    def __init__(self, modelo, cenarios, n_passos=5000, escala_acao='estado', extras=()):
        super().__init__(modelo, cenarios, escala_acao, extras)
        self.n_passos = n_passos

    def treinar(self, dados, semente):
        import d3rlpy
        from d3rlpy.algos import BCConfig

        d3rlpy.seed(semente)
        treino = dados.iloc[: int(len(dados) * FRACAO_TREINO)]
        buffer = self._buffer(treino, np.zeros(len(treino)))  # BC não usa recompensa
        self.algo = BCConfig(batch_size=256, learning_rate=1e-3,
                             observation_scaler=None, action_scaler=None).create(
            device='cuda:0' if _cuda() else 'cpu')
        self.algo.fit(buffer, n_steps=self.n_passos, n_steps_per_epoch=min(1000, self.n_passos),
                      logger_adapter=d3rlpy.logging.NoopAdapterFactory(), show_progress=False)
        return self


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

