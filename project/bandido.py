"""
Bandido contextual de precificação: a referência forte do RL.

(No fim do módulo, BandidoTabular: UCB1 e Thompson sampling clássicos, sem ML.)

Três políticas, todas sobre o mesmo modelo de lucro(estado, preço)
(LightGBM com alvo em log, ver politicas.RegressorGrade):

  BandidoLGBM            offline e guloso: treina nos dados logados e escolhe o
                         preço de maior lucro previsto. É o que se compara com o CQL.
  BandidoEpsilonGreedy   em operação: com probabilidade ε testa um preço aleatório
                         da faixa; senão, o guloso. ε cai a cada lote.
  BandidoThompson        em operação: Thompson sampling por bootstrap. Mantém K
                         modelos, cada um treinado numa reamostragem do histórico;
                         cada decisão usa um modelo sorteado. Explora onde os
                         modelos discordam (onde há pouca informação).

As variantes em operação recebem o lucro observado (com ruído) de cada decisão
via atualizar() e reajustam o modelo a cada lote, como um produto real faria
(ex.: reajuste diário). A simulação está em simular_bandido_online.py.
"""

import zlib

import numpy as np
import pandas as pd

from politicas import ALVO, FRACAO_TREINO, Politica, RegressorGrade, chaves


class BandidoLGBM(RegressorGrade):
    """Alvo em log(lucro): os lucros vão de centenas (Low Ticket, orçamento 100) a
    milhões (High Ticket); em escala bruta o erro quadrático ignora os cenários
    pequenos. O lucro é positivo em todas as faixas, então o argmax não muda.
    (Decisão tomada depois da primeira avaliação; ver protocolo, seção 12.)"""
    nome = 'bandido'
    parametros = {'n_estimators': 300, 'learning_rate': 0.05, 'num_leaves': 63}
    escala_log = True


class BandidoOnline(BandidoLGBM):
    """Base das variantes que aprendem em operação.

    treinar(dados, semente)  aquecimento com os dados logados disponíveis no início
    precos(estados)          decisões do lote atual (com exploração)
    atualizar(estados, precos, lucros)  registra o resultado e reajusta o modelo
    """
    explicabilidade = 'alta: mesma curva de lucro do bandido; a exploração é registrada por decisão'

    def treinar(self, dados, semente, fracao_treino=1.0):
        self.semente = semente
        self.rng = np.random.default_rng(semente)
        self.lote = 0
        self.historico = dados.iloc[: int(len(dados) * fracao_treino)].reset_index(drop=True)
        self._ajustar()
        return self

    def atualizar(self, estados, precos, lucros):
        novos = estados.drop(columns='Cenario', errors='ignore').copy()
        novos['Preco_Amostra'] = np.asarray(precos, dtype=float)
        novos[ALVO[self.modelo]] = np.asarray(lucros, dtype=float)
        self.historico = pd.concat([self.historico, novos], ignore_index=True)
        self.lote += 1
        self._ajustar()

    def _ajustar(self):
        raise NotImplementedError

    def _preco_aleatorio(self, estados):
        return np.array([self.rng.uniform(lo, hi) for lo, hi in self.faixas(estados)])


class BandidoEpsilonGreedy(BandidoOnline):
    nome = 'bandido_eps'

    def __init__(self, modelo, cenarios, eps_inicial=0.2, decaimento=0.85, eps_minimo=0.02, **kw):
        super().__init__(modelo, cenarios, **kw)
        self.eps_inicial, self.decaimento, self.eps_minimo = eps_inicial, decaimento, eps_minimo

    @property
    def eps(self):
        return max(self.eps_minimo, self.eps_inicial * self.decaimento ** self.lote)

    def _ajustar(self):
        BandidoLGBM.treinar(self, self.historico, self.semente + self.lote, fracao_treino=1.0)

    def precos(self, estados):
        precos = super().precos(estados)
        explora = self.rng.random(len(estados)) < self.eps
        if explora.any():
            precos[explora] = self._preco_aleatorio(estados[explora])
        return precos


class BandidoThompson(BandidoOnline):
    nome = 'bandido_thompson'

    def __init__(self, modelo, cenarios, n_modelos=5, **kw):
        super().__init__(modelo, cenarios, **kw)
        self.n_modelos = n_modelos

    def _ajustar(self):
        self.modelos = []
        n = len(self.historico)
        for k in range(self.n_modelos):
            amostra = self.historico.iloc[self.rng.integers(0, n, n)]  # bootstrap
            m = BandidoLGBM(self.modelo, self.cenarios, n_grade=self.n_grade)
            self.modelos.append(m.treinar(amostra, self.semente + 1000 * self.lote + k, fracao_treino=1.0))

    def precos(self, estados):
        sorteio = self.rng.integers(0, self.n_modelos, len(estados))
        precos = np.empty(len(estados))
        for k in np.unique(sorteio):
            m = sorteio == k
            precos[m] = self.modelos[k].precos(estados[m])
        return precos

    def prever_lucro(self, estados, precos):
        return np.mean([m.prever_lucro(estados, precos) for m in self.modelos], axis=0)


class BandidoTabular(Politica):
    """Bandido clássico, sem aprendizado de máquina: uma tabela por estado observável.

    Cada estado observável (região, plataforma, Tier, orçamento e, na assinatura, a
    memória) tem K braços: preços igualmente espaçados em log na faixa observável do
    estado. Cada linha logada vai para o braço mais próximo do seu preço, e a tabela
    guarda contagem, média e variância do lucro por braço. O lucro entra dividido pela
    média do estado (escala comparável entre estados de centenas e de milhões).

      regra='ucb1'      índice média + c·sqrt(2 ln N / n) (Auer et al., 2002)
      regra='thompson'  amostra de N(média, variância/n) por braço (Thompson
                        sampling gaussiano), com semente fixa por estado

    Treinado nos dados logados e avaliado sem novas interações, como as demais
    políticas offline. UCB1 e Thompson foram feitos para aprender em operação; com a
    coleta uniforme (contagens parecidas em todos os braços) os dois ficam próximos
    da média empírica gulosa.
    """
    treinavel = True
    explicabilidade = 'total: tabela de lucro médio por estado e faixa de preço'

    def __init__(self, modelo, cenarios, regra='ucb1', n_bracos=10, c=1.0):
        super().__init__(modelo, cenarios)
        if regra not in ('ucb1', 'thompson'):
            raise ValueError(regra)
        self.regra, self.n_bracos, self.c = regra, n_bracos, c
        self.nome = 'ucb1' if regra == 'ucb1' else 'thompson_tab'

    def _chaves(self, estados):
        return [(r, p) + tuple(k) for r, p, k in zip(estados['Regiao'], estados['Plataforma'],
                                                      chaves(self.modelo, estados))]

    def _bracos(self, estados):
        return np.stack([np.geomspace(lo, hi, self.n_bracos) for lo, hi in self.faixas(estados)])

    def treinar(self, dados, semente, fracao_treino=FRACAO_TREINO):
        self.semente = semente
        treino = dados.iloc[: int(len(dados) * fracao_treino)]
        bracos = self._bracos(treino)
        braco = np.abs(np.log(bracos) - np.log(treino['Preco_Amostra'].to_numpy())[:, None]).argmin(axis=1)
        lucro = treino[ALVO[self.modelo]].to_numpy(dtype=float)
        df = pd.DataFrame({'estado': [str(k) for k in self._chaves(treino)], 'braco': braco, 'lucro': lucro})
        df['lucro'] /= df.groupby('estado')['lucro'].transform('mean')
        self.tabela = {e: g.groupby('braco')['lucro'].agg(['count', 'mean', 'var']).reindex(range(self.n_bracos))
                       for e, g in df.groupby('estado')}
        return self

    def precos(self, estados):
        bracos = self._bracos(estados)
        out = np.empty(len(estados))
        for i, chave in enumerate(self._chaves(estados)):
            t = self.tabela.get(str(chave))
            if t is None:  # estado nunca logado: meio da faixa
                out[i] = bracos[i, self.n_bracos // 2]
                continue
            n = t['count'].fillna(0).to_numpy()
            media = t['mean'].fillna(0).to_numpy()
            if self.regra == 'ucb1':
                indice = np.where(n > 0, media + self.c * np.sqrt(2 * np.log(max(n.sum(), 1)) / np.maximum(n, 1)),
                                  np.inf)
            else:
                rng = np.random.default_rng([self.semente, zlib.crc32(str(chave).encode())])
                var = t['var'].fillna(t['var'].mean()).to_numpy()
                indice = np.where(n > 1, rng.normal(media, np.sqrt(var / np.maximum(n, 1))), np.inf)
            out[i] = bracos[i, int(np.argmax(indice))]
        return out
