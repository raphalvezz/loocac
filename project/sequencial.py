"""
Ambiente sequencial: o preço de hoje afeta o cliente de amanhã (reputação).

Cada campanha dura T períodos. Além do contexto (cenário, região, plataforma),
o estado tem a reputação ρ_t da marca naquele público:

    lucro_t   = ρ_t · lucro_estático(p_t)          (simulador.lucro_esperado, venda única)
    ρ_{t+1}   = clip(ρ_t + κ · (p_ref − p_t) / p_ref, ρ_min, ρ_max)
    p_ref     = λ · p*_estático(contexto)          (o "preço justo" percebido pelo público)

Com λ < 1, cobrar o preço míope ótimo (p*) corrói a reputação e reduz a demanda
futura; cobrar abaixo de p_ref constrói reputação. Há, portanto, um conflito
entre o lucro de hoje e o de amanhã — o único cenário em que planejar (RL com
γ > 0) tem motivo real para superar um bandido míope.

O ótimo exato vem de programação dinâmica (reputação discretizada); a métrica é o
lucro total esperado da campanha como % desse ótimo, a partir de ρ_0 = 1, nos
144 contextos de simulador.contextos.

O módulo NÃO altera o simulador v1.0: reaproveita a economia estática dele.
"""

import numpy as np
import pandas as pd

import simulador as sim

T = 12                    # períodos por campanha
# Calibração (escolhida antes de treinar qualquer política): com κ = 0,15 e λ = 0,85
# o oráculo míope obtém ~90% do ótimo, ou seja, planejar vale ~10 p.p.; κ = 0,5
# deixava o míope em 55% e a PD presa no preço mínimo (dinâmica exagerada).
KAPPA = 0.15              # velocidade de ajuste da reputação
LAMBDA = 0.85             # preço justo = 85% do preço míope ótimo
RHO_MIN, RHO_MAX, RHO_0 = 0.5, 1.5, 1.0
MODELO = 'venda_unica'
EXTRAS = ('reputacao', 'periodo')   # colunas extras do estado para as políticas
N_RHO, N_PRECO = 201, 200          # grades da programação dinâmica


def preco_referencia(cenario, regiao, plataforma):
    return LAMBDA * sim.preco_otimo(cenario, regiao, plataforma, MODELO)[0]


def proxima_reputacao(rho, preco, p_ref):
    return np.clip(rho + KAPPA * (p_ref - preco) / p_ref, RHO_MIN, RHO_MAX)


def lucro(rho, preco, cenario, regiao, plataforma):
    return rho * sim.lucro_esperado(preco, cenario, regiao, plataforma, MODELO)


# ---------------------------------------------------------------------------
# Ótimo por programação dinâmica
# ---------------------------------------------------------------------------
def programacao_dinamica(cenario, regiao, plataforma, faixa=None):
    """Valor ótimo V_0(ρ_0) e a política (preço por período e reputação).

    `faixa` limita os preços (padrão: faixa do cenário)."""
    lo, hi = faixa if faixa is not None else (cenario['Price_Min'], cenario['Price_Max'])
    rhos = np.linspace(RHO_MIN, RHO_MAX, N_RHO)
    precos = np.geomspace(lo, hi, N_PRECO)
    p_ref = preco_referencia(cenario, regiao, plataforma)
    estatico = sim.lucro_esperado(precos, cenario, regiao, plataforma, MODELO)      # (P,)
    prox = proxima_reputacao(rhos[:, None], precos[None, :], p_ref)                 # (R, P)
    V = np.zeros(N_RHO)
    politica = np.zeros((T, N_RHO))
    for t in reversed(range(T)):
        Q = rhos[:, None] * estatico[None, :] + np.interp(prox, rhos, V)
        politica[t] = precos[Q.argmax(axis=1)]
        V = Q.max(axis=1)
    return float(np.interp(RHO_0, rhos, V)), (rhos, politica)


# ---------------------------------------------------------------------------
# Dados logados (política de coleta aleatória) e estados
# ---------------------------------------------------------------------------
def gerar_dados(cenarios, episodios_por_cenario=200, semente=sim.SEED):
    """Campanhas de T períodos com preço uniforme na faixa do cenário.

    Os episódios saem embaralhados e com tamanho fixo, então o corte 80/20 das
    políticas (FRACAO_TREINO) cai sempre entre episódios quando o total de
    episódios é múltiplo de 5."""
    rng = np.random.default_rng(semente)
    linhas, ep = [], 0
    for idx, c in enumerate(cenarios):
        for _ in range(episodios_por_cenario):
            regiao, plataforma = rng.choice(sim.REGIOES), rng.choice(sim.PLATAFORMAS)
            p_ref = preco_referencia(c, regiao, plataforma)
            rho = RHO_0
            for t in range(T):
                preco = rng.uniform(c['Price_Min'], c['Price_Max'])
                linhas.append({
                    'episodio': ep, 'passo': t, 'Cenario': idx, 'Regiao': regiao, 'Plataforma': plataforma,
                    'Tier': c['Tier'], 'Orcamento': c['Budget'], **sim.CATEGORICAS_FIXAS,
                    **sim.memoria_assinatura(c), 'reputacao': rho, 'periodo': t / T,
                    'Preco_Amostra': preco,
                    'Lucro_Real': rho * sim.amostrar_lucro(preco, c, regiao, plataforma, MODELO, rng),
                    'LTV_Real': 0.0,
                })
                rho = float(proxima_reputacao(rho, preco, p_ref))
            ep += 1
    df = pd.DataFrame(linhas)
    ordem = np.random.default_rng(semente).permutation(ep)
    df['_ordem'] = ordem[df['episodio']]
    return df.sort_values(['_ordem', 'passo']).drop(columns='_ordem').reset_index(drop=True)


# ---------------------------------------------------------------------------
# Avaliação: simula a campanha inteira para cada contexto
# ---------------------------------------------------------------------------
class Avaliador:
    """Lucro total esperado de uma política nos 144 contextos, como % do ótimo (PD)."""

    def __init__(self, cenarios):
        self.cenarios = cenarios
        self.contextos = sim.contextos(cenarios)
        self.p_ref = np.array([preco_referencia(cenarios[r.Cenario], r.Regiao, r.Plataforma)
                               for r in self.contextos.itertuples()])
        self.otimo = np.array([programacao_dinamica(cenarios[r.Cenario], r.Regiao, r.Plataforma)[0]
                               for r in self.contextos.itertuples()])

    def simular(self, escolher):
        """`escolher(estados) -> preços`; devolve (lucro total por contexto, trajetória de preços)."""
        rho = np.full(len(self.contextos), RHO_0)
        total = np.zeros(len(self.contextos))
        precos_t = []
        for t in range(T):
            estados = self.contextos.assign(reputacao=rho, periodo=t / T)
            precos = np.asarray(escolher(estados), dtype=float)
            total += np.array([lucro(r_, p, self.cenarios[c.Cenario], c.Regiao, c.Plataforma)
                               for r_, p, c in zip(rho, precos, self.contextos.itertuples())])
            rho = proxima_reputacao(rho, precos, self.p_ref)
            precos_t.append(precos)
        return total, np.array(precos_t)

    def pct(self, politica):
        import avaliar_politicas as av
        total, _ = self.simular(lambda e: av.consultar(politica, e))
        return 100 * total / self.otimo

    def __call__(self, politica):
        """Métrica de seleção de checkpoint (fração do ótimo, média dos contextos)."""
        return float(np.mean(self.pct(politica)) / 100)


# ---------------------------------------------------------------------------
# Referências privilegiadas
# ---------------------------------------------------------------------------
class OraculoMiope:
    """Conhece o cenário e cobra, a cada período, o preço míope ótimo (p*)."""
    nome, privilegiada = 'oraculo_miope', True

    def __init__(self, cenarios):
        self.cenarios = cenarios

    def precos(self, estados):
        return np.array([sim.preco_otimo(self.cenarios[r.Cenario], r.Regiao, r.Plataforma, MODELO)[0]
                         for r in estados.itertuples()])


class OraculoPD:
    """Política ótima da programação dinâmica (conhece o cenário e a dinâmica)."""
    nome, privilegiada = 'oraculo_pd', True

    def __init__(self, cenarios):
        self.cenarios = cenarios
        self._cache = {}

    def precos(self, estados):
        out = []
        for r in estados.itertuples():
            chave = (r.Cenario, r.Regiao, r.Plataforma)
            if chave not in self._cache:
                self._cache[chave] = programacao_dinamica(self.cenarios[r.Cenario], r.Regiao, r.Plataforma)[1]
            rhos, politica = self._cache[chave]
            t = min(int(round(r.periodo * T)), T - 1)
            out.append(float(np.interp(r.reputacao, rhos, politica[t])))
        return np.array(out)
