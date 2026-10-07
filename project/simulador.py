"""
Simulador econômico (Gêmeo Digital) compartilhado por:
  - Generator_NEW.py      (gera os dados de treino)
  - notebooks de RL       (seleção de checkpoint pelo lucro simulado)
  - avaliar_politicas.py  (comparação com baselines)
  - main.py               (conversão ação <-> preço)

Formulação: cada decisão de preço é independente (bandido contextual de um
passo). O lucro tem um ótimo INTERIOR em cada faixa de preço, para que haja
algo a aprender além de "cobrar o máximo".

Demanda: curva logística de disposição a pagar (WTP). O centro da WTP depende
do contexto (Região x Plataforma), então o preço ótimo também depende.
"""

import numpy as np

SEED = 42

# ============================================================================
# 1. Cenários do artigo (tabela)
# ============================================================================
CENARIOS_ARTIGO = [
    # --- LOW TICKET (CPA alvo ~25% a 30% do preço médio) ---
    {'Tier': 'Low Ticket', 'Price_Min': 10.0, 'Price_Max': 20.0, 'Budget': 100.0, 'CPA_Target': 5.0},
    {'Tier': 'Low Ticket', 'Price_Min': 50.0, 'Price_Max': 100.0, 'Budget': 200.0, 'CPA_Target': 20.0},
    {'Tier': 'Low Ticket', 'Price_Min': 20.0, 'Price_Max': 50.0, 'Budget': 1000.0, 'CPA_Target': 10.0},
    {'Tier': 'Low Ticket', 'Price_Min': 50.0, 'Price_Max': 100.0, 'Budget': 1000.0, 'CPA_Target': 22.0},
    {'Tier': 'Low Ticket', 'Price_Min': 20.0, 'Price_Max': 50.0, 'Budget': 10000.0, 'CPA_Target': 12.0},
    {'Tier': 'Low Ticket', 'Price_Min': 50.0, 'Price_Max': 100.0, 'Budget': 10000.0, 'CPA_Target': 25.0},
    # --- HIGH TICKET (CPA alvo ~15% a 25% do preço médio) ---
    {'Tier': 'High Ticket', 'Price_Min': 500.0, 'Price_Max': 1000.0, 'Budget': 2000.0, 'CPA_Target': 150.0},
    {'Tier': 'High Ticket', 'Price_Min': 1000.0, 'Price_Max': 2000.0, 'Budget': 5000.0, 'CPA_Target': 350.0},
    {'Tier': 'High Ticket', 'Price_Min': 2000.0, 'Price_Max': 5000.0, 'Budget': 10000.0, 'CPA_Target': 800.0},
    {'Tier': 'High Ticket', 'Price_Min': 2000.0, 'Price_Max': 5000.0, 'Budget': 50000.0, 'CPA_Target': 900.0},
    {'Tier': 'High Ticket', 'Price_Min': 5000.0, 'Price_Max': 10000.0, 'Budget': 100000.0, 'CPA_Target': 1800.0},
    {'Tier': 'High Ticket', 'Price_Min': 10000.0, 'Price_Max': 25000.0, 'Budget': 200000.0, 'CPA_Target': 4000.0},
]

REGIOES = ['North America', 'Europe', 'Asia', 'South America']
PLATAFORMAS = ['Instagram', 'Facebook', 'LinkedIn']

# Sensibilidade a preço por contexto: divide o centro da WTP.
# > 1 = público mais sensível a preço -> preço ótimo mais baixo.
ELASTICIDADE_REGIAO = {'North America': 0.90, 'Europe': 1.00, 'Asia': 1.10, 'South America': 1.10}
ELASTICIDADE_PLATAFORMA = {'Instagram': 1.00, 'Facebook': 1.05, 'LinkedIn': 0.90}

# Ruído multiplicativo nas conversões (lognormal com média 1)
SIGMA_RUIDO = 0.20

# Inclinação da curva de WTP (em unidades de p/a0). Maior = erro de preço custa mais.
# Com 6, o ponto médio da faixa fica em ~80% do lucro ótimo e o máximo em ~45%.
INCLINACAO_WTP = 6.0
# Centro da WTP (em p/a0) para multiplicador 1. Calibrado para o ótimo cair, em
# média, no meio da faixa do cenário e nunca na borda.
CENTRO_WTP = {'venda_unica': 1.20, 'assinatura': 1.35}

# Assinatura: churn mensal no preço médio da faixa e sua sensibilidade ao preço
CHURN_BASE = 0.05
ELASTICIDADE_CHURN = 0.4

# Features de memória da assinatura (mock, iguais para todo o dataset)
def memoria_assinatura(cenario):
    return {
        'dias_desde_ultima_interacao': 30,
        'clv_estimate_percentile': 0.5,
        'avg_price_offered_segment_90d': cenario['Price_Min'],
        'price_volatility_30d': 1.0,
    }

# ============================================================================
# 2. Economia
# ============================================================================
def multiplicador_elasticidade(regiao, plataforma):
    return ELASTICIDADE_REGIAO[regiao] * ELASTICIDADE_PLATAFORMA[plataforma]

def _base(cenario):
    a0 = (cenario['Price_Min'] + cenario['Price_Max']) / 2  # preço de referência
    c0 = 100.0 * np.log1p(cenario['Budget']) / np.log1p(1000)  # demanda base ~ orçamento
    return a0, c0

def lucro_esperado(preco, cenario, regiao, plataforma, modelo='venda_unica'):
    """Lucro esperado (sem ruído). Aceita preço escalar ou array.

    x = p/a0, w = CENTRO_WTP[modelo] / m (m = multiplicador do contexto)
    Demanda:      D(p) = 2 c0 / (1 + exp(INCLINACAO_WTP (x - w)))      (D = c0 em x = w)
    Venda única:  lucro = D(p) * (p - CPA)
    Assinatura:   churn(p) = 0.05 exp(0.4 (x - 1));  LTV = D(p) * (p / churn(p) - CPA)
    """
    if modelo not in CENTRO_WTP:
        raise ValueError(f"Modelo desconhecido: {modelo}")
    preco = np.asarray(preco, dtype=float)
    a0, c0 = _base(cenario)
    cpa = cenario['CPA_Target']
    x = preco / a0
    w = CENTRO_WTP[modelo] / multiplicador_elasticidade(regiao, plataforma)
    demanda = 2 * c0 / (1 + np.exp(INCLINACAO_WTP * (x - w)))
    if modelo == 'venda_unica':
        return demanda * (preco - cpa)
    churn = np.clip(CHURN_BASE * np.exp(ELASTICIDADE_CHURN * (x - 1)), 0.01, 0.95)
    return demanda * (preco / churn - cpa)

def amostrar_lucro(preco, cenario, regiao, plataforma, modelo, rng):
    """Lucro observado: esperado x ruído lognormal de média 1 nas conversões."""
    ruido = rng.lognormal(-SIGMA_RUIDO ** 2 / 2, SIGMA_RUIDO)
    return float(lucro_esperado(preco, cenario, regiao, plataforma, modelo) * ruido)

def faixa_observavel(cenario, cenarios):
    """Faixa de preço que um agente consegue associar ao estado (Tier, Orçamento).

    Cenários com mesmo Tier e orçamento (3/4 e 5/6) são indistinguíveis no estado
    da venda única, então a faixa observável é a união das faixas deles.
    """
    grupo = [c for c in cenarios if c['Tier'] == cenario['Tier'] and c['Budget'] == cenario['Budget']]
    return min(c['Price_Min'] for c in grupo), max(c['Price_Max'] for c in grupo)

def preco_otimo(cenario, regiao, plataforma, modelo='venda_unica', faixa=None, n=2001):
    """Preço que maximiza o lucro esperado (busca em grade). Por padrão, na faixa do cenário."""
    lo, hi = faixa if faixa is not None else (cenario['Price_Min'], cenario['Price_Max'])
    grade = np.geomspace(lo, hi, n)
    lucros = lucro_esperado(grade, cenario, regiao, plataforma, modelo)
    i = int(np.argmax(lucros))
    return float(grade[i]), float(lucros[i])

# ============================================================================
# 3. Ação normalizada (min-max por tier, em escala log)
# ============================================================================
# O ator do CQL usa tanh -> ação em [-1, 1]. Mapeamos [-1, 1] para a faixa
# de preços do Tier do estado. A escala log é necessária porque o High Ticket
# vai de 500 a 25.000: na escala linear, o cenário 500-1000 ocuparia só 2% do
# intervalo da ação.
def faixas_tier(cenarios):
    faixas = {}
    for c in cenarios:
        lo, hi = faixas.get(c['Tier'], (np.inf, -np.inf))
        faixas[c['Tier']] = (min(lo, c['Price_Min']), max(hi, c['Price_Max']))
    return {t: [float(lo), float(hi)] for t, (lo, hi) in faixas.items()}

def preco_para_acao(preco, tier, faixas):
    lo, hi = np.log(faixas[tier][0]), np.log(faixas[tier][1])
    return 2 * (np.log(preco) - lo) / (hi - lo) - 1

def acao_para_preco(acao, tier, faixas):
    lo, hi = np.log(faixas[tier][0]), np.log(faixas[tier][1])
    acao = np.clip(acao, -1.0, 1.0)
    return np.exp(lo + (acao + 1) / 2 * (hi - lo))
