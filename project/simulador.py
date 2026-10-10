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
import pandas as pd

# ----------------------------------------------------------------------------
# SIMULADOR CONGELADO (ver docs/protocolo_avaliacao.md)
# Qualquer mudança na economia abaixo muda os resultados do capítulo 5. Só se
# altera para corrigir um erro: nesse caso, suba VERSAO, registre o motivo no
# protocolo e atualize os valores de referência em tests/test_simulador.py.
# ----------------------------------------------------------------------------
VERSAO = "1.0"

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

# Campos do estado que não variam nos dados (ceteris paribus)
CATEGORICAS_FIXAS = {'Idade': '25-34', 'Genero': 'Female', 'Conteudo': 'Video',
                     'Tipo_Produto': 'InfoProduto', 'Modelo_Cobranca': 'Venda Unica',
                     'Complexidade_Oferta': 'Media'}

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

# Forma funcional da demanda. 'logistica' é o simulador v1.0 (padrão, congelado).
# 'linear' é a forma alternativa do teste de robustez (avaliar_robustez.py):
# D(p) = c0 · max(0, 1 − x/x_c), com o preço de saturação x_c por contexto calibrado
# para o ótimo cair em x* = 1/m (dentro da faixa). Venda única: lucro parabólico,
# x_c = 2x* − CPA/a0. Assinatura: x_c = x*(2 − 0,4x*)/(1 − 0,4x*) (anula a derivada
# de (1 − x/x_c)·x·e^(−0,4x), ignorando o CPA). Uma isoelástica foi descartada antes
# de qualquer treino: a curva de lucro ficava quase plana (meio da faixa ~98% do
# ótimo), e o teste não distinguiria as políticas. Não muda nada quando não é usada.
FORMAS_DEMANDA = ('logistica', 'linear')
FORMA_DEMANDA = 'logistica'

def usar_forma(forma):
    """Troca a forma da demanda para todo o simulador (geração, oráculos, avaliação)."""
    global FORMA_DEMANDA
    if forma not in FORMAS_DEMANDA:
        raise ValueError(f"Forma desconhecida: {forma}")
    FORMA_DEMANDA = forma

def _demanda_linear(x, c0, cenario, regiao, plataforma, modelo):
    a0, _ = _base(cenario)
    x_otimo = 1.0 / multiplicador_elasticidade(regiao, plataforma)
    if modelo == 'venda_unica':
        x_sat = 2 * x_otimo - cenario['CPA_Target'] / a0
    else:
        k = ELASTICIDADE_CHURN
        x_sat = x_otimo * (2 - k * x_otimo) / (1 - k * x_otimo)
    return c0 * np.maximum(0.0, 1 - x / x_sat)

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
    if FORMA_DEMANDA == 'linear':
        demanda = _demanda_linear(x, c0, cenario, regiao, plataforma, modelo)
    else:
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


# ============================================================================
# 4. Dados e contextos de avaliação
# ============================================================================
def gerar_dados(cenarios, n_por_cenario=5000, semente=SEED):
    """Dataset logado: política de coleta uniforme na faixa de cada cenário.

    Com semente=SEED reproduz exatamente o dataset do Generator_NEW.py.
    As linhas saem embaralhadas (o split 80/20 sequencial dos notebooks depende disso).
    """
    rng = np.random.default_rng(semente)
    linhas = []
    for idx, cenario in enumerate(cenarios):
        for _ in range(n_por_cenario):
            regiao = rng.choice(REGIOES)
            plataforma = rng.choice(PLATAFORMAS)
            preco = rng.uniform(cenario['Price_Min'], cenario['Price_Max'])
            linhas.append({
                'Cenario': idx,
                'Regiao': regiao,
                'Plataforma': plataforma,
                'Tier': cenario['Tier'],
                'Orcamento': cenario['Budget'],
                **CATEGORICAS_FIXAS,
                **memoria_assinatura(cenario),
                'Preco_Amostra': preco,
                'Lucro_Real': amostrar_lucro(preco, cenario, regiao, plataforma, 'venda_unica', rng),
                'LTV_Real': amostrar_lucro(preco, cenario, regiao, plataforma, 'assinatura', rng),
            })
    return pd.DataFrame(linhas).sample(frac=1.0, random_state=semente).reset_index(drop=True)

def contextos(cenarios):
    """Grade de avaliação: cada cenário x região x plataforma (144 estados na tabela original).

    A coluna 'Cenario' identifica o mundo verdadeiro; só políticas privilegiadas
    (oráculos) podem lê-la.
    """
    linhas = []
    for idx, c in enumerate(cenarios):
        for regiao in REGIOES:
            for plataforma in PLATAFORMAS:
                linhas.append({'Cenario': idx, 'Regiao': regiao, 'Plataforma': plataforma,
                               'Tier': c['Tier'], 'Orcamento': c['Budget'],
                               **CATEGORICAS_FIXAS, **memoria_assinatura(c)})
    return pd.DataFrame(linhas)
