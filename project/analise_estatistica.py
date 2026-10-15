#!/usr/bin/env python3
"""
Análises estatísticas sobre os resultados por semente já gravados (não treina nada).

1. Correção para múltiplas comparações (Holm e Bonferroni) dos critérios C1–C3
   (comparação principal) e S1–S2 (ambiente sequencial).
2. Poder: desvio-padrão entre sementes das diferenças pareadas e o menor efeito
   detectável com 5 e 10 sementes; sementes necessárias para detectar 0,5 p.p.
3. Margem do C2: a escala de −1 p.p. frente à variação observada entre sementes.
4. Distância exata do CQL γ = 0,95 ao ótimo da programação dinâmica.

Testes (pareados por semente, t de Student, unilaterais):
  C1  cql − aleatorio > 0            S1  cql_g095 − bandido > 0
  C2  cql − bandido   > −1 p.p.      S2  cql_g095 − cql_g0  > 0
  C3  cql − meio_faixa > 0
O critério do protocolo (limite inferior do IC 95% bilateral acima do limiar) é o
mesmo que um teste unilateral a 2,5%; as correções usam α de família = 2,5%.
Duas famílias: as 10 hipóteses primárias (5 critérios × 2 cobranças, no total) e as
30 com os recortes por tier.

Entradas: resultados/final_10_sementes/por_semente.csv, resultados/sequencial/por_semente.csv
Saídas:   resultados/estatistica/{testes,poder,distancia_oraculo}.csv
Uso:      python analise_estatistica.py
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats

ALFA = 0.025  # unilateral, equivalente ao IC 95% bilateral do protocolo
HIPOTESES = [  # (nome, arquivo, política A, política B, limiar)
    ('C1', 'estatico', 'cql', 'aleatorio', 0.0),
    ('C2', 'estatico', 'cql', 'bandido', -1.0),
    ('C3', 'estatico', 'cql', 'meio_faixa', 0.0),
    ('S1', 'sequencial', 'cql_g095', 'bandido', 0.0),
    ('S2', 'sequencial', 'cql_g095', 'cql_g0', 0.0),
]


def teste(dif, limiar):
    n = len(dif)
    media, dp = float(np.mean(dif)), float(np.std(dif, ddof=1))
    t = (media - limiar) / (dp / np.sqrt(n))
    return media, dp, n, float(t), float(stats.t.sf(t, n - 1))


def holm(p, alfa):
    """Rejeições pelo procedimento de Holm (passo a passo, do menor p ao maior)."""
    ordem = np.argsort(p)
    rejeita = np.zeros(len(p), dtype=bool)
    for k, i in enumerate(ordem):
        if p[i] <= alfa / (len(p) - k):
            rejeita[i] = True
        else:
            break
    return rejeita


def efeito_minimo(dp, n, alfa=ALFA, poder=0.8):
    """Menor diferença média detectável (teste t pareado unilateral)."""
    return (stats.t.ppf(1 - alfa, n - 1) + stats.t.ppf(poder, n - 1)) * dp / np.sqrt(n)


def sementes_para(efeito, dp, alfa=ALFA, poder=0.8):
    for n in range(3, 1000):
        if efeito_minimo(dp, n, alfa, poder) <= efeito:
            return n
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--estatico', default='resultados/final_10_sementes/por_semente.csv')
    ap.add_argument('--sequencial', default='resultados/sequencial/por_semente.csv')
    ap.add_argument('--saida', default='resultados/estatistica')
    args = ap.parse_args()

    dados = {'estatico': pd.read_csv(args.estatico), 'sequencial': pd.read_csv(args.sequencial)}
    piv = {k: d.pivot_table(index=['modelo', 'tier', 'semente'], columns='politica', values='pct_do_otimo')
           for k, d in dados.items()}

    linhas, poder = [], []
    for nome, fonte, a, b, limiar in HIPOTESES:
        for (modelo, tier), g in piv[fonte].groupby(level=['modelo', 'tier']):
            dif = (g[a] - g[b]).to_numpy()
            media, dp, n, t, p = teste(dif, limiar)
            linhas.append({'hipotese': nome, 'modelo': modelo, 'tier': tier, 'comparacao': f'{a} - {b}',
                           'limiar_pp': limiar, 'n_sementes': n, 'diferenca_pp': media, 'dp_entre_sementes': dp,
                           't': t, 'p_unilateral': p})
            if tier == 'Todos':
                poder.append({'hipotese': nome, 'modelo': modelo, 'comparacao': f'{a} - {b}', 'dp_entre_sementes': dp,
                              'efeito_minimo_5_sementes': efeito_minimo(dp, 5),
                              'efeito_minimo_10_sementes': efeito_minimo(dp, 10),
                              'sementes_para_0_5_pp': sementes_para(0.5, dp),
                              'distancia_ao_limiar_pp': media - limiar})
    tab = pd.DataFrame(linhas)
    tab['sem_correcao'] = tab['p_unilateral'] <= ALFA
    for familia, mascara in [('primarias', tab['tier'] == 'Todos'), ('todas', np.ones(len(tab), dtype=bool))]:
        sub = tab[mascara]
        m = len(sub)
        tab.loc[mascara, f'bonferroni_{familia}'] = sub['p_unilateral'] <= ALFA / m
        tab.loc[mascara, f'holm_{familia}'] = holm(sub['p_unilateral'].to_numpy(), ALFA)
        tab.loc[mascara, f'p_bonferroni_{familia}'] = np.minimum(sub['p_unilateral'] * m, 1.0)

    # distância exata ao ótimo da programação dinâmica (100%) no sequencial
    dist = []
    for (modelo, tier), g in piv['sequencial'].groupby(level=['modelo', 'tier']):
        for pol in ('cql_g095', 'oraculo_miope', 'bandido'):
            gap = 100 - g[pol].to_numpy()
            meia = stats.t.ppf(0.975, len(gap) - 1) * gap.std(ddof=1) / np.sqrt(len(gap))
            dist.append({'modelo': modelo, 'tier': tier, 'politica': pol, 'n_sementes': len(gap),
                         'pct_do_otimo': 100 - gap.mean(), 'distancia_pp': gap.mean(),
                         'ic95_inf': gap.mean() - meia, 'ic95_sup': gap.mean() + meia,
                         'min_pp': gap.min(), 'max_pp': gap.max()})

    os.makedirs(args.saida, exist_ok=True)
    tab.to_csv(os.path.join(args.saida, 'testes.csv'), index=False)
    pd.DataFrame(poder).to_csv(os.path.join(args.saida, 'poder.csv'), index=False)
    pd.DataFrame(dist).to_csv(os.path.join(args.saida, 'distancia_oraculo.csv'), index=False)
    with pd.option_context('display.width', 220, 'display.float_format', '{:,.4g}'.format):
        print(tab[['hipotese', 'modelo', 'tier', 'diferenca_pp', 'p_unilateral', 'sem_correcao',
                   'holm_primarias', 'holm_todas', 'bonferroni_todas']].to_string(index=False))
        print(pd.DataFrame(poder).to_string(index=False))
        print(pd.DataFrame(dist).to_string(index=False))


if __name__ == '__main__':
    main()
