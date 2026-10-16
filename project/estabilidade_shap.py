#!/usr/bin/env python3
"""
Estabilidade do SHAP do SL entre sementes.

Para cada semente: gera o dataset logado (simulador.gerar_dados), treina o SL
(politicas.SL, como no SL_FINAL) e calcula o SHAP (TreeExplainer) em 2.000 linhas
do teste, exatamente como o notebook. A importância é a média de |SHAP| por coluna
e também por variável original (as colunas do one-hot somadas: Regiao, Plataforma,
Tier...).

Compara entre as sementes: posição de cada variável no ranking, Kendall τ médio
entre pares de sementes, se as 3 mais importantes são as mesmas, e o coeficiente
de variação da participação de cada variável na importância total.

Saídas: resultados/shap/{importancia_por_semente,estabilidade}.csv
Uso:    python estabilidade_shap.py [--sementes 5]
"""

import argparse
import itertools
import os

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

import avaliar_politicas as av
import politicas as pol
import simulador as sim


def variavel(coluna):
    for prefixo in pol.CATEGORICAS:
        if coluna.startswith(prefixo + '_'):
            return prefixo
    return coluna


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--sementes', type=int, default=5)
    ap.add_argument('--amostra', type=int, default=2000)
    ap.add_argument('--saida', default='resultados/shap')
    args = ap.parse_args()
    import shap

    cenarios = av.carregar_cenarios()
    linhas = []
    for semente in range(42, 42 + args.sementes):
        dados = sim.gerar_dados(cenarios, 5000, semente)
        teste = dados.iloc[int(len(dados) * pol.FRACAO_TREINO):]
        amostra = teste.sample(args.amostra, random_state=semente)
        for modelo in av.MODELOS:
            sl = pol.SL(modelo, cenarios).treinar(dados, semente)
            X = sl._X(amostra, amostra['Preco_Amostra'].to_numpy())
            valores = np.abs(shap.TreeExplainer(sl.lgb).shap_values(X)).mean(axis=0)
            for coluna, v in zip(sl.nomes_features(), valores):
                linhas.append({'modelo': modelo, 'semente': semente, 'coluna': coluna,
                               'variavel': variavel(coluna), 'media_abs_shap': float(v)})
            print(f'  s{semente} {modelo:12s} ok', flush=True)

    det = pd.DataFrame(linhas)
    os.makedirs(args.saida, exist_ok=True)
    det.to_csv(os.path.join(args.saida, 'importancia_por_semente.csv'), index=False)

    saida = []
    for nivel in ('variavel', 'coluna'):
        imp = det.groupby(['modelo', 'semente', nivel])['media_abs_shap'].sum().reset_index()
        for modelo, g in imp.groupby('modelo', sort=False):
            tab = g.pivot_table(index=nivel, columns='semente', values='media_abs_shap').fillna(0)
            tab = tab[tab.sum(axis=1) > 0]
            participacao = tab / tab.sum(axis=0)
            ranks = tab.rank(ascending=False)
            taus = [kendalltau(tab[a], tab[b]).statistic for a, b in itertools.combinations(tab.columns, 2)]
            top3 = [tuple(tab[s].nlargest(3).index) for s in tab.columns]
            for nome in tab.index:
                saida.append({'nivel': nivel, 'modelo': modelo, 'nome': nome,
                              'participacao_media': participacao.loc[nome].mean(),
                              'participacao_cv': participacao.loc[nome].std() / participacao.loc[nome].mean(),
                              'posicao_media': ranks.loc[nome].mean(),
                              'posicao_min': int(ranks.loc[nome].min()), 'posicao_max': int(ranks.loc[nome].max()),
                              'kendall_tau_medio': float(np.mean(taus)), 'kendall_tau_min': float(np.min(taus)),
                              'top3_igual_em_todas': len(set(top3)) == 1})
    est = pd.DataFrame(saida).sort_values(['nivel', 'modelo', 'posicao_media'])
    est.to_csv(os.path.join(args.saida, 'estabilidade.csv'), index=False)
    with pd.option_context('display.width', 200, 'display.float_format', '{:,.3f}'.format, 'display.max_rows', 200):
        print(est[est['nivel'] == 'variavel'].to_string(index=False))
        print(est[(est['nivel'] == 'coluna') & (est['posicao_media'] <= 6)].to_string(index=False))


if __name__ == '__main__':
    main()
