#!/usr/bin/env python3
"""
Generalização dos regressores de lucro (SL e bandido) para contextos fora do treino.

O R² do SL_FINAL é medido num split aleatório da mesma distribuição. Aqui o
modelo é treinado SEM um grupo inteiro e avaliado só nesse grupo:

  - in_dist   split 80/20 habitual (referência)
  - regiao    deixa uma região de fora (4 rodadas); a categoria nunca vista vira
              um one-hot nulo
  - cenario   deixa um cenário de fora (12 rodadas); o modelo nunca viu aquela
              combinação de orçamento e faixa de preço

Métricas no grupo de fora: R² do lucro previsto (US$) nas linhas logadas e % do
ótimo da política (preço escolhido na grade) nos estados do grupo.

Saídas: resultados/generalizacao.csv (por grupo) e resultados/generalizacao_resumo.csv
Uso:    python avaliar_generalizacao.py [--sementes 3]
"""

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

import avaliar_politicas as av
import politicas as pol
import simulador as sim

POLITICAS = [pol.SL, pol.BandidoLGBM]


def pct_otimo(politica, estados, cenarios, modelo):
    otimo = av.lucros(av.consultar(pol.Oraculo(modelo, cenarios), estados), estados, cenarios, modelo)
    return 100 * float(np.mean(av.lucros(av.consultar(politica, estados), estados, cenarios, modelo) / otimo))


def r2(politica, linhas, modelo):
    return float(r2_score(linhas[pol.ALVO[modelo]], politica.prever_lucro(linhas, linhas['Preco_Amostra'].to_numpy())))


def rodar(cenarios, semente, n_por_cenario):
    dados = sim.gerar_dados(cenarios, n_por_cenario, semente)
    estados = sim.contextos(cenarios)
    corte = int(len(dados) * pol.FRACAO_TREINO)
    out = []
    for modelo in av.MODELOS:
        for cls in POLITICAS:
            p = cls(modelo, cenarios).treinar(dados, semente)
            out.append({'modelo': modelo, 'politica': p.nome, 'semente': semente, 'tipo': 'in_dist', 'grupo': 'todos',
                        'r2': r2(p, dados.iloc[corte:], modelo),
                        'pct_do_otimo': pct_otimo(p, estados, cenarios, modelo)})
            grupos = [('regiao', 'Regiao', r) for r in sim.REGIOES] + \
                     [('cenario', 'Cenario', k) for k in range(len(cenarios))]
            for tipo, coluna, valor in grupos:
                fora = dados[coluna] == valor
                p = cls(modelo, cenarios).treinar(dados[~fora], semente, fracao_treino=1.0)
                out.append({'modelo': modelo, 'politica': p.nome, 'semente': semente, 'tipo': tipo,
                            'grupo': str(valor), 'r2': r2(p, dados[fora], modelo),
                            'pct_do_otimo': pct_otimo(p, estados[estados[coluna] == valor], cenarios, modelo)})
            print(f"  semente {semente} {modelo:12s} {cls.nome}: ok")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--sementes', type=int, default=1)
    ap.add_argument('--n-por-cenario', type=int, default=5000)
    ap.add_argument('--saida', default='resultados')
    args = ap.parse_args()

    cenarios = av.carregar_cenarios()
    linhas = []
    for semente in range(42, 42 + args.sementes):
        linhas += rodar(cenarios, semente, args.n_por_cenario)
    det = pd.DataFrame(linhas)
    resumo = (det.groupby(['modelo', 'politica', 'tipo'], sort=False)
              .agg(r2_medio=('r2', 'mean'), r2_min=('r2', 'min'),
                   pct_do_otimo_medio=('pct_do_otimo', 'mean'), pct_do_otimo_min=('pct_do_otimo', 'min'))
              .reset_index())
    os.makedirs(args.saida, exist_ok=True)
    det.to_csv(os.path.join(args.saida, 'generalizacao.csv'), index=False)
    resumo.to_csv(os.path.join(args.saida, 'generalizacao_resumo.csv'), index=False)
    with pd.option_context('display.float_format', '{:,.3f}'.format, 'display.width', 140):
        print(resumo.to_string(index=False))


if __name__ == '__main__':
    main()
