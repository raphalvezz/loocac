#!/usr/bin/env python3
"""
Teste de deslocamento de público: treinar com um tipo de público e avaliar em outro.

Origem (treino):   North America e Europe   (multiplicador de elasticidade 0,9 e 1,0)
Destino (teste):   Asia e South America     (1,1: público mais sensível a preço)

Os dados de treino têm só campanhas da origem. Cada política é avaliada nos
estados da origem (referência) e nos do destino (deslocamento), em % do ótimo.
As regiões do destino nunca aparecem no treino: para os modelos, viram uma
categoria desconhecida (one-hot nulo). O checkpoint do CQL é escolhido só com
os estados da origem, sem olhar o destino.

Saídas: resultados/deslocamento/{por_semente,tabela}.csv
Uso:    python avaliar_deslocamento.py [--sementes 3] [--passos-cql 20000] [--sem-cql]
"""

import argparse
import os

import numpy as np
import pandas as pd

import avaliar_politicas as av
import bandido as ban
import politicas as pol
import simulador as sim

ORIGEM = ['North America', 'Europe']
DESTINO = ['Asia', 'South America']


def metrica(modelo, cenarios, estados):
    otimo = av.lucros(av.consultar(pol.Oraculo(modelo, cenarios), estados), estados, cenarios, modelo)
    return lambda p: float(np.mean(av.lucros(av.consultar(p, estados), estados, cenarios, modelo) / otimo))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--sementes', type=int, default=3)
    ap.add_argument('--passos-cql', type=int, default=20000)
    ap.add_argument('--sem-cql', action='store_true')
    ap.add_argument('--saida', default='resultados/deslocamento')
    args = ap.parse_args()

    cenarios = av.carregar_cenarios()
    todos = sim.contextos(cenarios)
    grupos = {'origem': todos[todos['Regiao'].isin(ORIGEM)].reset_index(drop=True),
              'destino': todos[todos['Regiao'].isin(DESTINO)].reset_index(drop=True)}
    os.makedirs(args.saida, exist_ok=True)
    linhas = []
    for semente in range(42, 42 + args.sementes):
        dados = sim.gerar_dados(cenarios, 5000, semente)
        dados = dados[dados['Regiao'].isin(ORIGEM)].reset_index(drop=True)
        for modelo in av.MODELOS:
            pols = [pol.Aleatoria(modelo, cenarios), pol.MeioFaixa(modelo, cenarios),
                    pol.SL(modelo, cenarios), ban.BandidoLGBM(modelo, cenarios)]
            if not args.sem_cql:
                pols += [pol.BC(modelo, cenarios),
                         pol.CQL(modelo, cenarios, n_passos=args.passos_cql,
                                 avaliador=metrica(modelo, cenarios, grupos['origem']))]
            pols.append(pol.OraculoObservavel(modelo, cenarios))
            for p in pols:
                p.treinar(dados, semente)
                for grupo, estados in grupos.items():
                    pct = 100 * metrica(modelo, cenarios, estados)(p)
                    linhas.append({'modelo': modelo, 'politica': p.nome, 'semente': semente,
                                   'avaliado_em': grupo, 'pct_do_otimo': pct})
                print(f"  s{semente} {modelo:12s} {p.nome:12s} origem {linhas[-2]['pct_do_otimo']:5.1f}%  "
                      f"destino {linhas[-1]['pct_do_otimo']:5.1f}%", flush=True)
                pd.DataFrame(linhas).to_csv(os.path.join(args.saida, 'por_semente.csv'), index=False)

    det = pd.DataFrame(linhas)
    tabela = []
    for (modelo, nome), g in det.groupby(['modelo', 'politica'], sort=False):
        linha = {'modelo': modelo, 'politica': nome, 'n_sementes': g['semente'].nunique()}
        for grupo in grupos:
            m, lo, hi = av.ic95(g[g['avaliado_em'] == grupo]['pct_do_otimo'])
            linha.update({f'pct_{grupo}': m, f'{grupo}_ic95_inf': lo, f'{grupo}_ic95_sup': hi})
        # queda pareada por semente (origem - destino)
        piv = g.pivot_table(index='semente', columns='avaliado_em', values='pct_do_otimo')
        m, lo, hi = av.ic95(piv['origem'] - piv['destino'])
        linha.update({'queda_pp': m, 'queda_ic95_inf': lo, 'queda_ic95_sup': hi})
        tabela.append(linha)
    tabela = pd.DataFrame(tabela)
    tabela.to_csv(os.path.join(args.saida, 'tabela.csv'), index=False)
    with pd.option_context('display.float_format', '{:,.1f}'.format, 'display.width', 140):
        print(tabela[['modelo', 'politica', 'pct_origem', 'pct_destino', 'queda_pp']].to_string(index=False))


if __name__ == '__main__':
    main()
