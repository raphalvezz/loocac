#!/usr/bin/env python3
"""
Compara as abordagens no ambiente sequencial (sequencial.py): o preço de hoje
muda a reputação e a demanda de amanhã. É o teste que decide se o RL se
justifica: um método míope (bandido, SL) otimiza só o lucro do período; o CQL
com γ > 0 pode trocar lucro de hoje por reputação.

Políticas (venda única):
  aleatorio, meio_faixa       heurísticas
  sl, bandido                 regressor de lucro do período + grade (míopes por construção)
  bc                          imitação da coleta
  cql_g0                      CQL com γ = 0 (míope)
  cql_g095                    CQL com γ = 0,95 (planeja)
  oraculo_miope               conhece o cenário, cobra o ótimo de cada período
  oraculo_pd                  ótimo da programação dinâmica (referência = 100%)

Todas as não privilegiadas veem o contexto, a reputação atual e o período.
Métrica: lucro total esperado da campanha (T períodos) como % do ótimo, por tier.

Saídas: resultados/sequencial/{por_semente,tabela,comparacoes,trajetorias}.csv
Uso:    python avaliar_sequencial.py [--sementes 3] [--passos-cql 20000] [--sem-cql]
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

import avaliar_politicas as av
import bandido as ban
import politicas as pol
import sequencial as sq

COMPARACOES = [('cql_g095', 'cql_g0'), ('cql_g095', 'bandido'), ('cql_g095', 'oraculo_miope'),
               ('bandido', 'aleatorio'), ('cql_g0', 'bandido')]


def construir(cenarios, args, avaliador):
    m, ex = sq.MODELO, sq.EXTRAS
    pols = {
        'aleatorio': pol.Aleatoria(m, cenarios),
        'meio_faixa': pol.MeioFaixa(m, cenarios),
        'sl': pol.SL(m, cenarios, extras=ex),
        'bandido': ban.BandidoLGBM(m, cenarios, extras=ex),
    }
    if not args.sem_cql:
        pols['bc'] = pol.BC(m, cenarios, extras=ex)
        for nome, gamma in [('cql_g0', 0.0), ('cql_g095', 0.95)]:
            # recompensa por contexto (lucro / média do contexto - 1): constante dentro do
            # episódio, então não muda a política ótima; evita a escala de centenas a milhões
            pols[nome] = pol.CQL(m, cenarios, n_passos=args.passos_cql, gamma=gamma, extras=ex,
                                 escala_recompensa='estado', avaliador=avaliador)
    pols['oraculo_miope'] = sq.OraculoMiope(cenarios)
    pols['oraculo_pd'] = sq.OraculoPD(cenarios)
    return pols


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--sementes', type=int, default=3)
    ap.add_argument('--episodios', type=int, default=200, help='campanhas logadas por cenário (múltiplo de 5)')
    ap.add_argument('--passos-cql', type=int, default=20000)
    ap.add_argument('--sem-cql', action='store_true')
    ap.add_argument('--saida', default='resultados/sequencial')
    args = ap.parse_args()

    cenarios = av.carregar_cenarios()
    avaliador = sq.Avaliador(cenarios)
    baixo = (avaliador.contextos['Tier'] == 'Low Ticket').to_numpy()
    os.makedirs(args.saida, exist_ok=True)
    linhas, trajetorias = [], []
    for semente in range(42, 42 + args.sementes):
        dados = sq.gerar_dados(cenarios, args.episodios, semente)
        for nome, p in construir(cenarios, args, avaliador).items():
            t0 = time.perf_counter()
            if hasattr(p, 'treinar'):
                p.treinar(dados, semente)
            treino = time.perf_counter() - t0
            total, precos = avaliador.simular(lambda e: av.consultar(p, e))
            pct = 100 * total / avaliador.otimo
            for tier, m in [('Low Ticket', baixo), ('High Ticket', ~baixo), ('Todos', np.ones_like(baixo))]:
                linhas.append({'politica': nome, 'semente': semente, 'tier': tier,
                               'pct_do_otimo': float(pct[m].mean()), 'tempo_treino_s': treino})
            # trajetória média do preço relativo ao preço míope ótimo (mostra se a política planeja)
            rel = precos / np.array([sq.preco_referencia(cenarios[r.Cenario], r.Regiao, r.Plataforma) / sq.LAMBDA
                                     for r in avaliador.contextos.itertuples()])[None, :]
            trajetorias += [{'politica': nome, 'semente': semente, 'periodo': t, 'preco_rel_miope': float(rel[t].mean())}
                            for t in range(sq.T)]
            print(f"  semente {semente} {nome:14s} {pct.mean():6.1f}% do ótimo ({treino:.0f}s)", flush=True)
            pd.DataFrame(linhas).to_csv(os.path.join(args.saida, 'por_semente.csv'), index=False)

    det = pd.DataFrame(linhas)
    tabela = []
    for (nome, tier), g in det.groupby(['politica', 'tier'], sort=False):
        m, lo, hi = av.ic95(g['pct_do_otimo'])
        tabela.append({'politica': nome, 'tier': tier, 'n_sementes': len(g), 'pct_do_otimo': m,
                       'ic95_inf': lo, 'ic95_sup': hi, 'tempo_treino_s': g['tempo_treino_s'].mean()})
    tabela = pd.DataFrame(tabela)
    tabela.to_csv(os.path.join(args.saida, 'tabela.csv'), index=False)
    pd.DataFrame(trajetorias).to_csv(os.path.join(args.saida, 'trajetorias.csv'), index=False)

    comp = []
    piv = det.pivot_table(index=['tier', 'semente'], columns='politica', values='pct_do_otimo')
    for a, b in COMPARACOES:
        if a in piv and b in piv:
            for tier, g in piv.groupby(level='tier'):
                m, lo, hi = av.ic95(g[a] - g[b])
                comp.append({'tier': tier, 'comparacao': f'{a} - {b}', 'diferenca_pp': m, 'ic95_inf': lo,
                             'ic95_sup': hi, 'conclusao': 'A > B' if lo > 0 else ('A < B' if hi < 0 else 'inconclusivo')})
    pd.DataFrame(comp).to_csv(os.path.join(args.saida, 'comparacoes.csv'), index=False)
    with open(os.path.join(args.saida, 'manifesto.json'), 'w') as f:
        json.dump({'T': sq.T, 'kappa': sq.KAPPA, 'lambda': sq.LAMBDA, 'sementes': args.sementes,
                   'episodios_por_cenario': args.episodios, 'passos_cql': None if args.sem_cql else args.passos_cql},
                  f, indent=2)
    with pd.option_context('display.float_format', '{:,.1f}'.format, 'display.width', 140):
        print(tabela.pivot_table(index='politica', columns='tier', values='pct_do_otimo', sort=False).to_string())
        print(pd.DataFrame(comp).to_string(index=False))


if __name__ == '__main__':
    main()
