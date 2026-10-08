#!/usr/bin/env python3
"""
Simulação online curta: o bandido opera, observa o lucro e aprende.

Cenário: o produto começa com POUCOS dados logados (política de coleta
aleatória, --n-inicial amostras por cenário) e passa a precificar sozinho. Em
cada lote chegam --tamanho-lote campanhas (estados sorteados da grade de
avaliação); cada política escolhe os preços, o simulador devolve o lucro com
ruído e as políticas em operação reajustam o modelo.

Todas as políticas veem o MESMO fluxo de estados. Métrica de cada decisão:
lucro esperado do preço escolhido / lucro do oráculo naquele estado.

Políticas:
  aleatorio          continua coletando ao acaso (o que havia antes)
  offline_inicial    bandido treinado só nos dados iniciais e congelado
  bandido_eps        ε-greedy, aprende em operação (mesmo ponto de partida)
  bandido_thompson   Thompson por bootstrap, aprende em operação (mesmo ponto de partida)
  offline_completo   bandido treinado no dataset logado completo (48 mil linhas), congelado
  oraculo_obs        teto do que o estado permite

A diferença bandido_eps/thompson − offline_inicial é o ganho de aprender em
operação. offline_completo mostra quanto vale ter muitos dados logados antes.

Saídas (pasta --saida):
  bandido_online_curva.csv   % do ótimo por lote, política e semente
  bandido_online_resumo.csv  média e IC 95%: últimos lotes e média em operação
  bandido_online_comparacoes.csv  diferenças pareadas (IC 95%)
"""

import argparse
import os
import time

import numpy as np
import pandas as pd

import avaliar_politicas as av
import bandido as ban
import politicas as pol
import simulador as sim

COMPARACOES = [('bandido_eps', 'offline_inicial'), ('bandido_thompson', 'offline_inicial'),
               ('bandido_thompson', 'bandido_eps'), ('bandido_thompson', 'offline_completo'),
               ('bandido_eps', 'offline_completo')]


def politicas_da_rodada(modelo, cenarios, semente, n_inicial):
    iniciais = sim.gerar_dados(cenarios, n_inicial, semente)
    completos = sim.gerar_dados(cenarios, 5000, semente)
    pols = {
        'aleatorio': pol.Aleatoria(modelo, cenarios),
        'offline_inicial': ban.BandidoLGBM(modelo, cenarios),
        'bandido_eps': ban.BandidoEpsilonGreedy(modelo, cenarios),
        'bandido_thompson': ban.BandidoThompson(modelo, cenarios),
        'offline_completo': ban.BandidoLGBM(modelo, cenarios),
        'oraculo_obs': pol.OraculoObservavel(modelo, cenarios),
    }
    pols['aleatorio'].treinar(None, semente)
    pols['offline_inicial'].treinar(iniciais, semente, fracao_treino=1.0)
    pols['bandido_eps'].treinar(iniciais, semente)
    pols['bandido_thompson'].treinar(iniciais, semente)
    pols['offline_completo'].treinar(completos, semente)
    return pols


def rodar(modelo, cenarios, semente, args):
    grade = sim.contextos(cenarios)
    otimo = av.lucros(av.consultar(pol.Oraculo(modelo, cenarios), grade), grade, cenarios, modelo)
    pols = politicas_da_rodada(modelo, cenarios, semente, args.n_inicial)
    fluxo = np.random.default_rng([semente, 1]).integers(0, len(grade), (args.lotes, args.tamanho_lote))
    ruido = {nome: np.random.default_rng([semente, 2, i]) for i, nome in enumerate(pols)}

    linhas = []
    for lote, idx in enumerate(fluxo):
        estados = grade.iloc[idx].reset_index(drop=True)
        for nome, p in pols.items():
            precos = av.consultar(p, estados)
            esperado = av.lucros(precos, estados, cenarios, modelo)
            linhas.append({'modelo': modelo, 'politica': nome, 'semente': semente, 'lote': lote + 1,
                           'decisoes': (lote + 1) * args.tamanho_lote,
                           'pct_do_otimo': 100 * float(np.mean(esperado / otimo[idx]))})
            if isinstance(p, ban.BandidoOnline):
                lucros = [sim.amostrar_lucro(pr, cenarios[r.Cenario], r.Regiao, r.Plataforma, modelo, ruido[nome])
                          for pr, r in zip(precos, estados.itertuples())]
                p.atualizar(estados, precos, lucros)
    return linhas


def resumir(curva, ultimos):
    out = []
    final = curva[curva['lote'] > curva['lote'].max() - ultimos]
    for (modelo, politica), g in curva.groupby(['modelo', 'politica'], sort=False):
        f = final[(final['modelo'] == modelo) & (final['politica'] == politica)]
        fim, fim_lo, fim_hi = av.ic95(f.groupby('semente')['pct_do_otimo'].mean())
        op, op_lo, op_hi = av.ic95(g.groupby('semente')['pct_do_otimo'].mean())
        out.append({'modelo': modelo, 'politica': politica, 'n_sementes': g['semente'].nunique(),
                    'pct_final': fim, 'pct_final_ic95_inf': fim_lo, 'pct_final_ic95_sup': fim_hi,
                    'pct_em_operacao': op, 'pct_em_operacao_ic95_inf': op_lo, 'pct_em_operacao_ic95_sup': op_hi})
    return pd.DataFrame(out)


def comparar(curva, ultimos):
    out = []
    for janela, d in [('final', curva[curva['lote'] > curva['lote'].max() - ultimos]), ('em_operacao', curva)]:
        tab = d.groupby(['modelo', 'semente', 'politica'])['pct_do_otimo'].mean().unstack('politica')
        for a, b in COMPARACOES:
            for modelo, g in tab.groupby(level='modelo'):
                media, lo, hi = av.ic95(g[a] - g[b])
                out.append({'modelo': modelo, 'janela': janela, 'comparacao': f'{a} - {b}', 'diferenca_pp': media,
                            'ic95_inf': lo, 'ic95_sup': hi,
                            'conclusao': 'A > B' if lo > 0 else ('A < B' if hi < 0 else 'inconclusivo')})
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--sementes', type=int, default=5)
    ap.add_argument('--n-inicial', type=int, default=100, help='amostras logadas por cenário no início (100 x 12 = 1.200)')
    ap.add_argument('--lotes', type=int, default=20)
    ap.add_argument('--tamanho-lote', type=int, default=600)
    ap.add_argument('--ultimos', type=int, default=5, help='lotes que contam como desempenho final')
    ap.add_argument('--saida', default='resultados')
    args = ap.parse_args()

    cenarios = av.carregar_cenarios()
    linhas = []
    for semente in range(42, 42 + args.sementes):
        for modelo in av.MODELOS:
            t0 = time.perf_counter()
            linhas += rodar(modelo, cenarios, semente, args)
            print(f"semente {semente} {modelo:12s} ok ({time.perf_counter() - t0:.0f}s)")
    curva = pd.DataFrame(linhas)
    os.makedirs(args.saida, exist_ok=True)
    curva.to_csv(os.path.join(args.saida, 'bandido_online_curva.csv'), index=False)
    resumo = resumir(curva, args.ultimos)
    resumo.to_csv(os.path.join(args.saida, 'bandido_online_resumo.csv'), index=False)
    comparar(curva, args.ultimos).to_csv(os.path.join(args.saida, 'bandido_online_comparacoes.csv'), index=False)
    with pd.option_context('display.float_format', '{:,.1f}'.format, 'display.width', 160):
        print(resumo.to_string(index=False))


if __name__ == '__main__':
    main()
