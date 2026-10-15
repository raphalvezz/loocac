#!/usr/bin/env python3
"""
Compara as abordagens no ambiente sequencial (sequencial.py): o preço de hoje
muda a reputação e a demanda de amanhã. É o teste que decide se o RL se
justifica: um método míope (bandido, SL) otimiza só o lucro do período; o CQL
com γ > 0 pode trocar lucro de hoje por reputação.

Políticas (para cada cobrança: venda única e assinatura):
  aleatorio, meio_faixa       heurísticas
  sl, bandido                 regressor de lucro do período + grade (míopes por construção)
  bc                          imitação da coleta
  cql_g0                      CQL com γ = 0 (míope)
  cql_g095                    CQL com γ = 0,95 (planeja)
  oraculo_miope               conhece o cenário, cobra o ótimo de cada período
  oraculo_pd                  ótimo da programação dinâmica (referência = 100%)

Todas as não privilegiadas veem o contexto, a reputação atual e o período.
Métrica: lucro total esperado da campanha (T períodos) como % do ótimo, por
cobrança e por tier.

Saídas: resultados/sequencial/{por_semente,tabela,tabela_legivel,comparacoes,trajetorias}.csv
Uso:    python avaliar_sequencial.py [--sementes 10] [--passos-cql 50000] [--modelos venda_unica assinatura]
        # em paralelo: uma semente por processo e depois juntar
        python avaliar_sequencial.py --semente-inicial 44 --sementes 1 --saida tmp/s44
        python avaliar_sequencial.py --agregar tmp/s42 tmp/s43 ... --saida resultados/sequencial
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

import avaliar_politicas as av
import bandido as ban
import cenarios_sensibilidade as cs
import politicas as pol
import sequencial as sq

COMPARACOES = [('cql_g095', 'cql_g0'), ('cql_g095', 'bandido'), ('cql_g095', 'oraculo_miope'),
               ('bandido', 'aleatorio'), ('cql_g0', 'bandido'),
               ('iql_g095', 'cql_g095'), ('iql_g095', 'bandido'), ('iql_g095', 'cql_g0')]
OPCIONAIS = ('iql_g095',)  # só com --politicas
TIERS = ['Low Ticket', 'High Ticket', 'Todos']


def construir(modelo, cenarios, args, avaliador):
    ex = sq.EXTRAS
    pols = {
        'aleatorio': pol.Aleatoria(modelo, cenarios),
        'meio_faixa': pol.MeioFaixa(modelo, cenarios),
        'sl': pol.SL(modelo, cenarios, extras=ex),
        'bandido': ban.BandidoLGBM(modelo, cenarios, extras=ex),
    }
    if not args.sem_cql:
        pols['bc'] = pol.BC(modelo, cenarios, extras=ex)
        for nome, gamma in [('cql_g0', 0.0), ('cql_g095', 0.95)]:
            # recompensa por contexto (lucro / média do contexto - 1): constante dentro do
            # episódio, então não muda a política ótima; evita a escala de centenas a milhões
            pols[nome] = pol.CQL(modelo, cenarios, n_passos=args.passos_cql, gamma=gamma, extras=ex,
                                 escala_recompensa='estado', avaliador=avaliador)
    if not args.sem_cql and 'iql_g095' in (args.politicas or ()):
        pols['iql_g095'] = pol.IQL(modelo, cenarios, n_passos=args.passos_cql, gamma=0.95, extras=ex,
                                   escala_recompensa='estado', avaliador=avaliador)
    pols['oraculo_miope'] = sq.OraculoMiope(cenarios, modelo)
    pols['oraculo_pd'] = sq.OraculoPD(cenarios, modelo)
    if args.politicas:  # só as pedidas, sem treinar as outras (oráculos sempre)
        pols = {n: p for n, p in pols.items() if n in args.politicas or n.startswith('oraculo')}
    return pols


def rodar(args, cenarios):
    linhas, trajetorias = [], []
    for modelo in args.modelos:
        avaliador = sq.Avaliador(cenarios, modelo)
        baixo = (avaliador.contextos['Tier'] == 'Low Ticket').to_numpy()
        miope = np.array([sq.preco_referencia(cenarios[r.Cenario], r.Regiao, r.Plataforma, modelo) / sq.LAMBDA
                          for r in avaliador.contextos.itertuples()])
        for semente in range(args.semente_inicial, args.semente_inicial + args.sementes):
            dados = sq.gerar_dados(cenarios, args.episodios, semente, modelo)
            for nome, p in construir(modelo, cenarios, args, avaliador).items():
                t0 = time.perf_counter()
                if hasattr(p, 'treinar'):
                    p.treinar(dados, semente)
                treino = time.perf_counter() - t0
                total, precos = avaliador.simular(lambda e: av.consultar(p, e))
                pct = 100 * total / avaliador.otimo
                for tier, m in zip(TIERS, [baixo, ~baixo, np.ones_like(baixo)]):
                    linhas.append({'modelo': modelo, 'politica': nome, 'semente': semente, 'tier': tier,
                                   'pct_do_otimo': float(pct[m].mean()), 'tempo_treino_s': treino})
                # trajetória média do preço relativo ao preço míope ótimo (mostra se a política planeja)
                rel = precos / miope[None, :]
                trajetorias += [{'modelo': modelo, 'politica': nome, 'semente': semente, 'periodo': t,
                                 'preco_rel_miope': float(rel[t].mean())} for t in range(sq.T)]
                print(f"  {modelo:12s} semente {semente} {nome:14s} {pct.mean():6.1f}% do ótimo ({treino:.0f}s)",
                      flush=True)
                pd.DataFrame(linhas).to_csv(os.path.join(args.saida, 'por_semente.csv'), index=False)
                pd.DataFrame(trajetorias).to_csv(os.path.join(args.saida, 'trajetorias.csv'), index=False)
    return pd.DataFrame(linhas), pd.DataFrame(trajetorias)


def resumir(det, saida):
    """Tabela com IC 95% por cobrança, política e tier; versão legível; comparações pareadas."""
    tabela = []
    for (modelo, nome, tier), g in det.groupby(['modelo', 'politica', 'tier'], sort=False):
        m, lo, hi = av.ic95(g['pct_do_otimo'])
        tabela.append({'modelo': modelo, 'politica': nome, 'tier': tier, 'n_sementes': len(g),
                       'pct_do_otimo': m, 'ic95_inf': lo, 'ic95_sup': hi,
                       'tempo_treino_s': g['tempo_treino_s'].mean()})
    tabela = pd.DataFrame(tabela)
    tabela.to_csv(os.path.join(saida, 'tabela.csv'), index=False)

    # uma linha por política, uma coluna por (cobrança, tier): "média [IC inf; IC sup]"
    legivel = tabela.assign(celula=[f"{m:.1f} [{lo:.1f}; {hi:.1f}]" if n > 1 else f"{m:.1f}"
                                    for m, lo, hi, n in zip(tabela['pct_do_otimo'], tabela['ic95_inf'],
                                                            tabela['ic95_sup'], tabela['n_sementes'])])
    legivel = legivel.pivot_table(index='politica', columns=['modelo', 'tier'], values='celula',
                                  aggfunc='first', sort=False)
    legivel.columns = [f'{m} | {t}' for m, t in legivel.columns]
    legivel.to_csv(os.path.join(saida, 'tabela_legivel.csv'))

    comp = []
    for modelo, dm in det.groupby('modelo', sort=False):
        piv = dm.pivot_table(index=['tier', 'semente'], columns='politica', values='pct_do_otimo')
        for a, b in COMPARACOES:
            if a in piv and b in piv:
                for tier in TIERS:
                    g = piv.xs(tier, level='tier')[[a, b]].dropna()
                    if len(g) < 2:
                        continue
                    m, lo, hi = av.ic95(g[a] - g[b])
                    comp.append({'modelo': modelo, 'tier': tier, 'comparacao': f'{a} - {b}',
                                 'n_sementes': len(g), 'diferenca_pp': m, 'ic95_inf': lo, 'ic95_sup': hi,
                                 'conclusao': 'A > B' if lo > 0 else ('A < B' if hi < 0 else 'inconclusivo')})
    comp = pd.DataFrame(comp)
    comp.to_csv(os.path.join(saida, 'comparacoes.csv'), index=False)
    with pd.option_context('display.width', 200, 'display.max_columns', 20):
        print(legivel.to_string())
        if len(comp):
            print(comp.round(2).to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--sementes', type=int, default=10)
    ap.add_argument('--semente-inicial', type=int, default=42)
    ap.add_argument('--modelos', nargs='+', default=list(sq.MODELOS), choices=sq.MODELOS)
    ap.add_argument('--episodios', type=int, default=200, help='campanhas logadas por cenário (múltiplo de 5)')
    ap.add_argument('--passos-cql', type=int, default=50000)
    ap.add_argument('--sem-cql', action='store_true')
    ap.add_argument('--agregar', nargs='+', metavar='PASTA',
                    help='só junta por_semente.csv/trajetorias.csv de rodadas anteriores')
    ap.add_argument('--saida', default='resultados/sequencial')
    ap.add_argument('--cenarios', choices=cs.CONJUNTOS, help='conjunto de cenários (cenarios_sensibilidade.py)')
    ap.add_argument('--kappa', type=float, help=f'κ (padrão {sq.KAPPA})')
    ap.add_argument('--lambda', dest='lam', type=float, help=f'λ (padrão {sq.LAMBDA})')
    ap.add_argument('--politicas', nargs='+', help=f'só estas políticas; opcionais: {OPCIONAIS}')
    args = ap.parse_args()
    os.makedirs(args.saida, exist_ok=True)
    if args.kappa is not None:
        sq.KAPPA = args.kappa
    if args.lam is not None:
        sq.LAMBDA = args.lam

    if args.agregar:
        det = pd.concat([pd.read_csv(os.path.join(d, 'por_semente.csv')) for d in args.agregar], ignore_index=True)
        if det.duplicated(['modelo', 'politica', 'semente', 'tier']).any():
            raise SystemExit('a mesma semente aparece em mais de uma pasta')
        traj = pd.concat([pd.read_csv(os.path.join(d, 'trajetorias.csv')) for d in args.agregar], ignore_index=True)
        manifestos = [json.load(open(os.path.join(d, 'manifesto.json'))) for d in args.agregar]
        passos = sorted({m['passos_cql'] for m in manifestos}, key=str)
        if len(passos) > 1:
            raise SystemExit(f'rodadas com passos_cql diferentes: {passos}')
        passos = passos[0]
        det.to_csv(os.path.join(args.saida, 'por_semente.csv'), index=False)
        traj.to_csv(os.path.join(args.saida, 'trajetorias.csv'), index=False)
    else:
        det, traj = rodar(args, cs.conjunto(args.cenarios) if args.cenarios else av.carregar_cenarios())
        passos = None if args.sem_cql else args.passos_cql

    resumir(det, args.saida)
    with open(os.path.join(args.saida, 'manifesto.json'), 'w') as f:
        json.dump({'T': sq.T, 'kappa': sq.KAPPA, 'lambda': sq.LAMBDA, 'cenarios': args.cenarios or 'padrao',
                   'politicas': args.politicas,
                   'modelos': sorted(det['modelo'].unique().tolist()),
                   'sementes': sorted(int(x) for x in det['semente'].unique()),
                   'episodios_por_cenario': args.episodios, 'passos_cql': passos}, f, indent=2)


if __name__ == '__main__':
    main()
