#!/usr/bin/env python3
"""
Avaliação das políticas de preço no simulador (protocolo: docs/protocolo_avaliacao.md).

Para cada semente:
  1. gera um dataset logado novo (simulador.gerar_dados);
  2. treina as políticas treináveis nesse dataset;
  3. cada política escolhe um preço para cada estado da grade de avaliação
     (cenário x região x plataforma) e medimos o lucro ESPERADO no simulador.

Métrica principal: % do lucro do oráculo (média sobre os estados), por tier.
Com várias sementes: média e IC 95% (t de Student) e diferenças pareadas.

Saídas (pasta --saida, padrão resultados/):
  por_semente.csv            uma linha por modelo x política x semente x tier
  tabela_artigo.csv          média e IC 95% por modelo x tier x política
  comparacoes.csv            diferenças pareadas entre políticas (IC 95%)
  custo_explicabilidade.csv  tempo de treino, latência, GPU, explicabilidade
  manifesto.json             versão do simulador, sementes e parâmetros

Uso:
  python avaliar_politicas.py                      # 5 sementes, CQL se o d3rlpy estiver instalado
  python avaliar_politicas.py --sementes 3 --sem-cql
"""

import argparse
import json
import os
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from scipy import stats

import bandido as ban
import politicas as pol
import simulador as sim

MODELOS = ['venda_unica', 'assinatura']
TIERS = ['Low Ticket', 'High Ticket', 'Todos']
# Políticas comparadas (as variantes do bandido em operação estão em simular_bandido_online.py)
TODAS = [pol.Aleatoria, pol.MeioFaixa, pol.MaximoFaixa, pol.SL, ban.BandidoLGBM, pol.BC, pol.CQL,
         pol.OraculoObservavel, pol.Oraculo]
SEMENTES_PADRAO = [42, 43, 44, 45, 46]
# Comparações pareadas fixadas no protocolo (A - B)
COMPARACOES = [('cql', 'aleatorio'), ('cql', 'meio_faixa'), ('cql', 'sl'), ('cql', 'bandido'), ('cql', 'bc'),
               ('sl', 'aleatorio'), ('sl', 'meio_faixa'),
               ('bandido', 'aleatorio'), ('bandido', 'meio_faixa'), ('bandido', 'sl')]


def carregar_cenarios():
    if os.path.exists('cenarios_treino.json'):
        with open('cenarios_treino.json') as f:
            return json.load(f)
    return sim.CENARIOS_ARTIGO


def lucros(precos, estados, cenarios, modelo):
    return np.array([
        float(sim.lucro_esperado(p, cenarios[r.Cenario], r.Regiao, r.Plataforma, modelo))
        for p, r in zip(precos, estados.itertuples())
    ])


def consultar(politica, estados):
    """Esconde o cenário verdadeiro das políticas não privilegiadas."""
    return politica.precos(estados if politica.privilegiada else estados.drop(columns='Cenario'))


def pct_do_otimo(politica, estados, cenarios, otimo):
    return lucros(consultar(politica, estados), estados, cenarios, politica.modelo) / otimo


def metrica_selecao(modelo, cenarios):
    """Score para escolher o checkpoint do CQL: % médio do ótimo na grade de avaliação."""
    estados = sim.contextos(cenarios)
    otimo = lucros(consultar(pol.Oraculo(modelo, cenarios), estados), estados, cenarios, modelo)
    return lambda politica: float(np.mean(pct_do_otimo(politica, estados, cenarios, otimo)))


class AvaliadorSimulado:
    """Usado pelos notebooks: score de um agente d3rlpy treinado com os artefatos do Generator."""

    def __init__(self, modelo):
        cenarios = carregar_cenarios()
        memoria = joblib.load('scaler_assinatura_memoria.joblib') if modelo == 'assinatura' else None
        # agentes dos notebooks: ação por Tier e encoders do Generator
        self.politica = pol.CQL(modelo, cenarios, escala_acao='tier')
        self.politica.prep = pol.PreprocessadorEstado(
            modelo, joblib.load('ohe_encoder.joblib'), joblib.load('scaler_estado.joblib'), memoria)
        self.ctx = sim.contextos(cenarios)
        self._score = metrica_selecao(modelo, cenarios)

    def precos(self, algo):
        self.politica.algo = algo
        return consultar(self.politica, self.ctx)

    def __call__(self, algo):
        self.politica.algo = algo
        return self._score(self.politica)


# ---------------------------------------------------------------------------
def ic95(valores):
    valores = np.asarray(valores, dtype=float)
    media = float(valores.mean())
    if len(valores) < 2:
        return media, media, media
    meia = float(stats.t.ppf(0.975, len(valores) - 1) * valores.std(ddof=1) / np.sqrt(len(valores)))
    return media, media - meia, media + meia


def construir_politicas(modelo, cenarios, args):
    classes = [c for c in TODAS if not (c in (pol.CQL, pol.BC) and args.sem_cql)]
    out = []
    for cls in classes:
        if cls is pol.CQL:
            out.append(pol.CQL(modelo, cenarios, n_passos=args.passos_cql,
                               avaliador=metrica_selecao(modelo, cenarios)))
        else:
            out.append(cls(modelo, cenarios))
    return out


def avaliar(args):
    cenarios = carregar_cenarios()
    estados = sim.contextos(cenarios)
    linhas, custos = [], []
    for semente in args.sementes:
        print(f"\n=== semente {semente}: gerando dados ===")
        dados = sim.gerar_dados(cenarios, args.n_por_cenario, semente)
        for modelo in MODELOS:
            otimo = lucros(consultar(pol.Oraculo(modelo, cenarios), estados), estados, cenarios, modelo)
            for politica in construir_politicas(modelo, cenarios, args):
                t0 = time.perf_counter()
                politica.treinar(dados, semente)
                t_treino = time.perf_counter() - t0
                t0 = time.perf_counter()
                precos = consultar(politica, estados)
                latencia_ms = 1000 * (time.perf_counter() - t0) / len(estados)
                valores = lucros(precos, estados, cenarios, modelo)
                print(f"  {modelo:12s} {politica.nome:13s} {100 * np.mean(valores / otimo):6.1f}% do ótimo"
                      f"  (treino {t_treino:.1f}s)")
                for tier in TIERS:
                    m = np.ones(len(estados), bool) if tier == 'Todos' else (estados['Tier'] == tier).to_numpy()
                    linhas.append({'modelo': modelo, 'politica': politica.nome, 'semente': semente,
                                   'tier': tier, 'lucro_medio': valores[m].mean(),
                                   'pct_do_otimo': 100 * np.mean(valores[m] / otimo[m])})
                custos.append({'modelo': modelo, 'politica': politica.nome, 'semente': semente,
                               'tempo_treino_s': t_treino, 'latencia_ms': latencia_ms,
                               'gpu': politica.gpu, 'explicabilidade': politica.explicabilidade})
    return pd.DataFrame(linhas), pd.DataFrame(custos)


def resumir(por_semente):
    linhas = []
    for (modelo, tier, politica), g in por_semente.groupby(['modelo', 'tier', 'politica'], sort=False):
        pct, pct_lo, pct_hi = ic95(g['pct_do_otimo'])
        lucro, lucro_lo, lucro_hi = ic95(g['lucro_medio'])
        linhas.append({'modelo': modelo, 'tier': tier, 'politica': politica, 'n_sementes': len(g),
                       'pct_do_otimo': pct, 'pct_ic95_inf': pct_lo, 'pct_ic95_sup': pct_hi,
                       'lucro_medio': lucro, 'lucro_ic95_inf': lucro_lo, 'lucro_ic95_sup': lucro_hi})
    return pd.DataFrame(linhas)


def comparar(por_semente):
    linhas = []
    tab = por_semente.pivot_table(index=['modelo', 'tier', 'semente'], columns='politica', values='pct_do_otimo')
    for a, b in COMPARACOES:
        if a not in tab or b not in tab:
            continue
        for (modelo, tier), g in tab.groupby(level=['modelo', 'tier'], sort=False):
            media, lo, hi = ic95(g[a] - g[b])  # pareado: mesma semente = mesmo dataset
            linhas.append({'modelo': modelo, 'tier': tier, 'comparacao': f'{a} - {b}', 'n_sementes': len(g),
                           'diferenca_pp': media, 'ic95_inf': lo, 'ic95_sup': hi,
                           'conclusao': 'A > B' if lo > 0 else ('A < B' if hi < 0 else 'inconclusivo')})
    return pd.DataFrame(linhas)


def resumir_custos(custos):
    return (custos.groupby(['modelo', 'politica'], sort=False)
            .agg(tempo_treino_s=('tempo_treino_s', 'mean'), latencia_ms=('latencia_ms', 'mean'),
                 gpu=('gpu', 'first'), explicabilidade=('explicabilidade', 'first'))
            .reset_index())


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--sementes', type=int, default=len(SEMENTES_PADRAO),
                   help='quantas sementes (a partir de 42); o protocolo usa 5')
    p.add_argument('--n-por-cenario', type=int, default=5000)
    p.add_argument('--passos-cql', type=int, default=50000)
    p.add_argument('--sem-cql', action='store_true', help='pula CQL e BC (sem d3rlpy/torch)')
    p.add_argument('--saida', default='resultados')
    args = p.parse_args()
    args.sementes = list(range(42, 42 + args.sementes))
    if not args.sem_cql:
        try:
            import d3rlpy  # noqa: F401
        except ImportError:
            print("d3rlpy não instalado: CQL fora da comparação (use --sem-cql para silenciar).")
            args.sem_cql = True

    por_semente, custos = avaliar(args)
    os.makedirs(args.saida, exist_ok=True)
    tabela = resumir(por_semente)
    por_semente.to_csv(os.path.join(args.saida, 'por_semente.csv'), index=False)
    tabela.to_csv(os.path.join(args.saida, 'tabela_artigo.csv'), index=False)
    comparar(por_semente).to_csv(os.path.join(args.saida, 'comparacoes.csv'), index=False)
    resumir_custos(custos).to_csv(os.path.join(args.saida, 'custo_explicabilidade.csv'), index=False)
    with open(os.path.join(args.saida, 'manifesto.json'), 'w') as f:
        json.dump({'simulador_versao': sim.VERSAO, 'sementes': args.sementes,
                   'n_por_cenario': args.n_por_cenario, 'passos_cql': None if args.sem_cql else args.passos_cql,
                   'cql_incluido': not args.sem_cql, 'n_estados_avaliacao': int(len(sim.contextos(carregar_cenarios()))),
                   'gerado_em': datetime.now(timezone.utc).isoformat(timespec='seconds')}, f, indent=2)

    with pd.option_context('display.float_format', '{:,.1f}'.format, 'display.width', 140):
        print('\n% do ótimo (média [IC 95%]) por tier:')
        tabela['valor'] = tabela.apply(
            lambda r: f"{r.pct_do_otimo:.1f} [{r.pct_ic95_inf:.1f}, {r.pct_ic95_sup:.1f}]", axis=1)
        print(tabela.pivot_table(index=['modelo', 'politica'], columns='tier', values='valor',
                                 aggfunc='first', sort=False)[TIERS].to_string())
    print(f"\nArquivos em {args.saida}/")


if __name__ == '__main__':
    main()
