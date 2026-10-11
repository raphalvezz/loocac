#!/usr/bin/env python3
"""
Ablações curtas do CQL (seção 5.6): o que muda o desempenho e o diagnóstico do
Low Ticket (preço quase constante).

Configurações (uma alteração por vez em relação à base = configuração dos notebooks):
  base                 γ = 0, w = 5, ação por Tier, recompensa padronizada globalmente
  w_0.5 / w_20         peso conservador (conservative_weight)
  acao_global          ação [-1, 1] = faixa de TODOS os preços (sem escala por Tier)
  gamma_0.98           γ > 0 com transições independentes (como nos notebooks antigos)
  recompensa_log       log(lucro) padronizado
  recompensa_estado    lucro / lucro médio da chave de estado - 1
  acao_estado          ação [-1, 1] = faixa observável do estado (correção do diagnóstico)
  acao_estado+recompensa_estado   as duas juntas

Treino curto (--passos, padrão 5.000) com seleção do checkpoint pelo lucro
simulado, como no treino completo. Métricas: % do ótimo por tier e, no Low
Ticket, a dispersão dos preços escolhidos e a correlação (log) com o preço do
oráculo observável — se a política só repete um preço, a dispersão é ~0.

Saídas: resultados/ablacoes_cql.csv (por semente) e resultados/ablacoes_cql_resumo.csv
O CSV por semente é gravado a cada execução concluída; rodar de novo retoma de
onde parou (pula config x modelo x semente já presentes).
Uso:    python ablacoes_cql.py [--sementes 3] [--passos 5000] [--processos 2]
"""

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

# A base reproduz os notebooks (ação por Tier); escala_acao='estado' virou o padrão
# de politicas.CQL depois destas ablações (protocolo, seção 12).
_TIER = {'escala_acao': 'tier'}
CONFIGS = {
    'base': {**_TIER},
    'w_0.5': {**_TIER, 'peso_conservador': 0.5},
    'w_20': {**_TIER, 'peso_conservador': 20.0},
    'acao_global': {'escala_acao': 'global'},
    'gamma_0.98': {**_TIER, 'gamma': 0.98},
    'recompensa_log': {**_TIER, 'escala_recompensa': 'log'},
    'recompensa_estado': {**_TIER, 'escala_recompensa': 'estado'},
    # Correção proposta pelo diagnóstico (diagnostico_cql.py)
    'acao_estado': {'escala_acao': 'estado'},
    'acao_estado+recompensa_estado': {'escala_acao': 'estado', 'escala_recompensa': 'estado'},
}


def rodar(tarefa):
    nome, modelo, semente, passos, threads = tarefa
    import torch
    torch.set_num_threads(threads)
    import avaliar_politicas as av
    import politicas as pol
    import simulador as sim

    cenarios = av.carregar_cenarios()
    dados = sim.gerar_dados(cenarios, 5000, semente)
    estados = sim.contextos(cenarios)
    t0 = time.perf_counter()
    cql = pol.CQL(modelo, cenarios, n_passos=passos, passos_por_epoca=1000, paciencia=passos,
                  avaliador=av.metrica_selecao(modelo, cenarios), **CONFIGS[nome])
    cql.treinar(dados, semente)
    tempo = time.perf_counter() - t0

    otimo = av.lucros(av.consultar(pol.Oraculo(modelo, cenarios), estados), estados, cenarios, modelo)
    precos = av.consultar(cql, estados)
    pct = av.lucros(precos, estados, cenarios, modelo) / otimo
    ref = av.consultar(pol.OraculoObservavel(modelo, cenarios), estados)
    low = (estados['Tier'] == 'Low Ticket').to_numpy()
    return {
        'config': nome, 'modelo': modelo, 'semente': semente, 'passos': passos, 'tempo_s': tempo,
        'pct_low': 100 * pct[low].mean(), 'pct_high': 100 * pct[~low].mean(), 'pct_todos': 100 * pct.mean(),
        'low_preco_min': precos[low].min(), 'low_preco_max': precos[low].max(),
        'low_dispersao_log': float(np.std(np.log(precos[low]))),
        'low_corr_oraculo_obs': float(np.corrcoef(np.log(precos[low]), np.log(ref[low]))[0, 1]),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--sementes', type=int, default=3)
    ap.add_argument('--passos', type=int, default=5000)
    ap.add_argument('--processos', type=int, default=2)
    ap.add_argument('--configs', nargs='*', default=list(CONFIGS))
    ap.add_argument('--saida', default='resultados')
    args = ap.parse_args()

    threads = max(1, (os.cpu_count() or 2) // args.processos)
    os.makedirs(args.saida, exist_ok=True)
    caminho = os.path.join(args.saida, 'ablacoes_cql.csv')
    feitas = set()
    if os.path.exists(caminho):
        anteriores = pd.read_csv(caminho)
        feitas = set(zip(anteriores['config'], anteriores['modelo'], anteriores['semente']))
    tarefas = [(c, m, s, args.passos, threads) for s in range(42, 42 + args.sementes)
               for c in args.configs for m in ['venda_unica', 'assinatura'] if (c, m, s) not in feitas]
    print(f"{len(tarefas)} execuções a fazer ({len(feitas)} já no CSV)")
    with ProcessPoolExecutor(max_workers=args.processos) as ex:
        for r in ex.map(rodar, tarefas):
            pd.DataFrame([r]).to_csv(caminho, mode='a', header=not os.path.exists(caminho), index=False)
            print(f"{r['config']:18s} {r['modelo']:12s} s{r['semente']}  Low {r['pct_low']:5.1f}%  "
                  f"High {r['pct_high']:5.1f}%  preços Low {r['low_preco_min']:.0f}-{r['low_preco_max']:.0f}  "
                  f"({r['tempo_s']:.0f}s)", flush=True)

    import avaliar_politicas as av
    det = pd.read_csv(caminho)
    det = det[det['config'].isin(args.configs)]
    resumo = []
    for (cfg, modelo), g in det.groupby(['config', 'modelo'], sort=False):
        linha = {'config': cfg, 'modelo': modelo, 'n_sementes': len(g)}
        for col in ['pct_low', 'pct_high', 'pct_todos']:
            m, lo, hi = av.ic95(g[col])
            linha.update({col: m, f'{col}_ic95_inf': lo, f'{col}_ic95_sup': hi})
        linha.update({c: g[c].mean() for c in ['low_dispersao_log', 'low_corr_oraculo_obs', 'tempo_s']})
        resumo.append(linha)
    pd.DataFrame(resumo).to_csv(os.path.join(args.saida, 'ablacoes_cql_resumo.csv'), index=False)
    with pd.option_context('display.float_format', '{:,.2f}'.format, 'display.width', 160):
        print(pd.DataFrame(resumo)[['config', 'modelo', 'pct_low', 'pct_high', 'pct_todos',
                                    'low_dispersao_log', 'low_corr_oraculo_obs']].to_string(index=False))


if __name__ == '__main__':
    main()
