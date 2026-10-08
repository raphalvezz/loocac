#!/usr/bin/env python3
"""
Diagnóstico do CQL no Low Ticket: o preço recomendado fica quase constante.

Pergunta: o problema está no CRÍTICO (Q(s, a) não distingue os preços) ou no
ATOR (o crítico distingue, mas o ator não segue)?

Para cada estado da grade, avalia o crítico numa grade de 101 ações e compara:
  - preço do ator (o que a política recomenda)
  - argmax do crítico (o preço que o próprio crítico acha melhor)
  - preço do oráculo observável (o melhor possível dado o estado)
Métricas por tier: correlação (em log) de ator e argmax do crítico com o oráculo
e % do ótimo de cada um.

Treina as configurações pedidas (padrão: base e recompensa_estado, ver
ablacoes_cql.py) com uma semente e gera docs/figuras/diagnostico_cql_low_ticket.png:
curva do crítico x curva de lucro verdadeira num estado Low Ticket.

Uso: python diagnostico_cql.py [--passos 5000] [--configs base recompensa_estado]
"""

import argparse
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import ablacoes_cql  # noqa: E402
import avaliar_politicas as av  # noqa: E402
import politicas as pol  # noqa: E402
import simulador as sim  # noqa: E402

MODELO = 'venda_unica'
ACOES = np.linspace(-1, 1, 101)
# Estado mostrado na figura: cenário 0 (Low Ticket, 10-20, orçamento 100), Europa/Instagram
ESTADO_FIGURA = {'Cenario': 0, 'Regiao': 'Europe', 'Plataforma': 'Instagram'}


def curva_critico(cql, obs):
    """Q esperado (média do ensemble e dos quantis) para cada ação da grade, por estado."""
    n = len(obs)
    obs_rep = np.repeat(obs, len(ACOES), axis=0)
    acoes_rep = np.tile(ACOES, n).reshape(-1, 1).astype(np.float32)
    return cql.algo.predict_value(obs_rep, acoes_rep).reshape(n, len(ACOES))


def diagnosticar(nome, cenarios, dados, estados, passos, semente):
    cql = pol.CQL(MODELO, cenarios, n_passos=passos, passos_por_epoca=1000, paciencia=passos,
                  avaliador=av.metrica_selecao(MODELO, cenarios), **ablacoes_cql.CONFIGS[nome])
    cql.treinar(dados, semente)
    obs = cql.prep.transform(estados.drop(columns='Cenario'))
    q = curva_critico(cql, obs)
    preco_ator = av.consultar(cql, estados)
    preco_critico = np.array([float(sim.acao_para_preco(ACOES[i], t, cql.faixas_tier))
                              for i, t in zip(q.argmax(axis=1), estados['Tier'])])
    preco_oraculo = av.consultar(pol.OraculoObservavel(MODELO, cenarios), estados)
    otimo = av.lucros(av.consultar(pol.Oraculo(MODELO, cenarios), estados), estados, cenarios, MODELO)
    linhas = []
    for tier in ['Low Ticket', 'High Ticket']:
        m = (estados['Tier'] == tier).to_numpy()
        corr = lambda p: float(np.corrcoef(np.log(p[m]), np.log(preco_oraculo[m]))[0, 1])  # noqa: E731
        pct = lambda p: 100 * float(np.mean(av.lucros(p, estados, cenarios, MODELO)[m] / otimo[m]))  # noqa: E731
        linhas.append({'config': nome, 'tier': tier,
                       'corr_ator_oraculo': corr(preco_ator), 'corr_critico_oraculo': corr(preco_critico),
                       'pct_ator': pct(preco_ator), 'pct_argmax_critico': pct(preco_critico),
                       'pct_oraculo_obs': pct(preco_oraculo),
                       'faixa_precos_ator': f"{preco_ator[m].min():.0f}-{preco_ator[m].max():.0f}"})
    i = estados.index[(estados['Cenario'] == ESTADO_FIGURA['Cenario']) & (estados['Regiao'] == ESTADO_FIGURA['Regiao'])
                      & (estados['Plataforma'] == ESTADO_FIGURA['Plataforma'])][0]
    figura = {'q': q[i], 'preco_ator': preco_ator[i], 'faixas': cql.faixas_tier}
    return pd.DataFrame(linhas), figura


def desenhar(figuras, cenarios, saida):
    c = cenarios[ESTADO_FIGURA['Cenario']]
    tier = c['Tier']
    fig, eixos = plt.subplots(1, len(figuras), figsize=(5.2 * len(figuras), 3.8), facecolor='#fcfcfb', squeeze=False)
    for ax, (nome, f) in zip(eixos[0], figuras.items()):
        precos = np.array([float(sim.acao_para_preco(a, tier, f['faixas'])) for a in ACOES])
        lucro = sim.lucro_esperado(precos, c, ESTADO_FIGURA['Regiao'], ESTADO_FIGURA['Plataforma'], MODELO)
        norm = lambda v: (v - v.min()) / (v.max() - v.min() + 1e-12)  # noqa: E731
        ax.set_facecolor('#fcfcfb')
        ax.axvspan(c['Price_Min'], c['Price_Max'], color='#e4e3df', alpha=0.6, linewidth=0, label='Faixa dos dados')
        ax.plot(precos, norm(lucro), color='#52514e', linewidth=2, label='Lucro verdadeiro')
        ax.plot(precos, norm(f['q']), color='#2a78d6', linewidth=2, label='Q do crítico')
        ax.axvline(f['preco_ator'], color='#eb6834', linewidth=2, label=f"Preço do ator ({f['preco_ator']:.0f})")
        ax.set_xscale('log')
        ax.set_title(nome, loc='left', fontsize=11)
        ax.set_xlabel('Preço (US$, escala log)')
        for lado in ['top', 'right']:
            ax.spines[lado].set_visible(False)
    eixos[0][0].set_ylabel('Valor normalizado (0 a 1)')
    eixos[0][0].legend(frameon=False, fontsize=8, loc='lower left')
    fig.suptitle(f"CQL, venda única - estado Low Ticket (cenário {ESTADO_FIGURA['Cenario'] + 1}, "
                 f"{ESTADO_FIGURA['Regiao']}/{ESTADO_FIGURA['Plataforma']})", x=0.01, ha='left', fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(saida), exist_ok=True)
    fig.savefig(saida, dpi=200, facecolor='#fcfcfb')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--passos', type=int, default=5000)
    ap.add_argument('--semente', type=int, default=42)
    ap.add_argument('--configs', nargs='*', default=['base', 'recompensa_estado'])
    ap.add_argument('--saida', default='resultados')
    ap.add_argument('--figura', default='docs/figuras/diagnostico_cql_low_ticket.png')
    args = ap.parse_args()

    cenarios = av.carregar_cenarios()
    dados = sim.gerar_dados(cenarios, 5000, args.semente)
    estados = sim.contextos(cenarios)
    tabelas, figuras = [], {}
    for nome in args.configs:
        tabela, figuras[nome] = diagnosticar(nome, cenarios, dados, estados, args.passos, args.semente)
        tabelas.append(tabela)
        print(tabela.round(2).to_string(index=False), flush=True)
    os.makedirs(args.saida, exist_ok=True)
    pd.concat(tabelas).to_csv(os.path.join(args.saida, 'diagnostico_cql.csv'), index=False)
    desenhar(figuras, cenarios, args.figura)
    print(f"Figura: {args.figura}")


if __name__ == '__main__':
    main()
