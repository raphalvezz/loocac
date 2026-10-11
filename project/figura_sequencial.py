#!/usr/bin/env python3
"""Figura: preço médio por período (relativo ao preço míope ótimo) no ambiente sequencial.

Entrada: resultados/sequencial/trajetorias.csv (avaliar_sequencial.py)
Saída:   docs/figuras/sequencial_trajetorias.png (um painel por cobrança)
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

ESTILO = {'oraculo_pd': ('k', '-'), 'oraculo_miope': ('k', ':'), 'cql_g095': ('C3', '-'),
          'cql_g0': ('C3', '--'), 'bandido': ('C0', '-'), 'sl': ('C2', '-'), 'bc': ('C1', '-')}
TITULO = {'venda_unica': 'Venda única', 'assinatura': 'Assinatura'}

t = pd.read_csv('resultados/sequencial/trajetorias.csv')
if 'modelo' not in t:
    t['modelo'] = 'venda_unica'
modelos = list(dict.fromkeys(t['modelo']))
n_sementes = t['semente'].nunique()
fig, eixos = plt.subplots(1, len(modelos), figsize=(6.5 * len(modelos), 4), sharey=True, squeeze=False)
for ax, modelo in zip(eixos[0], modelos):
    media = t[t['modelo'] == modelo].groupby(['politica', 'periodo'])['preco_rel_miope'].mean().unstack(0)
    for nome, (cor, linha) in ESTILO.items():
        if nome in media:
            ax.plot(media.index + 1, media[nome], color=cor, linestyle=linha, label=nome, marker='o', ms=3)
    ax.set_title(f'{TITULO.get(modelo, modelo)} (144 contextos, {n_sementes} sementes)')
    ax.set_xlabel('período da campanha')
    ax.grid(alpha=.3)
eixos[0][0].set_ylabel('preço / preço míope ótimo')
eixos[0][-1].legend(ncol=2, fontsize=8)
fig.suptitle('Ambiente sequencial: trajetória média de preço')
fig.tight_layout()
fig.savefig('docs/figuras/sequencial_trajetorias.png', dpi=150)
