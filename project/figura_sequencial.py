#!/usr/bin/env python3
"""Figura: preço médio por período (relativo ao preço míope ótimo) no ambiente sequencial.

Entrada: resultados/sequencial/trajetorias.csv (avaliar_sequencial.py)
Saída:   docs/figuras/sequencial_trajetorias.png
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

ESTILO = {'oraculo_pd': ('k', '-'), 'oraculo_miope': ('k', ':'), 'cql_g095': ('C3', '-'),
          'cql_g0': ('C3', '--'), 'bandido': ('C0', '-'), 'sl': ('C2', '-'), 'bc': ('C1', '-')}

t = pd.read_csv('resultados/sequencial/trajetorias.csv')
media = t.groupby(['politica', 'periodo'])['preco_rel_miope'].mean().unstack(0)
fig, ax = plt.subplots(figsize=(7, 4))
for nome, (cor, linha) in ESTILO.items():
    if nome in media:
        ax.plot(media.index + 1, media[nome], color=cor, linestyle=linha, label=nome, marker='o', ms=3)
ax.set_xlabel('período da campanha')
ax.set_ylabel('preço / preço míope ótimo')
ax.set_title('Ambiente sequencial: trajetória média de preço (144 contextos, 3 sementes)')
ax.legend(ncol=2, fontsize=8)
ax.grid(alpha=.3)
fig.tight_layout()
fig.savefig('docs/figuras/sequencial_trajetorias.png', dpi=150)
