#!/usr/bin/env python3
"""
Figura da simulação online do bandido (monografia): % do ótimo por decisões
acumuladas, média das sementes com faixa de IC 95%, um painel por modelo.

Entrada: resultados/bandido_online_curva.csv (simular_bandido_online.py)
Saída:   docs/figuras/bandido_online.png
"""

import argparse
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

SUPERFICIE, TINTA, TINTA_2, GRADE, REFERENCIA = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df', '#8a8984'
# Séries que aprendem: slots 1-3 da paleta categórica (validados em todos os pares)
SERIES = {
    'offline_inicial': ('Bandido offline (dados iniciais)', '#2a78d6', '-'),
    'bandido_eps': ('Bandido ε-greedy (aprende operando)', '#eb6834', '-'),
    'bandido_thompson': ('Bandido Thompson (aprende operando)', '#1baf7a', '-'),
}
REFERENCIAS = {
    'oraculo_obs': ('Teto observável', '--'),
    'offline_completo': ('Offline com 48 mil linhas', ':'),
    'aleatorio': ('Aleatório (coleta)', '-'),
}
TITULOS = {'venda_unica': 'Venda única', 'assinatura': 'Assinatura'}


def media_ic(g):
    por_lote = g.groupby('decisoes')['pct_do_otimo']
    media = por_lote.mean()
    n = por_lote.count()
    meia = por_lote.std(ddof=1).fillna(0) * stats.t.ppf(0.975, np.maximum(n - 1, 1)) / np.sqrt(n)
    return media.index.to_numpy(), media.to_numpy(), meia.to_numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--entrada', default='resultados/bandido_online_curva.csv')
    ap.add_argument('--saida', default='docs/figuras/bandido_online.png')
    args = ap.parse_args()
    curva = pd.read_csv(args.entrada)
    n_sementes = curva['semente'].nunique()

    plt.rcParams.update({'font.size': 10, 'axes.edgecolor': GRADE, 'axes.labelcolor': TINTA_2,
                         'xtick.color': TINTA_2, 'ytick.color': TINTA_2, 'text.color': TINTA})
    fig, eixos = plt.subplots(1, 2, figsize=(11, 4.4), facecolor=SUPERFICIE)
    for ax, modelo in zip(eixos, ['venda_unica', 'assinatura']):
        d = curva[curva['modelo'] == modelo]
        ax.set_facecolor(SUPERFICIE)
        ax.grid(axis='y', color=GRADE, linewidth=1)
        ax.set_axisbelow(True)
        for lado in ['top', 'right']:
            ax.spines[lado].set_visible(False)
        for nome, (rotulo, estilo) in REFERENCIAS.items():
            x, m, _ = media_ic(d[d['politica'] == nome])
            ax.plot(x, m, color=REFERENCIA, linestyle=estilo, linewidth=1.5, label=rotulo)
        for nome, (rotulo, cor, estilo) in SERIES.items():
            x, m, h = media_ic(d[d['politica'] == nome])
            ax.fill_between(x, m - h, m + h, color=cor, alpha=0.15, linewidth=0)
            ax.plot(x, m, color=cor, linestyle=estilo, linewidth=2, solid_capstyle='round', label=rotulo)
            ax.annotate(f'{m[-1]:.1f}', (x[-1], m[-1]), xytext=(6, 0), textcoords='offset points',
                        va='center', fontsize=9, color=TINTA_2)
        ax.set_title(TITULOS[modelo], loc='left', fontsize=11, color=TINTA)
        ax.set_xlabel('Decisões de preço em operação')
        ax.set_xlim(0, d['decisoes'].max() * 1.08)
    eixos[0].set_ylabel('% do lucro ótimo')
    handles, labels = eixos[0].get_legend_handles_labels()
    ordem = [3, 4, 5, 0, 1, 2]  # séries que aprendem primeiro
    fig.legend([handles[i] for i in ordem], [labels[i] for i in ordem], loc='lower center', ncol=3,
               frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f'Bandido contextual aprendendo em operação (média de {n_sementes} sementes, faixa = IC 95%)',
                 x=0.01, ha='left', fontsize=12)
    fig.tight_layout(rect=(0, 0.1, 1, 0.97))
    os.makedirs(os.path.dirname(args.saida), exist_ok=True)
    fig.savefig(args.saida, dpi=200, facecolor=SUPERFICIE)
    print(f'Figura salva em {args.saida}')


if __name__ == '__main__':
    main()
