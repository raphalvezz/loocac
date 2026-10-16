#!/usr/bin/env python3
"""
Seleção de checkpoint do CQL sem o simulador: FQE contra importance sampling
ponderado (WIS), comparados com o lucro verdadeiro.

O CQL escolhe o checkpoint pelo lucro no simulador (seção 4 do protocolo), o que
um sistema real não teria. Aqui, durante um treino do CQL, guardamos checkpoints
a cada `--intervalo` épocas e estimamos o valor de cada um com dois métodos fora
da política, usando só os 20% de dados logados que o agente não viu:

  FQE   Fitted Q Evaluation (d3rlpy.ope, γ = 0), recompensa relativa à média da
        chave de estado (avaliar_fqe.py). Aprende Q^π e lê Q^π(s, π(s)).
  WIS   importance sampling ponderado com núcleo: a política é determinística e a
        ação é contínua, então cada linha logada recebe o peso
            w = K_h(a_logada − π(s)) / μ(a_logada | s)
        com K_h gaussiano na escala da ação [-1, 1] (h = --banda) e μ a densidade
        da coleta (preço uniforme na faixa do cenário, convertida para a escala da
        ação). V̂(contexto) = Σ w·lucro / Σ w, por contexto (cenário, região,
        plataforma). Não ajusta modelo nenhum; depende só das propensões da coleta.

Para cada checkpoint: % do ótimo verdadeiro (simulador), estimado pelo FQE e
estimado pelo WIS (os dois na mesma métrica: média nos 144 contextos de
estimativa / lucro ótimo). Concordância: correlação de Spearman com o verdadeiro,
checkpoint escolhido por cada método e arrependimento (p.p. do ótimo perdidos em
relação ao melhor checkpoint verdadeiro).

Saída: resultados/selecao_ope/{checkpoints,resumo}.csv
Uso:   python avaliar_selecao_ope.py [--sementes 1] [--passos-cql 50000] [--intervalo 5]
"""

import argparse
import os
import tempfile

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import avaliar_fqe
import avaliar_politicas as av
import politicas as pol
import simulador as sim

CONTEXTO = ['Cenario', 'Regiao', 'Plataforma']


class Captura:
    """Avaliador do CQL: devolve o % do ótimo no simulador (a seleção original) e,
    a cada `intervalo` épocas, grava o checkpoint para a análise fora da política."""

    def __init__(self, metrica, pasta, intervalo):
        self.metrica, self.pasta, self.intervalo = metrica, pasta, intervalo
        self.epoca, self.checkpoints = 0, []

    def __call__(self, agente):
        self.epoca += 1
        score = self.metrica(agente)
        if self.epoca % self.intervalo == 0:
            caminho = os.path.join(self.pasta, f'epoca_{self.epoca:03d}.pt')
            agente.algo.save_model(caminho)
            self.checkpoints.append((self.epoca, caminho, 100 * score))
        return score


def densidade_coleta(agente, teste, cenarios):
    """μ(a | s) na escala da ação do agente: preço uniforme em [lo, hi] do cenário.
    a = 2 (ln p − ln L) / (ln U − ln L) − 1  ⇒  dp/da = p (ln U − ln L) / 2."""
    L, U = agente.faixas_acao(teste).T
    lo = np.array([cenarios[c]['Price_Min'] for c in teste['Cenario']])
    hi = np.array([cenarios[c]['Price_Max'] for c in teste['Cenario']])
    p = teste['Preco_Amostra'].to_numpy(dtype=float)
    return p * (np.log(U) - np.log(L)) / 2 / (hi - lo)


def wis(agente, teste, cenarios, banda):
    """Lucro estimado da política por contexto (Series indexada por CONTEXTO) e ESS médio."""
    sem_cenario = teste.drop(columns='Cenario')
    a_log = agente.preco_para_acao(teste['Preco_Amostra'].to_numpy(), sem_cenario)
    a_pol = agente.preco_para_acao(agente.precos(sem_cenario), sem_cenario)
    nucleo = np.exp(-0.5 * ((a_log - a_pol) / banda) ** 2) / (banda * np.sqrt(2 * np.pi))
    df = teste[CONTEXTO].copy()
    df['w'] = nucleo / densidade_coleta(agente, teste, cenarios)
    df['wr'] = df['w'] * teste[pol.ALVO[agente.modelo]].to_numpy(dtype=float)
    df['w2'] = df['w'] ** 2
    g = df.groupby(CONTEXTO)[['w', 'wr', 'w2']].sum()
    ess = (g['w'] ** 2 / g['w2']).mean()
    return g['wr'] / g['w'], float(ess)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--sementes', type=int, default=1)
    ap.add_argument('--passos-cql', type=int, default=50000)
    ap.add_argument('--intervalo', type=int, default=5, help='épocas (1.000 passos) entre checkpoints')
    ap.add_argument('--passos-fqe', type=int, default=5000)
    ap.add_argument('--banda', type=float, default=0.1, help='banda do núcleo do WIS, na escala da ação')
    ap.add_argument('--saida', default='resultados/selecao_ope')
    args = ap.parse_args()

    cenarios = av.carregar_cenarios()
    estados = sim.contextos(cenarios)
    os.makedirs(args.saida, exist_ok=True)
    linhas = []
    for semente in range(42, 42 + args.sementes):
        dados = sim.gerar_dados(cenarios, 5000, semente)
        teste = dados.iloc[int(len(dados) * pol.FRACAO_TREINO):].reset_index(drop=True)
        for modelo in av.MODELOS:
            otimo = av.lucros(av.consultar(pol.Oraculo(modelo, cenarios), estados), estados, cenarios, modelo)
            chave_ctx = pd.MultiIndex.from_frame(estados[CONTEXTO])
            with tempfile.TemporaryDirectory() as pasta:
                captura = Captura(av.metrica_selecao(modelo, cenarios), pasta, args.intervalo)
                agente = pol.CQL(modelo, cenarios, n_passos=args.passos_cql, paciencia=10 ** 9, avaliador=captura)
                agente.treinar(dados, semente)
                for epoca, caminho, pct_sim in captura.checkpoints:
                    agente.algo.load_model(caminho)
                    fqe, media = avaliar_fqe.ajustar_fqe(agente, teste, args.passos_fqe)
                    est_fqe, _ = avaliar_fqe.estimar(agente, fqe, media, estados)
                    v_wis, ess = wis(agente, teste, cenarios, args.banda)
                    est_wis = v_wis.reindex(chave_ctx).to_numpy()
                    linhas.append({'modelo': modelo, 'semente': semente, 'epoca': epoca, 'passos': epoca * 1000,
                                   'pct_simulador': pct_sim,
                                   'pct_fqe': 100 * float(np.mean(est_fqe / otimo)),
                                   'pct_wis': 100 * float(np.nanmean(est_wis / otimo)),
                                   'ess_wis_medio': ess})
                    u = linhas[-1]
                    print(f"  s{semente} {modelo:12s} época {epoca:3d}: simulador {u['pct_simulador']:5.1f}% | "
                          f"FQE {u['pct_fqe']:5.1f}% | WIS {u['pct_wis']:5.1f}% (ESS {ess:.1f})", flush=True)
                    pd.DataFrame(linhas).to_csv(os.path.join(args.saida, 'checkpoints.csv'), index=False)

    det = pd.DataFrame(linhas)
    resumo = []
    for (modelo, semente), g in det.groupby(['modelo', 'semente'], sort=False):
        melhor = g['pct_simulador'].max()
        linha = {'modelo': modelo, 'semente': semente, 'n_checkpoints': len(g),
                 'melhor_simulador': melhor, 'epoca_melhor_simulador': int(g.loc[g['pct_simulador'].idxmax(), 'epoca'])}
        for est in ('fqe', 'wis'):
            escolhido = g.loc[g[f'pct_{est}'].idxmax()]
            linha.update({f'spearman_{est}': spearmanr(g[f'pct_{est}'], g['pct_simulador']).statistic,
                          f'epoca_escolhida_{est}': int(escolhido['epoca']),
                          f'arrependimento_{est}_pp': melhor - escolhido['pct_simulador'],
                          f'erro_medio_{est}_pp': float((g[f'pct_{est}'] - g['pct_simulador']).mean())})
        linha['spearman_fqe_wis'] = spearmanr(g['pct_fqe'], g['pct_wis']).statistic
        linha['mesma_escolha'] = linha['epoca_escolhida_fqe'] == linha['epoca_escolhida_wis']
        resumo.append(linha)
    resumo = pd.DataFrame(resumo)
    resumo.to_csv(os.path.join(args.saida, 'resumo.csv'), index=False)
    with pd.option_context('display.float_format', '{:,.2f}'.format, 'display.width', 200):
        print(resumo.to_string(index=False))


if __name__ == '__main__':
    main()
