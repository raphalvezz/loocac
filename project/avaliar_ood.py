#!/usr/bin/env python3
"""
O CQL recomenda ações fora do suporte dos dados de treino? Verificação empírica.

Não é prova formal; mede, para as recomendações do CQL nos estados de avaliação,
quão longe elas estão dos dados logados com que ele foi treinado.

Problema estático (cql_<modelo>, 144 estados):
  fora_faixa_estado     preço fora da faixa observável do estado (união das faixas
                        dos cenários com a mesma chave). Na escala 'estado' a ação
                        [-1, 1] é essa faixa, então deve ser 0 por construção.
  na_borda              |ação| ≥ 0,99: o ator saturou num extremo da faixa.
  fora_faixa_cenario    preço fora da faixa do cenário VERDADEIRO do estado. Nos
                        cenários que dividem a chave (venda única, Low Ticket), o
                        agente não sabe qual é o verdadeiro; um preço da faixa do
                        outro cenário não tem nenhum dado logado naquele cenário.
  vizinhos              linhas de treino do mesmo contexto (cenário, região,
                        plataforma) com ação a menos de 0,05 da recomendada; 0 =
                        nenhum dado perto da ação escolhida.

Ambiente sequencial (cql_seq_<modelo>, trajetórias dos 144 contextos, 12 períodos):
  fora_reputacao        reputação visitada fora do intervalo [mín, máx] das
                        reputações logadas naquele contexto e período.
  dist_reputacao        distância à reputação logada mais próxima (mesmo contexto e
                        período), em unidades de ρ.
  mais o fora_faixa_cenario e o na_borda de cada passo.

Entradas: cql_*.d3/.joblib, cql_seq_*.d3/.joblib e sl_dataset_combined.csv na pasta
atual (os do train_pipeline.py --sequencial). Sem os sequenciais, só a parte estática.
Saída:    resultados/ood/{resumo,estatico_por_estado,sequencial_por_passo}.csv
Uso:      python avaliar_ood.py
"""

import argparse
import os

import numpy as np
import pandas as pd

import avaliar_politicas as av
import politicas as pol
import sequencial as sq
import simulador as sim

CONTEXTO = ['Cenario', 'Regiao', 'Plataforma']
RAIO = 0.05


def medidas(agente, estados, precos, cenarios):
    sem = estados.drop(columns='Cenario')
    acao = agente.preco_para_acao(precos, sem)
    lo_est, hi_est = np.array(agente.faixas(sem), dtype=float).T
    lo_cen = np.array([cenarios[c]['Price_Min'] for c in estados['Cenario']])
    hi_cen = np.array([cenarios[c]['Price_Max'] for c in estados['Cenario']])
    tol = 1e-6
    return pd.DataFrame({
        'acao': acao,
        'fora_faixa_estado': (precos < lo_est * (1 - tol)) | (precos > hi_est * (1 + tol)),
        'na_borda': np.abs(acao) >= 0.99,
        'fora_faixa_cenario': (precos < lo_cen * (1 - tol)) | (precos > hi_cen * (1 + tol)),
    })


def vizinhos(agente, estados, acao, treino):
    """Linhas de treino do mesmo contexto com ação a menos de RAIO da recomendada."""
    acoes_treino = agente.preco_para_acao(treino['Preco_Amostra'].to_numpy(), treino.drop(columns='Cenario'))
    por_ctx = pd.Series(acoes_treino).groupby([treino[c].to_numpy() for c in CONTEXTO]).apply(np.asarray)
    return np.array([int(np.sum(np.abs(por_ctx[(c, r, p)] - a) < RAIO))
                     for (c, r, p), a in zip(estados[CONTEXTO].itertuples(index=False), acao)])


def estatico(modelo, cenarios, dados):
    agente = pol.CQL.carregar(f'cql_{modelo}')
    estados = sim.contextos(cenarios)
    precos = av.consultar(agente, estados)
    treino = dados.iloc[: int(len(dados) * pol.FRACAO_TREINO)]
    m = medidas(agente, estados, precos, cenarios)
    m['vizinhos'] = vizinhos(agente, estados, m['acao'].to_numpy(), treino)
    return pd.concat([estados[CONTEXTO + ['Tier']].reset_index(drop=True), m], axis=1).assign(
        modelo=modelo, preco=precos)


def sequencial(modelo, cenarios):
    agente = pol.CQL.carregar(f'cql_seq_{modelo}')
    logados = sq.gerar_dados(cenarios, 200, sim.SEED, modelo)
    treino = logados.iloc[: int(len(logados) * pol.FRACAO_TREINO)]
    rep_log = treino.groupby(CONTEXTO + ['passo'])['reputacao'].apply(np.asarray)
    av_seq = sq.Avaliador(cenarios, modelo)
    linhas = []
    rho = np.full(len(av_seq.contextos), sq.RHO_0)
    for t in range(sq.T):
        estados = av_seq.contextos.assign(reputacao=rho, periodo=t / sq.T)
        precos = np.asarray(av.consultar(agente, estados), dtype=float)
        m = medidas(agente, estados, precos, cenarios)
        dist, fora = [], []
        for (c, r, p), rr in zip(estados[CONTEXTO].itertuples(index=False), rho):
            obs = rep_log.get((c, r, p, t))
            if obs is None:  # contexto/período sem episódio logado no treino
                dist.append(np.nan), fora.append(True)
                continue
            dist.append(float(np.min(np.abs(obs - rr))))
            fora.append(bool(rr < obs.min() - 1e-9 or rr > obs.max() + 1e-9))
        linhas.append(pd.concat([estados[CONTEXTO + ['Tier']].reset_index(drop=True), m], axis=1).assign(
            modelo=modelo, periodo=t, reputacao=rho, preco=precos, dist_reputacao=dist, fora_reputacao=fora))
        rho = sq.proxima_reputacao(rho, precos, av_seq.p_ref)
    return pd.concat(linhas, ignore_index=True)


def resumir(df, ambiente):
    out = []
    todos = pd.concat([df, df.assign(Tier='Todos')], ignore_index=True)
    for (modelo, tier), g in todos.groupby(['modelo', 'Tier'], sort=False):
        linha = {'ambiente': ambiente, 'modelo': modelo, 'tier': tier, 'n': len(g),
                 'fora_faixa_estado_pct': 100 * g['fora_faixa_estado'].mean(),
                 'na_borda_pct': 100 * g['na_borda'].mean(),
                 'fora_faixa_cenario_pct': 100 * g['fora_faixa_cenario'].mean()}
        if 'vizinhos' in g:
            linha.update({'sem_vizinhos_pct': 100 * (g['vizinhos'] == 0).mean(),
                          'vizinhos_mediana': float(g['vizinhos'].median())})
        if 'fora_reputacao' in g:
            linha.update({'fora_reputacao_pct': 100 * g['fora_reputacao'].mean(),
                          'dist_reputacao_mediana': float(g['dist_reputacao'].median()),
                          'dist_reputacao_p95': float(g['dist_reputacao'].quantile(0.95))})
        out.append(linha)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--saida', default='resultados/ood')
    args = ap.parse_args()
    cenarios = av.carregar_cenarios()
    dados = pd.read_csv('sl_dataset_combined.csv')
    os.makedirs(args.saida, exist_ok=True)

    est = pd.concat([estatico(m, cenarios, dados) for m in av.MODELOS], ignore_index=True)
    est.to_csv(os.path.join(args.saida, 'estatico_por_estado.csv'), index=False)
    resumo = resumir(est, 'estatico')
    if all(os.path.exists(f'cql_seq_{m}.d3') for m in sq.MODELOS):
        seq = pd.concat([sequencial(m, cenarios) for m in sq.MODELOS], ignore_index=True)
        seq.to_csv(os.path.join(args.saida, 'sequencial_por_passo.csv'), index=False)
        resumo += resumir(seq, 'sequencial')
    else:
        print('cql_seq_* ausentes: só a parte estática (rode treinar_cql_sequencial.py)')
    resumo = pd.DataFrame(resumo)
    resumo.to_csv(os.path.join(args.saida, 'resumo.csv'), index=False)
    with pd.option_context('display.width', 220, 'display.float_format', '{:,.2f}'.format):
        print(resumo.to_string(index=False))


if __name__ == '__main__':
    main()
