#!/usr/bin/env python3
"""
Treina o CQL (versão canônica, politicas.CQL) e exporta um agente por modelo
de cobrança. Substitui os notebooks de RL no pipeline; eles ficam só como
documentação da versão anterior.

    cql_venda_unica.d3 + cql_venda_unica.joblib
    cql_assinatura.d3  + cql_assinatura.joblib

Configuração: a padrão de politicas.CQL (gamma=0, ação na faixa observável do
estado, peso conservador 5, 64 quantis), até --passos passos, com o checkpoint
escolhido pelo lucro simulado (% do ótimo) e parada após --paciencia épocas
sem melhora. Carregar: politicas.CQL.carregar('cql_venda_unica').
"""

import argparse
import time

import numpy as np
import pandas as pd

import avaliar_politicas as av
import politicas as pol
import simulador as sim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--passos', type=int, default=50000)
    ap.add_argument('--paciencia', type=int, default=20)
    ap.add_argument('--modelos', nargs='*', default=av.MODELOS)
    args = ap.parse_args()
    dados = pd.read_csv('sl_dataset_combined.csv')
    cenarios = av.carregar_cenarios()
    estados = sim.contextos(cenarios)
    for modelo in args.modelos:
        inicio = time.perf_counter()
        cql = pol.CQL(modelo, cenarios, n_passos=args.passos, paciencia=args.paciencia,
                      avaliador=av.metrica_selecao(modelo, cenarios))
        cql.treinar(dados, sim.SEED)
        prefixo = f'cql_{modelo}'
        cql.salvar(prefixo)
        recarregado = pol.CQL.carregar(prefixo)
        assert np.allclose(av.consultar(recarregado, estados), av.consultar(cql, estados))
        print(f"✓ {prefixo}.d3/.joblib: {100 * cql.score_selecao:.1f}% do ótimo no melhor checkpoint "
              f"({time.perf_counter() - inicio:.0f}s)", flush=True)


if __name__ == '__main__':
    main()
