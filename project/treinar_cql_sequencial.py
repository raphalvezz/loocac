#!/usr/bin/env python3
"""
Treina e exporta o CQL do ambiente sequencial (γ = 0,95) usado pela rota
experimental /recomendar_sequencial da API:

    cql_seq_venda_unica.d3/.joblib   cql_seq_assinatura.d3/.joblib

Mesma configuração do cql_g095 de avaliar_sequencial.py (estado com reputação e
período, recompensa por contexto, checkpoint pelo lucro total simulado), com a
semente padrão do simulador.

Uso: python treinar_cql_sequencial.py [--passos 50000] [--episodios 200]
"""

import argparse

import numpy as np

import avaliar_politicas as av
import politicas as pol
import sequencial as sq
import simulador as sim


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--passos', type=int, default=50000)
    ap.add_argument('--episodios', type=int, default=200, help='campanhas logadas por cenário')
    args = ap.parse_args()

    cenarios = av.carregar_cenarios()
    for modelo in sq.MODELOS:
        avaliador = sq.Avaliador(cenarios, modelo)
        dados = sq.gerar_dados(cenarios, args.episodios, sim.SEED, modelo)
        agente = pol.CQL(modelo, cenarios, n_passos=args.passos, gamma=0.95, extras=sq.EXTRAS,
                         escala_recompensa='estado', avaliador=avaliador).treinar(dados, sim.SEED)
        agente.salvar(f'cql_seq_{modelo}')
        pct = float(np.mean(avaliador.pct(pol.CQL.carregar(f'cql_seq_{modelo}'))))
        print(f"✓ cql_seq_{modelo}.d3/.joblib: {pct:.1f}% do ótimo da programação dinâmica")


if __name__ == '__main__':
    main()
