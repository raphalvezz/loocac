#!/usr/bin/env python3
"""
Treina o behavior cloning (BC) nos dados do Generator e exporta um agente por
modelo de cobrança, como treinar_bandido.py faz com o bandido:

    bc_venda_unica.d3 + bc_venda_unica.joblib
    bc_assinatura.d3  + bc_assinatura.joblib

Carregar: politicas.BC.carregar('bc_venda_unica').
"""

import argparse

import numpy as np
import pandas as pd

import avaliar_politicas as av
import politicas as pol
import simulador as sim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--passos', type=int, default=5000)
    args = ap.parse_args()
    dados = pd.read_csv('sl_dataset_combined.csv')
    cenarios = av.carregar_cenarios()
    estados = sim.contextos(cenarios)
    for modelo in av.MODELOS:
        bc = pol.BC(modelo, cenarios, n_passos=args.passos).treinar(dados, sim.SEED)
        prefixo = f'bc_{modelo}'
        bc.salvar(prefixo)
        # Conferência: o agente salvo reproduz as decisões
        recarregado = pol.BC.carregar(prefixo)
        assert np.allclose(av.consultar(recarregado, estados), av.consultar(bc, estados))
        otimo = av.lucros(av.consultar(pol.Oraculo(modelo, cenarios), estados), estados, cenarios, modelo)
        pct = 100 * np.mean(av.lucros(av.consultar(bc, estados), estados, cenarios, modelo) / otimo)
        print(f"✓ {prefixo}.d3/.joblib: {pct:.1f}% do ótimo na grade de avaliação")


if __name__ == '__main__':
    main()
