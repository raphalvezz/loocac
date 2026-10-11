#!/usr/bin/env python3
"""
Treina o bandido contextual (offline) nos dados do Generator e exporta um
.joblib por modelo de cobrança, como o SL_FINAL faz com o SL:

    bandido_venda_unica.joblib   lucro (venda única)
    bandido_assinatura.joblib    LTV (assinatura)

Cada arquivo é um objeto bandido.BandidoLGBM: precos(estados) recomenda o preço
e prever_lucro(estados, precos) devolve o lucro previsto.
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

import avaliar_politicas as av
import bandido as ban
import politicas as pol
import simulador as sim


def main():
    dados = pd.read_csv('sl_dataset_combined.csv')
    cenarios = av.carregar_cenarios()
    teste = dados.iloc[int(len(dados) * pol.FRACAO_TREINO):]
    estados = sim.contextos(cenarios)
    for modelo in av.MODELOS:
        b = ban.BandidoLGBM(modelo, cenarios).treinar(dados, sim.SEED)
        caminho = f'bandido_{modelo}.joblib'
        joblib.dump(b, caminho)

        # Conferência: o arquivo salvo reproduz as decisões
        recarregado = joblib.load(caminho)
        assert np.allclose(av.consultar(recarregado, estados), av.consultar(b, estados))
        r2 = r2_score(teste[pol.ALVO[modelo]], b.prever_lucro(teste, teste['Preco_Amostra'].to_numpy()))
        otimo = av.lucros(av.consultar(pol.Oraculo(modelo, cenarios), estados), estados, cenarios, modelo)
        pct = 100 * np.mean(av.lucros(av.consultar(b, estados), estados, cenarios, modelo) / otimo)
        print(f"✓ {caminho}: R² {r2:.4f} | {pct:.1f}% do ótimo na grade de avaliação")


if __name__ == '__main__':
    main()
