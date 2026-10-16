#!/usr/bin/env python3
"""
Regressão quantílica do SL e do bandido (incerteza do lucro previsto).

Carrega sl_*_regressor_model.joblib e bandido_*.joblib, treina um LightGBM com
objective='quantile' para cada quantil de politicas.QUANTIS_RISCO (mesmas
features e escala do alvo de cada método) e salva no MESMO arquivo. A API passa
a devolver VaR/CVaR 5% próprios desses métodos (var_5_metodo / cvar_5_metodo),
como já faz com o crítico quantílico do CQL.

Entradas: sl_dataset_combined.csv e os .joblib acima, na pasta atual.
Uso:      python treinar_quantis.py
"""

import joblib
import pandas as pd

import politicas as pol
import simulador as sim

ARQUIVOS = {('sl', 'venda_unica'): 'sl_profit_regressor_model.joblib',
            ('sl', 'assinatura'): 'sl_ltv_regressor_model.joblib',
            ('bandido', 'venda_unica'): 'bandido_venda_unica.joblib',
            ('bandido', 'assinatura'): 'bandido_assinatura.joblib'}


def cobertura(p, teste):
    """Fração do lucro observado abaixo de cada quantil previsto (ideal: o próprio quantil)."""
    Q = p.prever_quantis(teste, teste['Preco_Amostra'].to_numpy())
    y = teste[pol.ALVO[p.modelo]].to_numpy(dtype=float)[:, None]
    return (y < Q).mean(axis=0)


def main():
    dados = pd.read_csv('sl_dataset_combined.csv')
    teste = dados.iloc[int(len(dados) * pol.FRACAO_TREINO):]
    for (metodo, modelo), caminho in ARQUIVOS.items():
        p = joblib.load(caminho)
        p.treinar_quantis(dados, sim.SEED)
        joblib.dump(p, caminho)
        cob = cobertura(joblib.load(caminho), teste)
        texto = ' '.join(f'q{q:g}:{c:.3f}' for q, c in zip(p.quantis, cob))
        print(f"✓ {caminho}: quantis {p.quantis} | cobertura no teste {texto}")


if __name__ == '__main__':
    main()
