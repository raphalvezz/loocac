#!/usr/bin/env python3
"""
Sensibilidade do ambiente sequencial a κ (velocidade da reputação) e λ (preço
justo = λ · preço míope ótimo). Não treina políticas: só o ótimo por programação
dinâmica e o oráculo míope, nos 144 contextos.

Para cada (κ, λ) e cobrança: % do ótimo do oráculo míope (quanto vale planejar =
100 − esse valor) e a fração de contextos em que a política ótima começa no preço
mínimo da faixa e a fração dos 12 períodos que a trajetória ótima passa no piso
(dinâmica "exagerada": a PD fica presa no piso a campanha toda).

A calibração usada (κ = 0,15, λ = 0,85) foi escolhida por inspeção, antes de treinar
qualquer política (protocolo, seção 9b); esta tabela mostra onde ela fica na grade.

Saída: resultados/calibracao_sequencial.csv
Uso:   python calibracao_sequencial.py
"""

import os

import numpy as np
import pandas as pd

import avaliar_politicas as av
import sequencial as sq

KAPPAS = (0.05, 0.10, 0.15, 0.25, 0.50)
LAMBDAS = (0.75, 0.85, 0.95)


def main():
    cenarios = av.carregar_cenarios()
    original = sq.KAPPA, sq.LAMBDA
    linhas = []
    try:
        for kappa in KAPPAS:
            for lam in LAMBDAS:
                sq.KAPPA, sq.LAMBDA = kappa, lam
                for modelo in sq.MODELOS:
                    avaliador = sq.Avaliador(cenarios, modelo)
                    miope = avaliador.pct(sq.OraculoMiope(cenarios, modelo))
                    no_piso = []
                    for r in avaliador.contextos.itertuples():
                        c = cenarios[r.Cenario]
                        rhos, politica = sq.programacao_dinamica(c, r.Regiao, r.Plataforma, modelo=modelo)[1]
                        p0 = np.interp(sq.RHO_0, rhos, politica[0])
                        no_piso.append(p0 <= c['Price_Min'] * 1.01)
                    _, precos_pd = avaliador.simular(sq.OraculoPD(cenarios, modelo).precos)
                    piso = np.array([cenarios[r.Cenario]['Price_Min'] for r in avaliador.contextos.itertuples()])
                    periodos_piso = 100 * float(np.mean(precos_pd <= piso[None, :] * 1.01))
                    linhas.append({'kappa': kappa, 'lambda': lam, 'modelo': modelo,
                                   'miope_pct_do_otimo': float(miope.mean()),
                                   'valor_de_planejar_pp': float(100 - miope.mean()),
                                   'pd_comeca_no_piso_pct': 100 * float(np.mean(no_piso)),
                                   'pd_periodos_no_piso_pct': periodos_piso,
                                   'usada': (kappa, lam) == original})
                    print(f'  κ={kappa:.2f} λ={lam:.2f} {modelo:12s} míope {miope.mean():5.1f}% | '
                          f'PD começa no piso {100 * np.mean(no_piso):5.1f}% | períodos no piso {periodos_piso:5.1f}%', flush=True)
    finally:
        sq.KAPPA, sq.LAMBDA = original
    os.makedirs('resultados', exist_ok=True)
    pd.DataFrame(linhas).to_csv('resultados/calibracao_sequencial.csv', index=False)


if __name__ == '__main__':
    main()
