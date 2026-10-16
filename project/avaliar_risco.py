#!/usr/bin/env python3
"""
Calibração do risco (VaR/CVaR 5%) de cada método contra o valor exato do simulador.

O lucro observado é lucro_esperado(p) × ruído, com ruído lognormal de média 1
(simulador.amostrar_lucro, σ = SIGMA_RUIDO). Então, para cada (estado, preço):

    VaR 5% exato  = E(p) · exp(−σ²/2 + σ·z₀,₀₅)
    CVaR 5% exato = E(p) · Φ(z₀,₀₅ − σ) / 0,05
    cobertura(v)  = P(lucro < v) = Φ((ln(v/E(p)) + σ²/2) / σ)      (ideal: 5%)

Erros em % do lucro ótimo do estado ((estimado − exato) / lucro ótimo): dividir pelo
valor exato explode onde o lucro do preço é quase zero (na venda única, os cenários
que dividem a mesma chave de estado têm faixas diferentes, e o preço escolhido pode
cair onde um deles quase não lucra).

Fontes de risco comparadas, todas no mesmo par (estado, preço):
  critico_quantilico     quantis do crítico do CQL (rl)
  regressao_quantilica   LightGBM quantílico do SL e do bandido (treinar_quantis.py)

Pontos avaliados: o preço recomendado por cada método nos 144 estados e uma grade
de 5 preços por estado (calibração fora do ponto escolhido); resumo por tier e no
total. Na venda única, o Low Ticket tem estados que não identificam o cenário: o
risco de um modelo que só vê o estado é o da mistura dos cenários, não o exato.

Entradas: cql_*, sl_*_regressor_model.joblib, bandido_*.joblib (com quantis) na pasta atual.
Saída:    resultados/risco.csv (+ risco_por_estado.csv)
Uso:      python avaliar_risco.py
"""

import argparse
import os

import joblib
import numpy as np
import pandas as pd
from scipy.stats import norm

import avaliar_politicas as av
import politicas as pol
import simulador as sim

ALFA = 0.05
ARQUIVOS_SL = {'venda_unica': 'sl_profit_regressor_model.joblib', 'assinatura': 'sl_ltv_regressor_model.joblib'}


def exato(esperado):
    s, z = sim.SIGMA_RUIDO, norm.ppf(ALFA)
    return esperado * np.exp(-s ** 2 / 2 + s * z), esperado * norm.cdf(z - s) / ALFA


def cobertura(var, esperado):
    s = sim.SIGMA_RUIDO
    return norm.cdf((np.log(np.maximum(var, 1e-12) / esperado) + s ** 2 / 2) / s)


def carregar(modelo, cenarios):
    pols = {'rl': pol.CQL.carregar(f'cql_{modelo}'),
            'sl': joblib.load(ARQUIVOS_SL[modelo]),
            'bandido': joblib.load(f'bandido_{modelo}.joblib')}
    for nome, p in pols.items():
        if nome != 'rl' and not p.tem_quantis:
            raise SystemExit(f'{nome}/{modelo} sem quantis: rode python treinar_quantis.py')
    return pols


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--saida', default='resultados')
    args = ap.parse_args()

    cenarios = av.carregar_cenarios()
    estados = sim.contextos(cenarios)
    sem_cenario = estados.drop(columns='Cenario')
    linhas = []
    for modelo in av.MODELOS:
        pols = carregar(modelo, cenarios)
        otimo = av.lucros(av.consultar(pol.Oraculo(modelo, cenarios), estados), estados, cenarios, modelo)
        pontos = {nome: av.consultar(p, estados) for nome, p in pols.items()}
        # grade: 5 preços por estado, igualmente espaçados (em log) na faixa do cenário
        grade = np.stack([np.geomspace(cenarios[c]['Price_Min'], cenarios[c]['Price_Max'], 5)
                          for c in estados['Cenario']])
        for nome, p in pols.items():
            fonte = 'critico_quantilico' if nome == 'rl' else 'regressao_quantilica'
            for ponto, precos_ponto in [('preco_recomendado', pontos[nome][:, None]), ('grade', grade)]:
                for j in range(precos_ponto.shape[1]):
                    precos = precos_ponto[:, j]
                    esperado = av.lucros(precos, estados, cenarios, modelo)
                    var_ex, cvar_ex = exato(esperado)
                    var_est, cvar_est = p.risco(sem_cenario, precos, ALFA)
                    linhas += [{'modelo': modelo, 'metodo': nome, 'fonte': fonte, 'ponto': ponto,
                                'cenario': int(c), 'tier': t, 'preco': pr, 'lucro_esperado': e, 'lucro_otimo': o,
                                'var_exato': ve, 'var_estimado': vs, 'cvar_exato': ce, 'cvar_estimado': cs,
                                'cobertura_var': cob}
                               for c, t, pr, e, o, ve, vs, ce, cs, cob in zip(
                                   estados['Cenario'], estados['Tier'], precos, esperado, otimo, var_ex, var_est,
                                   cvar_ex, cvar_est, cobertura(var_est, esperado))]
            print(f'  {modelo:12s} {nome:8s} ok', flush=True)

    det = pd.DataFrame(linhas)
    resumo = []
    todos = pd.concat([det, det.assign(tier='Todos')], ignore_index=True)
    for chave, g in todos.groupby(['modelo', 'metodo', 'fonte', 'ponto', 'tier'], sort=False):
        rv = 100 * (g['var_estimado'] - g['var_exato']) / g['lucro_otimo']
        rc = 100 * (g['cvar_estimado'] - g['cvar_exato']) / g['lucro_otimo']
        resumo.append(dict(zip(['modelo', 'metodo', 'fonte', 'ponto', 'tier'], chave), n=len(g),
                           vies_var_pct_otimo=rv.mean(), erro_abs_var_pct_otimo=rv.abs().mean(),
                           vies_cvar_pct_otimo=rc.mean(), erro_abs_cvar_pct_otimo=rc.abs().mean(),
                           cobertura_var_media=g['cobertura_var'].mean(),
                           cobertura_var_p10=g['cobertura_var'].quantile(0.1),
                           cobertura_var_p90=g['cobertura_var'].quantile(0.9)))
    resumo = pd.DataFrame(resumo)
    os.makedirs(args.saida, exist_ok=True)
    resumo.to_csv(os.path.join(args.saida, 'risco.csv'), index=False)
    det.to_csv(os.path.join(args.saida, 'risco_por_estado.csv'), index=False)
    colunas = ['modelo', 'metodo', 'ponto', 'tier', 'vies_var_pct_otimo', 'erro_abs_var_pct_otimo',
               'erro_abs_cvar_pct_otimo', 'cobertura_var_media']
    with pd.option_context('display.float_format', '{:,.3f}'.format, 'display.width', 160):
        print(resumo[colunas].to_string(index=False))


if __name__ == '__main__':
    main()
