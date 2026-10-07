#!/usr/bin/env python3
"""
Compara as políticas de preço no simulador (simulador.py).

Para cada contexto de avaliação (cenário x região x plataforma), cada política
escolhe um preço e medimos o lucro ESPERADO no simulador (sem ruído). O oráculo
é o preço ótimo do próprio simulador, o limite superior.

Políticas:
  - aleatorio      preço uniforme na faixa observável
  - meio_faixa     ponto médio da faixa observável
  - maximo_faixa   preço máximo da faixa observável
  - bandido        LightGBM (contexto + log-preço -> lucro) + busca em grade na faixa observável
  - cql            agente CQL treinado (se o .d3 existir e o d3rlpy estiver instalado)
  - oraculo_obs    melhor preço possível sabendo só o estado (teto real de qualquer política)
  - oraculo        ótimo do simulador na faixa do cenário (referência dos percentuais)

"Faixa observável" = união das faixas dos cenários com o mesmo (Tier, Orçamento),
que é o que o estado permite saber (simulador.faixa_observavel). Assim todas as
políticas, exceto o oráculo, usam a mesma informação. Os cenários 3/4 e 5/6 são
indistinguíveis no estado; o oráculo os distingue, então nem uma política
perfeita chega a 100% neles.

Uso:  python avaliar_politicas.py            -> resultados_politicas.csv
"""

import json
import os

import joblib
import numpy as np
import pandas as pd

import simulador as sim

MODELOS = {
    'venda_unica': {'agente': 'modelo_rl_final.d3', 'alvo': 'Lucro_Real',
                    'scaler_recompensa': 'scaler_recompensa.joblib'},
    'assinatura': {'agente': 'modelo_rl_assinatura.d3', 'alvo': 'LTV_Real',
                   'scaler_recompensa': 'scaler_assinatura_recompensa.joblib'},
}
CATEGORICAS_FIXAS = {'Idade': '25-34', 'Genero': 'Female', 'Conteudo': 'Video',
                     'Tipo_Produto': 'InfoProduto', 'Modelo_Cobranca': 'Venda Unica',
                     'Complexidade_Oferta': 'Media'}


def carregar_cenarios():
    if os.path.exists('cenarios_treino.json'):
        with open('cenarios_treino.json') as f:
            return json.load(f)
    return sim.CENARIOS_ARTIGO


def contextos(cenarios):
    linhas = []
    for idx, c in enumerate(cenarios):
        for regiao in sim.REGIOES:
            for plataforma in sim.PLATAFORMAS:
                linhas.append({'Cenario': idx, 'Regiao': regiao, 'Plataforma': plataforma,
                               'Tier': c['Tier'], 'Orcamento': c['Budget'],
                               **CATEGORICAS_FIXAS, **sim.memoria_assinatura(c)})
    return pd.DataFrame(linhas)


def lucros(precos, ctx, cenarios, modelo):
    return np.array([
        float(sim.lucro_esperado(p, cenarios[r.Cenario], r.Regiao, r.Plataforma, modelo))
        for p, r in zip(precos, ctx.itertuples())
    ])


# ---------------------------------------------------------------------------
# Políticas
# ---------------------------------------------------------------------------
def politica_oraculo(ctx, cenarios, modelo):
    return np.array([sim.preco_otimo(cenarios[r.Cenario], r.Regiao, r.Plataforma, modelo)[0]
                     for r in ctx.itertuples()])


def politica_oraculo_observavel(ctx, cenarios, modelo, n=2001):
    """Para cada estado, o preço que maximiza o lucro médio entre os cenários que
    o estado não distingue (pesos iguais, como nos dados)."""
    precos = []
    for r in ctx.itertuples():
        c = cenarios[r.Cenario]
        grupo = [d for d in cenarios if d['Tier'] == c['Tier'] and d['Budget'] == c['Budget']]
        grade = np.geomspace(*sim.faixa_observavel(c, cenarios), n)
        media = np.mean([sim.lucro_esperado(grade, d, r.Regiao, r.Plataforma, modelo) for d in grupo], axis=0)
        precos.append(float(grade[np.argmax(media)]))
    return np.array(precos)


def lucro_aleatorio(ctx, cenarios, modelo, n=401):
    """Lucro esperado da política uniforme na faixa observável (média sobre a faixa)."""
    out = []
    for r in ctx.itertuples():
        c = cenarios[r.Cenario]
        grade = np.linspace(*sim.faixa_observavel(c, cenarios), n)
        out.append(float(np.mean(sim.lucro_esperado(grade, c, r.Regiao, r.Plataforma, modelo))))
    return np.array(out)


def treinar_bandido(df, alvo, ohe):
    import lightgbm as lgb

    def X(d, preco):
        return np.column_stack([ohe.transform(d[list(ohe.feature_names_in_)]),
                                np.log(d['Orcamento'].to_numpy()), np.log(preco)])

    treino = df.iloc[: int(len(df) * 0.8)]  # mesmo split 80/20 dos notebooks (dados embaralhados)
    modelo = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=63,
                               random_state=sim.SEED, verbose=-1)
    modelo.fit(X(treino, treino['Preco_Amostra'].to_numpy()), treino[alvo])
    return modelo, X


def politica_bandido(ctx, cenarios, modelo_lgb, X, n=200):
    precos = []
    for i in range(len(ctx)):
        linha = ctx.iloc[[i]]
        # Só dentro da faixa observável: fora dela não há dados e as árvores extrapolam
        grade = np.geomspace(*sim.faixa_observavel(cenarios[linha['Cenario'].iloc[0]], cenarios), n)
        rep = linha.loc[linha.index.repeat(n)].reset_index(drop=True)
        precos.append(float(grade[np.argmax(modelo_lgb.predict(X(rep, grade)))]))
    return np.array(precos)


def montar_obs(ctx, modelo, ohe, scaler_estado, scaler_memoria=None):
    partes = [ohe.transform(ctx[list(ohe.feature_names_in_)]),
              scaler_estado.transform(ctx[list(scaler_estado.feature_names_in_)])]
    if modelo == 'assinatura':
        partes.append(scaler_memoria.transform(ctx[list(scaler_memoria.feature_names_in_)]))
    return np.concatenate(partes, axis=1).astype(np.float32)


def politica_cql(algo, obs, ctx, faixas):
    acoes = algo.predict(obs).reshape(-1)
    return np.array([float(sim.acao_para_preco(a, t, faixas)) for a, t in zip(acoes, ctx['Tier'])])


class AvaliadorSimulado:
    """Métrica de seleção de checkpoint para os notebooks: % médio do lucro ótimo.

    Substitui o average_q (estimativa do próprio crítico), que favorecia a época 1.
    """

    def __init__(self, modelo):
        self.modelo = modelo
        self.cenarios = carregar_cenarios()
        with open('faixas_tier.json') as f:
            self.faixas = json.load(f)
        self.ctx = contextos(self.cenarios)
        memoria = joblib.load('scaler_assinatura_memoria.joblib') if modelo == 'assinatura' else None
        self.obs = montar_obs(self.ctx, modelo, joblib.load('ohe_encoder.joblib'),
                              joblib.load('scaler_estado.joblib'), memoria)
        self.otimo = lucros(politica_oraculo(self.ctx, self.cenarios, modelo), self.ctx, self.cenarios, modelo)

    def __call__(self, algo):
        precos = politica_cql(algo, self.obs, self.ctx, self.faixas)
        return float(np.mean(lucros(precos, self.ctx, self.cenarios, self.modelo) / self.otimo))


# ---------------------------------------------------------------------------
def avaliar(modelo):
    cfg = MODELOS[modelo]
    cenarios = carregar_cenarios()
    with open('faixas_tier.json') as f:
        faixas = json.load(f)
    ctx = contextos(cenarios)
    df = pd.read_csv('sl_dataset_combined.csv')
    ohe = joblib.load('ohe_encoder.joblib')

    otimo = lucros(politica_oraculo(ctx, cenarios, modelo), ctx, cenarios, modelo)
    faixa = [sim.faixa_observavel(cenarios[i], cenarios) for i in ctx['Cenario']]
    resultados = {
        'aleatorio': lucro_aleatorio(ctx, cenarios, modelo),
        'meio_faixa': lucros([(lo + hi) / 2 for lo, hi in faixa], ctx, cenarios, modelo),
        'maximo_faixa': lucros([hi for _, hi in faixa], ctx, cenarios, modelo),
    }
    lgb_modelo, X = treinar_bandido(df, cfg['alvo'], ohe)
    resultados['bandido'] = lucros(politica_bandido(ctx, cenarios, lgb_modelo, X), ctx, cenarios, modelo)

    if os.path.exists(cfg['agente']):
        try:
            import d3rlpy
            algo = d3rlpy.load_learnable(cfg['agente'], device='cpu')
            memoria = joblib.load('scaler_assinatura_memoria.joblib') if modelo == 'assinatura' else None
            obs = montar_obs(ctx, modelo, ohe, joblib.load('scaler_estado.joblib'), memoria)
            resultados['cql'] = lucros(politica_cql(algo, obs, ctx, faixas), ctx, cenarios, modelo)
        except ImportError:
            print(f"  (d3rlpy não instalado; CQL de {modelo} fora da comparação)")
    else:
        print(f"  ({cfg['agente']} não encontrado; CQL de {modelo} fora da comparação)")
    resultados['oraculo_obs'] = lucros(politica_oraculo_observavel(ctx, cenarios, modelo), ctx, cenarios, modelo)
    resultados['oraculo'] = otimo

    linhas = []
    for nome, valores in resultados.items():
        for tier in ['Low Ticket', 'High Ticket', 'Todos']:
            m = (ctx['Tier'] == tier).to_numpy() if tier != 'Todos' else np.ones(len(ctx), bool)
            linhas.append({'modelo': modelo, 'politica': nome, 'tier': tier,
                           'lucro_medio': valores[m].mean(),
                           'pct_do_otimo': 100 * np.mean(valores[m] / otimo[m])})
    return pd.DataFrame(linhas)


if __name__ == '__main__':
    tabela = pd.concat([avaliar(m) for m in MODELOS], ignore_index=True)
    tabela.to_csv('resultados_politicas.csv', index=False)
    with pd.option_context('display.float_format', '{:,.1f}'.format, 'display.width', 120):
        print(tabela[tabela['tier'] == 'Todos'].to_string(index=False))
        print('\nPor tier: resultados_politicas.csv')
