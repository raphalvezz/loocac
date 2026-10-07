#!/usr/bin/env python3
"""
Generator - Gêmeo Digital Econômico (v6 - bandido contextual)
Gera os datasets de SL e RL a partir dos cenários da tabela do artigo.

A economia (demanda, churn, ótimo de preço) fica em simulador.py, para que os
notebooks e o avaliar_politicas.py usem exatamente o mesmo mundo simulado.

Mudanças da v6:
  - Lucro com ótimo interior em cada faixa (elasticidade relativa ao preço médio).
  - Elasticidade varia com Região/Plataforma, então o preço ótimo depende do contexto.
  - Ação salva normalizada por Tier em [-1, 1] (log min-max), alinhada ao tanh do ator.
  - Linhas embaralhadas: o split 80/20 dos notebooks deixa de separar por cenário.
  - Semente fixa e manifesto com hash dos dados (reprodutibilidade).
"""

import hashlib
import json
import os

import joblib
import numpy as np
import pandas as pd
from d3rlpy.dataset import ReplayBuffer, FIFOBuffer, Episode
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tqdm import tqdm

import simulador as sim

print("="*80)
print("GENERATOR (v6) - Bandido contextual sobre os cenários da tabela")
print("="*80)

N_POR_CENARIO = 5000
rng = np.random.default_rng(sim.SEED)

# ============================================================================
# 1. Cenários (tabela do artigo, opcionalmente restritos pelo painel)
# ============================================================================
def aplicar_config_mercado(cenarios, path="config_market.json"):
    """Restringe os cenários às faixas salvas pelo painel "Gêmeo Digital" (/configure_market).

    Sem o arquivo, os cenários da tabela são usados como estão.
    """
    if not os.path.exists(path):
        return cenarios
    with open(path) as f:
        cfg = json.load(f)
    faixas = cfg.get("price_ranges", {})
    budget = cfg.get("budget_range", {})

    ajustados = []
    for c in cenarios:
        faixa = faixas.get(c['Tier'], {})
        pmin = max(c['Price_Min'], faixa.get('min', c['Price_Min']))
        pmax = min(c['Price_Max'], faixa.get('max', c['Price_Max']))
        if pmin >= pmax:
            continue  # cenário fora da faixa configurada
        orcamento = float(np.clip(c['Budget'], budget.get('min', c['Budget']), budget.get('max', c['Budget'])))
        ajustados.append({**c, 'Price_Min': pmin, 'Price_Max': pmax, 'Budget': orcamento})

    if not ajustados:
        raise ValueError(f"Nenhum cenário compatível com as faixas de '{path}'.")
    print(f"Config de mercado '{path}' aplicada: {len(ajustados)}/{len(cenarios)} cenários mantidos.")
    return ajustados

CENARIOS = aplicar_config_mercado(sim.CENARIOS_ARTIGO)
FAIXAS_TIER = sim.faixas_tier(CENARIOS)

# ============================================================================
# 2. Gerar amostras
# ============================================================================
def generate_datasets():
    linhas = []
    print(f"Gerando {N_POR_CENARIO * len(CENARIOS)} amostras ({N_POR_CENARIO} por cenário)...")
    for idx, cenario in enumerate(tqdm(CENARIOS)):
        for _ in range(N_POR_CENARIO):
            regiao = rng.choice(sim.REGIOES)
            plataforma = rng.choice(sim.PLATAFORMAS)
            # Política de coleta: preço uniforme dentro da faixa do cenário
            preco = rng.uniform(cenario['Price_Min'], cenario['Price_Max'])
            linhas.append({
                'Cenario': idx,
                'Regiao': regiao,
                'Plataforma': plataforma,
                'Tier': cenario['Tier'],
                'Orcamento': cenario['Budget'],
                'Idade': '25-34', 'Genero': 'Female', 'Conteudo': 'Video',
                'Tipo_Produto': 'InfoProduto', 'Modelo_Cobranca': 'Venda Unica', 'Complexidade_Oferta': 'Media',
                **sim.memoria_assinatura(cenario),
                'Preco_Amostra': preco,
                'Lucro_Real': sim.amostrar_lucro(preco, cenario, regiao, plataforma, 'venda_unica', rng),
                'LTV_Real': sim.amostrar_lucro(preco, cenario, regiao, plataforma, 'assinatura', rng),
            })
    df = pd.DataFrame(linhas)
    # Embaralha: sem isso, o split 80/20 sequencial dos notebooks testa só os últimos cenários
    return df.sample(frac=1.0, random_state=sim.SEED).reset_index(drop=True)

df = generate_datasets()

# ============================================================================
# 3. Processamento e salvamento
# ============================================================================
print("Salvando artefatos...")

categorical_features = ['Regiao', 'Plataforma', 'Tier', 'Idade', 'Genero', 'Conteudo',
                        'Tipo_Produto', 'Modelo_Cobranca', 'Complexidade_Oferta']
numeric_features_base = ['Orcamento']
numeric_features_memoria = ['dias_desde_ultima_interacao', 'clv_estimate_percentile',
                            'avg_price_offered_segment_90d', 'price_volatility_30d']

# --- 3.1 SL ---
# LTV_Real fica no CSV para o baseline de bandido da assinatura (avaliar_politicas.py)
df.drop(columns=numeric_features_memoria).to_csv('sl_dataset_combined.csv', index=False)

ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit(df[categorical_features])
scaler_state = StandardScaler().fit(df[numeric_features_base])
scaler_price = StandardScaler().fit(df[['Preco_Amostra']])
scaler_profit = StandardScaler().fit(df[['Lucro_Real']])

joblib.dump(ohe, 'sl_ohe_encoder.joblib') # Mesmo nome usado pelo SL_FINAL e pela API
joblib.dump(scaler_state, 'sl_scaler_estado.joblib')
joblib.dump(scaler_price, 'sl_scaler_preco.joblib')
joblib.dump(scaler_profit, 'sl_scaler_lucro.joblib')

# --- 3.2 Estado e ação (comuns aos dois agentes) ---
obs_base = np.concatenate([
    ohe.transform(df[categorical_features]),
    scaler_state.transform(df[numeric_features_base]),
], axis=1)
acoes = np.array([
    sim.preco_para_acao(p, t, FAIXAS_TIER) for p, t in zip(df['Preco_Amostra'], df['Tier'])
]).reshape(-1, 1)
assert np.all(np.abs(acoes) <= 1 + 1e-9), "ação fora de [-1, 1]"

def salvar_buffer(path, obs, recompensas):
    # Um único episódio terminal; com gamma=0 nos notebooks, cada transição é
    # uma decisão independente (bandido contextual).
    episodio = Episode(
        obs.astype(np.float32),
        acoes.astype(np.float32),
        recompensas.reshape(-1, 1).astype(np.float32),
        True,
    )
    buffer = ReplayBuffer(FIFOBuffer(limit=len(obs)), episodes=[episodio])
    with open(path, 'w+b') as f:
        buffer.dump(f)

# --- 3.3 RL Venda Única ---
scaler_reward_rl = StandardScaler().fit(df[['Lucro_Real']])
salvar_buffer('rl_offline_buffer.h5', obs_base, scaler_reward_rl.transform(df[['Lucro_Real']]).ravel())

joblib.dump(ohe, 'ohe_encoder.joblib') # Compartilhado
joblib.dump(scaler_state, 'scaler_estado.joblib') # Compartilhado
joblib.dump(scaler_reward_rl, 'scaler_recompensa.joblib')

cols_base = list(ohe.get_feature_names_out()) + numeric_features_base
with open('colunas_estado_base.json', 'w') as f:
    json.dump(cols_base, f)

# --- 3.4 RL Assinatura (estado inclui memória) ---
scaler_memoria = StandardScaler().fit(df[numeric_features_memoria])
scaler_reward_sub = StandardScaler().fit(df[['LTV_Real']])
obs_sub = np.concatenate([obs_base, scaler_memoria.transform(df[numeric_features_memoria])], axis=1)
salvar_buffer('rl_assinatura_buffer.h5', obs_sub, scaler_reward_sub.transform(df[['LTV_Real']]).ravel())

joblib.dump(scaler_memoria, 'scaler_assinatura_memoria.joblib')
joblib.dump(scaler_reward_sub, 'scaler_assinatura_recompensa.joblib')

cols_sub = cols_base + numeric_features_memoria
with open('colunas_estado_assinatura.json', 'w') as f:
    json.dump(cols_sub, f)

# --- 3.5 Metadados para API, notebooks e avaliação ---
with open('faixas_tier.json', 'w') as f:
    json.dump(FAIXAS_TIER, f, indent=2)
with open('cenarios_treino.json', 'w') as f:
    json.dump(CENARIOS, f, indent=2)

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for bloco in iter(lambda: f.read(1 << 20), b''):
            h.update(bloco)
    return h.hexdigest()

arquivos = ['sl_dataset_combined.csv', 'rl_offline_buffer.h5', 'rl_assinatura_buffer.h5',
            'faixas_tier.json', 'cenarios_treino.json']
with open('data_manifest.json', 'w') as f:
    json.dump({
        'gerador': 'Generator_NEW.py v6',
        'seed': sim.SEED,
        'amostras': len(df),
        'amostras_por_cenario': N_POR_CENARIO,
        'sha256': {a: sha256(a) for a in arquivos},
    }, f, indent=2)

print("\n✅ SUCESSO: buffers, scalers e metadados gerados.")
print(f"  Cenários processados: {len(CENARIOS)}")
print(f"  Faixas de ação por Tier: {FAIXAS_TIER}")
