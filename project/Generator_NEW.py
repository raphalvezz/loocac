#!/usr/bin/env python3
"""
Generator - Gêmeo Digital Econômico (v5 - Artigo Table-Driven)
Gera datasets focados EXATAMENTE nos cenários da tabela do artigo.
"""

import numpy as np
import pandas as pd
import json
import joblib
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import d3rlpy
from d3rlpy.dataset import ReplayBuffer, FIFOBuffer, Episode
from tqdm import tqdm
import os

print("="*80)
print("GENERATOR (v5 - Artigo) - Focado em Cenários de Tabela")
print("="*80)

# ============================================================================
# 1. DEFINIÇÃO DOS CENÁRIOS DO ARTIGO (Sua Tabela)
# ============================================================================
# Aqui você define EXATAMENTE o que quer que a IA aprenda.
# Cada entrada é um "mundo" que será simulado.


CENARIOS_ARTIGO = [
    # --- LOW TICKET (CPA alvo ~25% a 30% do preço médio) ---
    
    # 1. Micro-ticket (10-20), Budget Baixo
    # Preço Médio: $15 -> CPA Ideal: $5
    {'Tier': 'Low Ticket', 'Price_Min': 10.0, 'Price_Max': 20.0, 'Budget': 100.0, 'CPA_Target': 5.0},
    
    # 2. Mid-Low (50-100), Budget Baixo
    # Preço Médio: $75 -> CPA Ideal: $20 (Não $5)
    {'Tier': 'Low Ticket', 'Price_Min': 50.0, 'Price_Max': 100.0, 'Budget': 200.0, 'CPA_Target': 20.0},
   
    # 3. Low-Mid (20-50), Budget Médio
    # Preço Médio: $35 -> CPA Ideal: $10
    {'Tier': 'Low Ticket', 'Price_Min': 20.0, 'Price_Max': 50.0, 'Budget': 1000.0, 'CPA_Target': 10.0},
    
    # 4. Mid-Low (50-100), Budget Médio
    # Preço Médio: $75 -> CPA Ideal: $22
    {'Tier': 'Low Ticket', 'Price_Min': 50.0, 'Price_Max': 100.0, 'Budget': 1000.0, 'CPA_Target': 22.0},
   
    # 5. Low-Mid (20-50), Budget Alto
    # Escala gera ineficiência, CPA sobe um pouco -> $12
    {'Tier': 'Low Ticket', 'Price_Min': 20.0, 'Price_Max': 50.0, 'Budget': 10000.0, 'CPA_Target': 12.0},
    
    # 6. Mid-Low (50-100), Budget Alto
    # CPA ajustado -> $25
    {'Tier': 'Low Ticket', 'Price_Min': 50.0, 'Price_Max': 100.0, 'Budget': 10000.0, 'CPA_Target': 25.0},
   
    # --- HIGH TICKET (CPA alvo ~15% a 25% do preço médio) ---
    # High Ticket tem margem maior, mas volume menor. O CPA absoluto é alto.

    # 7. High Entry (500-1000), Budget Baixo
    # Preço Médio: $750 -> CPA Ideal: $150 (Ok, estava certo)
    {'Tier': 'High Ticket', 'Price_Min': 500.0, 'Price_Max': 1000.0, 'Budget': 2000.0, 'CPA_Target': 150.0},

    # 8. High Mid (1000-2000), Budget Médio
    # Preço Médio: $1500 -> CPA Ideal: $350 (Não $150)
    {'Tier': 'High Ticket', 'Price_Min': 1000.0, 'Price_Max': 2000.0, 'Budget': 5000.0, 'CPA_Target': 350.0},

    # 9. High Premium (2000-5000), Budget Médio
    # Preço Médio: $3500 -> CPA Ideal: $800
    {'Tier': 'High Ticket', 'Price_Min': 2000.0, 'Price_Max': 5000.0, 'Budget': 10000.0, 'CPA_Target': 800.0},

    # 10. High Premium (2000-5000), Budget Alto
    # Escala agressiva -> CPA $900
    {'Tier': 'High Ticket', 'Price_Min': 2000.0, 'Price_Max': 5000.0, 'Budget': 50000.0, 'CPA_Target': 900.0},
    
    # 11. Ultra High (5000-10000), Budget Alto
    # Preço Médio: $7500 -> CPA Ideal: $1800
    {'Tier': 'High Ticket', 'Price_Min': 5000.0, 'Price_Max': 10000.0, 'Budget': 100000.0, 'CPA_Target': 1800.0},

    # 12. Enterprise/Mastermind (10000-25000), Budget Muito Alto
    # Preço Médio: $17500 -> CPA Ideal: $4000 (Venda complexa)
    {'Tier': 'High Ticket', 'Price_Min': 10000.0, 'Price_Max': 25000.0, 'Budget': 200000.0, 'CPA_Target': 4000.0}
]
    # Adicione mais linhas conforme sua tabela do artigo...


# Configurações fixas para o resto (não variam na tabela)
REGIOES = ['North America', 'Europe', 'Asia', 'South America']
PLATAFORMAS = ['Instagram', 'Facebook', 'LinkedIn']

# ============================================================================
# 2. Funções Econômicas (Ajustadas para os Cenários)
# ============================================================================

def get_scenario_data():
    """Escolhe um cenário da tabela aleatoriamente para gerar um dado."""
    cenario = np.random.choice(CENARIOS_ARTIGO)
    return cenario

def generate_price_cpa_from_scenario(cenario):
    """Gera preço dentro da faixa exata do cenário."""
    # Preço aleatório DENTRO da faixa específica (ex: 10 a 20)
    price = np.random.uniform(cenario['Price_Min'], cenario['Price_Max'])
    
    # CPA com leve ruído em torno do alvo
    cpa = cenario['CPA_Target'] * np.random.uniform(0.9, 1.1)
    
    return price, cpa

def calculate_demand_scenario(estado, preco, cenario):
    """Calcula demanda calibrada para o cenário específico."""
    # Preço médio da faixa (para referência)
    a0 = (cenario['Price_Min'] + cenario['Price_Max']) / 2
    
    # Demanda base (c0) ajustada pelo orçamento
    # Se orçamento é 100, c0 é menor do que se for 1000
    orcamento_factor = np.log1p(cenario['Budget']) / np.log1p(1000) # Normalizado em 1000
    c0 = 100.0 * orcamento_factor 
    
    beta = 0.05 # Sensibilidade padrão
    
    # Curva: Se preço > média, demanda cai.
    if preco > a0:
        conversoes = c0 * np.exp(-beta * (preco - a0))
    else:
        conversoes = c0 # Demanda máxima no preço baixo
        
    return max(0, conversoes)

class Transition:
    def __init__(self, observation, action, reward, terminal):
        self.observation = observation
        self.action = action
        self.reward = reward
        self.terminal = terminal

# ============================================================================
# 3. Gerar Datasets (Preenchendo com dados dos cenários)
# ============================================================================

def generate_datasets(num_samples_per_scenario=5000):
    """Gera dados balanceados para cada cenário da tabela."""
    
    data_sl = []
    transitions_rl = []
    transitions_sub = []
    
    total_samples = num_samples_per_scenario * len(CENARIOS_ARTIGO)
    print(f"Gerando {total_samples} amostras ({num_samples_per_scenario} por cenário)...")
    
    for cenario in tqdm(CENARIOS_ARTIGO):
        for _ in range(num_samples_per_scenario):
            # --- 1. Estado Base ---
            estado = {
                'Regiao': np.random.choice(REGIOES),
                'Plataforma': np.random.choice(PLATAFORMAS),
                'Tier': cenario['Tier'],
                'Orcamento': cenario['Budget'], # ORÇAMENTO EXATO DA TABELA
                'Idade': '25-34', 'Genero': 'Female', 'Conteudo': 'Video', # Fixos ou variados
                'Tipo_Produto': 'InfoProduto', 'Modelo_Cobranca': 'Venda Unica', 'Complexidade_Oferta': 'Media',
                # Features memória (mock)
                'dias_desde_ultima_interacao': 30, 'clv_estimate_percentile': 0.5,
                'avg_price_offered_segment_90d': cenario['Price_Min'], 'price_volatility_30d': 1.0
            }
            
            # --- 2. Ação e Recompensa ---
            preco, cpa = generate_price_cpa_from_scenario(cenario)
            
            # SL e RL Fixo (Lucro Imediato)
            conversoes = calculate_demand_scenario(estado, preco, cenario)
            lucro = max(0, conversoes * (preco - cpa))
            
            # RL Assinatura (LTV)
            churn = 0.1 + (preco / 1000) # Simplificado
            ltv = max(0, (preco / churn) - cpa)

            # --- 3. Salvar nas Listas ---
            
            # SL
            row = estado.copy()
            row['Preco_Amostra'] = preco
            row['Lucro_Real'] = lucro
            data_sl.append(row)
            
            # RL Fixo
            transitions_rl.append(Transition(
                observation=estado, action=preco, reward=lucro, terminal=0.0
            ))
            
            # RL Assinatura
            transitions_sub.append(Transition(
                observation=estado, action=preco, reward=ltv, terminal=0.0
            ))
            
    return pd.DataFrame(data_sl), transitions_rl, transitions_sub

# Executar Geração
df_sl, trans_rl, trans_sub = generate_datasets()

# ============================================================================
# 4. Processamento e Salvamento (Padrão do Projeto)
# ============================================================================

print("Salvando artefatos...")

# Definir colunas para OHE
categorical_features = ['Regiao', 'Plataforma', 'Tier', 'Idade', 'Genero', 'Conteudo', 
                       'Tipo_Produto', 'Modelo_Cobranca', 'Complexidade_Oferta']
numeric_features_base = ['Orcamento']
numeric_features_memoria = ['dias_desde_ultima_interacao', 'clv_estimate_percentile', 
                           'avg_price_offered_segment_90d', 'price_volatility_30d']

# --- 4.1 SL ---
df_sl_clean = df_sl.drop(columns=numeric_features_memoria, errors='ignore')
df_sl_clean.to_csv('sl_dataset_combined.csv', index=False)

# Scalers SL
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit(df_sl[categorical_features])
scaler_state = StandardScaler().fit(df_sl[numeric_features_base])
scaler_price = StandardScaler().fit(df_sl[['Preco_Amostra']])
scaler_profit = StandardScaler().fit(df_sl[['Lucro_Real']])

joblib.dump(ohe, 'sl_encoder.joblib')
joblib.dump(scaler_state, 'sl_scaler_estado.joblib')
joblib.dump(scaler_price, 'sl_scaler_preco.joblib')
joblib.dump(scaler_profit, 'sl_scaler_lucro.joblib')

# --- 4.2 RL Fixo ---
# Reaproveita os scalers do SL para consistência (ou cria novos se preferir)
# Vamos criar novos para garantir formato correto (d3rlpy)
scaler_acao_rl = StandardScaler().fit(np.array([t.action for t in trans_rl]).reshape(-1, 1))
scaler_reward_rl = StandardScaler().fit(np.array([t.reward for t in trans_rl]).reshape(-1, 1))

# Processa Observações
df_obs = pd.DataFrame([t.observation for t in trans_rl])
obs_processed = np.concatenate([
    ohe.transform(df_obs[categorical_features]),
    scaler_state.transform(df_obs[numeric_features_base])
], axis=1)
actions_processed = scaler_acao_rl.transform(np.array([t.action for t in trans_rl]).reshape(-1, 1))
rewards_processed = scaler_reward_rl.transform(np.array([t.reward for t in trans_rl]).reshape(-1, 1)).flatten()

# Salva Buffer RL Fixo
episode_rl = Episode(
    obs_processed.astype(np.float32),
    actions_processed.astype(np.float32),
    rewards_processed.reshape(-1, 1).astype(np.float32),
    False # terminated
)
buffer_rl = ReplayBuffer(FIFOBuffer(limit=len(trans_rl)), episodes=[episode_rl])
with open('rl_offline_buffer.h5', 'w+b') as f:
    buffer_rl.dump(f)

# Salva Scalers RL
joblib.dump(ohe, 'ohe_encoder.joblib') # Compartilhado
joblib.dump(scaler_state, 'scaler_estado.joblib') # Compartilhado
joblib.dump(scaler_acao_rl, 'scaler_acao.joblib')
joblib.dump(scaler_reward_rl, 'scaler_recompensa.joblib')

# Salva Metadados
cols_base = list(ohe.get_feature_names_out()) + numeric_features_base
with open('colunas_estado_base.json', 'w') as f:
    json.dump(cols_base, f)

# --- 4.3 RL Assinatura ---
# (Similar ao Fixo, mas inclui memória)
scaler_memoria = StandardScaler().fit(df_sl[numeric_features_memoria])
scaler_acao_sub = StandardScaler().fit(np.array([t.action for t in trans_sub]).reshape(-1, 1))
scaler_reward_sub = StandardScaler().fit(np.array([t.reward for t in trans_sub]).reshape(-1, 1))

df_obs_sub = pd.DataFrame([t.observation for t in trans_sub])
obs_sub_processed = np.concatenate([
    ohe.transform(df_obs_sub[categorical_features]),
    scaler_state.transform(df_obs_sub[numeric_features_base]),
    scaler_memoria.transform(df_obs_sub[numeric_features_memoria])
], axis=1)

actions_sub_proc = scaler_acao_sub.transform(np.array([t.action for t in trans_sub]).reshape(-1, 1))
rewards_sub_proc = scaler_reward_sub.transform(np.array([t.reward for t in trans_sub]).reshape(-1, 1)).flatten()

episode_sub = Episode(
    obs_sub_processed.astype(np.float32),
    actions_sub_proc.astype(np.float32),
    rewards_sub_proc.reshape(-1, 1).astype(np.float32),
    False
)
buffer_sub = ReplayBuffer(FIFOBuffer(limit=len(trans_sub)), episodes=[episode_sub])
with open('rl_assinatura_buffer.h5', 'w+b') as f:
    buffer_sub.dump(f)

joblib.dump(scaler_memoria, 'scaler_assinatura_memoria.joblib')
joblib.dump(scaler_acao_sub, 'scaler_assinatura_acao.joblib')
joblib.dump(scaler_reward_sub, 'scaler_assinatura_recompensa.joblib')

cols_sub = cols_base + numeric_features_memoria
with open('colunas_estado_assinatura.json', 'w') as f:
    json.dump(cols_sub, f)

print("\n✅ SUCESSO: Todos os buffers e scalers gerados para os cenários da tabela.")
print(f"  Cenários processados: {len(CENARIOS_ARTIGO)}")

CENARIOS_ARTIGO = [
    # Low Ticket (Faixa 10-20, Budget 100)
    {'Tier': 'Low Ticket', 'Price_Min': 10.0, 'Price_Max': 20.0, 'Budget': 100.0, 'CPA_Target': 5.0},
    
    # Low Ticket (Faixa 10-20, Budget 1000)
    {'Tier': 'Low Ticket', 'Price_Min': 10.0, 'Price_Max': 20.0, 'Budget': 1000.0, 'CPA_Target': 5.0},

    # High Ticket (Exemplo: Faixa 400-600, Budget 2000)
    {'Tier': 'High Ticket', 'Price_Min': 400.0, 'Price_Max': 600.0, 'Budget': 2000.0, 'CPA_Target': 150.0},
    
    # Adicione mais linhas conforme sua tabela do artigo...
]