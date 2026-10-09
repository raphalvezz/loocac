import pandas as pd
import numpy as np
import warnings

import politicas as pol

# Suprime avisos de versões
warnings.filterwarnings('ignore')

print("="*60)
print("🤖 GERADOR DE TABELA DE RESULTADOS (RL FIXO)")
print("="*60)

# --- 1. DEFINA AQUI AS LINHAS DA SUA TABELA ---
# Coloque exatamente os valores que você quer testar no artigo
# --- 1. DEFINA AQUI AS LINHAS DA SUA TABELA ---
CENARIOS_TABELA = [
    # --- LOW TICKET ---
    {'Nome': 'Cenário 1',  'Regiao': 'North America', 'Tier': 'Low Ticket',  'Orcamento': 100.0},   # Micro, Budget Baixo
    {'Nome': 'Cenário 2',  'Regiao': 'North America', 'Tier': 'Low Ticket',  'Orcamento': 200.0},   # Mid-Low, Budget Baixo
    {'Nome': 'Cenário 3',  'Regiao': 'North America', 'Tier': 'Low Ticket',  'Orcamento': 1000.0},  # Low-Mid, Budget Médio
    {'Nome': 'Cenário 4',  'Regiao': 'North America', 'Tier': 'Low Ticket',  'Orcamento': 1000.0},  # Mid-Low, Budget Médio (Repetido propositalmente para ver variação se houver, ou mude o Tier)
    {'Nome': 'Cenário 5',  'Regiao': 'North America', 'Tier': 'Low Ticket',  'Orcamento': 10000.0}, # Low-Mid, Budget Alto
    {'Nome': 'Cenário 6',  'Regiao': 'North America', 'Tier': 'Low Ticket',  'Orcamento': 10000.0}, # Mid-Low, Budget Alto

    # --- HIGH TICKET ---
    {'Nome': 'Cenário 7',  'Regiao': 'North America', 'Tier': 'High Ticket', 'Orcamento': 2000.0},  # High Entry, Budget Baixo
    {'Nome': 'Cenário 8',  'Regiao': 'North America', 'Tier': 'High Ticket', 'Orcamento': 5000.0},  # High Mid, Budget Médio
    {'Nome': 'Cenário 9',  'Regiao': 'North America', 'Tier': 'High Ticket', 'Orcamento': 10000.0}, # High Premium, Budget Médio
    {'Nome': 'Cenário 10', 'Regiao': 'North America', 'Tier': 'High Ticket', 'Orcamento': 50000.0}, # High Premium, Budget Alto
    {'Nome': 'Cenário 11', 'Regiao': 'North America', 'Tier': 'High Ticket', 'Orcamento': 100000.0},# Ultra High, Budget Alto
    {'Nome': 'Cenário 12', 'Regiao': 'North America', 'Tier': 'High Ticket', 'Orcamento': 200000.0} # Enterprise, Budget Muito Alto
]

# Configuração padrão para campos que não mudam na tabela (ceteris paribus)
BASE_CONFIG = {
    'Plataforma': 'Instagram',
    'Idade': '25-34',
    'Genero': 'Female',
    'Conteudo': 'Video',
    'Tipo_Produto': 'InfoProduto',
    'Modelo_Cobranca': 'Venda Unica',
    'Complexidade_Oferta': 'Media'
}

# --- 2. CARREGAR O AGENTE (versão canônica: treinar_cql.py) ---
print("Carregando modelo...", end=" ")
try:
    cql = pol.CQL.carregar("cql_venda_unica")
    print("✅ Sucesso!")
except Exception as e:
    print(f"\n❌ Erro ao carregar cql_venda_unica: {e} (rode treinar_cql.py)")
    raise SystemExit(1)

# --- 3. O LOOP DE INFERÊNCIA ---
print(f"\n{'CENÁRIO':<10} | {'TIER':<12} | {'ORÇAMENTO':<10} | {'PREÇO REC. (IA)':<15} | {'LUCRO ESPERADO':<15}")
print("-" * 75)

for linha in CENARIOS_TABELA:
    estado = pd.DataFrame([{**BASE_CONFIG, **{k: v for k, v in linha.items() if k != 'Nome'}}])
    preco = float(cql.precos(estado)[0])
    lucro = float(cql.quantis_reais(estado, [preco]).mean())  # média dos quantis do crítico
    print(f"{linha['Nome']:<10} | {linha['Tier']:<12} | ${linha['Orcamento']:<9,.0f} | ${preco:<14.2f} | ${lucro:<14.2f}")

print("-" * 75)
