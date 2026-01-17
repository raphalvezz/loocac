import os
import time
import json
import joblib
import d3rlpy
import numpy as np
import pandas as pd
import uvicorn
import subprocess
import sys
from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List

# --- 1. Inicialização do App FastAPI (ISSO DEVE VIR PRIMEIRO) ---
app = FastAPI(title="LOCAC API de Precificação")

# Configuração do CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 2. Definição do Estado Global ---
models_state: Dict[str, Any] = {}

# --- 3. Modelos de Entrada (Pydantic) ---
class CampaignInput(BaseModel):
    Regiao: str
    Plataforma: str
    Tier: str
    Orcamento: float
    Idade: str
    Genero: str
    Conteudo: str
    # Features de Assinatura opcionais
    dias_desde_ultima_interacao: float = 0.0
    clv_estimate_percentile: float = 0.0
    avg_price_offered_segment_90d: float = 0.0
    price_volatility_30d: float = 0.0

class PredictionResponse(BaseModel):
    modelo: str
    preco_recomendado: float
    lucro_estimado_sl: float
    var_5_percent: float
    cvar_5_percent: float
    latencia_ms: float
    kbs_applied: bool = False
    llm_explanation: str = "Explicação do LLM ainda não implementada."

class MarketConfig(BaseModel):
    lowMin: float
    lowMax: float
    highMin: float
    highMax: float
    budgetMin: float
    budgetMax: float

# --- 4. Funcionalidades de Re-treino (Dinâmico) ---

def run_retraining_pipeline():
    """Função background que executa o pipeline de treino"""
    try:
        print("🔄 [BACKGROUND] Iniciando pipeline de atualização...")
        # Executa o script train_pipeline.py
        # (Certifique-se de que você criou este arquivo na pasta project/)
        subprocess.check_call([sys.executable, "train_pipeline.py"])
        print("✅ [BACKGROUND] Pipeline concluído com sucesso.")
        
        # Opcional: Recarregar modelos aqui seria o ideal, mas requer 
        # reiniciar o worker ou ter uma função de reload segura.
        # Para este MVP, o usuário perceberá a mudança no próximo restart ou request
        # se o load_models for chamado dinamicamente (não implementado aqui para simplicidade).
    except Exception as e:
        print(f"❌ [BACKGROUND] Erro crítico no pipeline: {e}")

@app.post("/configure_market")
async def configure_market(config: MarketConfig, background_tasks: BackgroundTasks):
    print(f"📝 Recebendo nova configuração de mercado: {config}")
    
    # 1. Salvar JSON para o Generator_NEW ler
    new_settings = {
        "price_ranges": {
            "Low Ticket": {"min": config.lowMin, "max": config.lowMax},
            "High Ticket": {"min": config.highMin, "max": config.highMax}
        },
        "budget_range": {"min": config.budgetMin, "max": config.budgetMax},
        "cpa_ratio_range": [0.20, 0.45]
    }
    
    try:
        with open("config_market.json", "w") as f:
            json.dump(new_settings, f, indent=4)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao salvar config: {e}")
        
    # 2. Disparar pipeline em segundo plano
    background_tasks.add_task(run_retraining_pipeline)
    
    return {"status": "accepted", "message": "Configuração salva. Re-treinamento iniciado em background."}

# --- 5. Evento de Startup (Carregar os Modelos) ---
@app.on_event("startup")
async def load_models():
    print("Iniciando servidor...")
    global models_state
    
    artifact_paths = {
        "cql_venda_unica": "modelo_rl_final.pt",
        "cql_assinatura": "modelo_rl_assinatura.pt",
        "sl_profit": "sl_profit_regressor_model.joblib",
        "ohe": "ohe_encoder.joblib",
        "scaler_estado": "scaler_estado.joblib",
        "scaler_acao": "scaler_acao.joblib",
        "scaler_recompensa": "scaler_recompensa.joblib",
        "scaler_assinatura_memoria": "scaler_assinatura_memoria.joblib",
        "scaler_assinatura_recompensa": "scaler_assinatura_recompensa.joblib",
        "colunas_estado_base": "colunas_estado_base.json",
        "colunas_estado_assinatura": "colunas_estado_assinatura.json",
    }
    
    print("Carregando artefatos de IA...")
    try:
        # Carrega metadados
        with open(artifact_paths["colunas_estado_base"], 'r') as f:
            models_state["colunas_estado_base"] = json.load(f)
        with open(artifact_paths["colunas_estado_assinatura"], 'r') as f:
            models_state["colunas_estado_assinatura"] = json.load(f)
            
        # Carrega Scalers
        models_state["ohe"] = joblib.load(artifact_paths["ohe"])
        models_state["scaler_estado"] = joblib.load(artifact_paths["scaler_estado"])
        models_state["scaler_acao"] = joblib.load(artifact_paths["scaler_acao"])
        models_state["scaler_recompensa"] = joblib.load(artifact_paths["scaler_recompensa"])
        models_state["scaler_assinatura_memoria"] = joblib.load(artifact_paths["scaler_assinatura_memoria"])
        models_state["scaler_assinatura_recompensa"] = joblib.load(artifact_paths["scaler_assinatura_recompensa"])

        # Carrega Modelos
        models_state["sl_profit"] = joblib.load(artifact_paths["sl_profit"])
        models_state["cql_venda_unica"] = d3rlpy.load_learnable(artifact_paths["cql_venda_unica"], device="cpu")
        models_state["cql_assinatura"] = d3rlpy.load_learnable(artifact_paths["cql_assinatura"], device="cpu")

        print("✅ SUCESSO: Todos os modelos carregados.")

    except FileNotFoundError as e:
        print(f"❌ ERRO FATAL: Arquivo não encontrado: {e.filename}")
        # Não damos raise aqui para permitir que o servidor suba e receba a config inicial
        # mas os endpoints de inferência vão falhar se chamados.
    except Exception as e:
        print(f"❌ ERRO FATAL inesperado: {e}")

# --- 6. Função de Pré-processamento ---
def preprocess_input(input_data: CampaignInput, feature_type: str) -> np.ndarray:
    if "ohe" not in models_state:
        raise ValueError("Modelos não carregados. Configure o mercado primeiro.")

    ohe = models_state["ohe"]
    scaler_estado = models_state["scaler_estado"]
    colunas_estado = models_state["colunas_estado_base"]
    
    if feature_type == "assinatura":
        scaler_memoria = models_state["scaler_assinatura_memoria"]
        colunas_estado = models_state["colunas_estado_assinatura"]

    input_dict = input_data.model_dump()
    df = pd.DataFrame([input_dict])

    # Transformações
    categorical_cols = ohe.feature_names_in_
    df_categ = pd.DataFrame(ohe.transform(df[categorical_cols]).toarray(), columns=ohe.get_feature_names_out())
    
    numeric_cols_base = scaler_estado.feature_names_in_
    df_numeric_base = pd.DataFrame(scaler_estado.transform(df[numeric_cols_base]), columns=numeric_cols_base)

    df_processed = pd.concat([df_categ, df_numeric_base], axis=1)

    if feature_type == "assinatura":
        numeric_cols_memoria = scaler_memoria.feature_names_in_
        df_numeric_memoria = pd.DataFrame(scaler_memoria.transform(df[numeric_cols_memoria]), columns=numeric_cols_memoria)
        df_processed = pd.concat([df_processed, df_numeric_memoria], axis=1)

    df_final_state = df_processed.reindex(columns=colunas_estado, fill_value=0)
    return df_final_state.to_numpy().astype(np.float32)

# --- 7. Endpoints de Inferência ---

@app.post("/recommend_price", response_model=PredictionResponse)
async def recommend_price(input_data: CampaignInput):
    start_time = time.time()
    try:
        state_vector = preprocess_input(input_data, feature_type="venda_unica")
        
        # RL Prediction
        action_norm = models_state["cql_venda_unica"].predict(state_vector)[0]
        preco_real = models_state["scaler_acao"].inverse_transform(action_norm.reshape(1, -1))[0][0]
        
        # Risco
        quantis_norm = models_state["cql_venda_unica"].predict_value(state_vector, action_norm.reshape(1, -1))[0]
        quantis_reais = models_state["scaler_recompensa"].inverse_transform(quantis_norm.reshape(1, -1)).flatten()
        
        var_5 = np.percentile(quantis_reais, 5)
        cvar_5 = quantis_reais[quantis_reais <= var_5].mean()

        # SL Prediction
        lucro_sl = models_state["sl_profit"].predict(state_vector)[0]

        return PredictionResponse(
            modelo="RL (Venda Única) Dinâmico",
            preco_recomendado=float(preco_real),
            lucro_estimado_sl=float(lucro_sl),
            var_5_percent=float(var_5),
            cvar_5_percent=float(cvar_5),
            latencia_ms=(time.time() - start_time) * 1000
        )
    except Exception as e:
        print(f"Erro: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/recommend_subscription_price", response_model=PredictionResponse)
async def recommend_subscription_price(input_data: CampaignInput):
    start_time = time.time()
    try:
        state_vector = preprocess_input(input_data, feature_type="assinatura")
        
        action_norm = models_state["cql_assinatura"].predict(state_vector)[0]
        preco_real = models_state["scaler_assinatura_recompensa"].inverse_transform(action_norm.reshape(1, -1))[0][0]
        
        quantis_norm = models_state["cql_assinatura"].predict_value(state_vector, action_norm.reshape(1, -1))[0]
        quantis_reais = models_state["scaler_assinatura_recompensa"].inverse_transform(quantis_norm.reshape(1, -1)).flatten()
        
        var_5 = np.percentile(quantis_reais, 5)
        cvar_5 = quantis_reais[quantis_reais <= var_5].mean()
        
        lucro_sl = models_state["sl_profit"].predict(state_vector)[0]

        return PredictionResponse(
            modelo="RL (Assinatura) LTV",
            preco_recomendado=float(preco_real),
            lucro_estimado_sl=float(lucro_sl),
            var_5_percent=float(var_5),
            cvar_5_percent=float(cvar_5),
            latencia_ms=(time.time() - start_time) * 1000
        )
    except Exception as e:
        print(f"Erro: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"status": "LOCAC API Online", "models_loaded": "cql_venda_unica" in models_state}