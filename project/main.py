"""
LOCAC - API de precificação.

Quatro métodos, todos atrás da mesma interface (politicas.Politica.precos):
  rl       CQL canônico (treinar_cql.py: cql_venda_unica / cql_assinatura)
  sl       regressor supervisionado + grade de preços (SL_FINAL)
  bandido  bandido contextual LightGBM (treinar_bandido.py)
  bc       behavior cloning, imitação do histórico (treinar_bc.py)

Toda recomendação passa pelo MESMO caminho: preço do método -> regra de faixa
(KBS) -> lucro previsto pelo próprio método (quando ele tem modelo de lucro)
-> avaliação comum: lucro do SL e risco (VaR/CVaR 5%) dos quantis do crítico
do RL, ambos no preço final. Assim os métodos são comparáveis entre si.

Segurança mínima (protótipo acadêmico):
  - LOCAC_API_KEY: chave exigida no header X-API-Key em todas as rotas, exceto
    GET /. Sem a variável, uma chave aleatória é gerada e impressa no início.
  - LOCAC_CORS_ORIGINS: origens permitidas, separadas por vírgula
    (padrão: o servidor de desenvolvimento do Vite em localhost:5173).
  - Entradas validadas (categorias fechadas, limites numéricos).
  - /configure_market exige a chave e recusa um retreino enquanto outro roda.
Não há autenticação de usuários nem proteção de dados: ver Limitações.
"""

import json
import os
import secrets
import subprocess
import sys
import threading
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Literal, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field, model_validator

import bandido  # noqa: F401  (classes necessárias para carregar bandido_*.joblib)
import politicas as pol
import simulador as sim

# --- 1. Configuração e segurança --------------------------------------------
API_KEY = os.environ.get("LOCAC_API_KEY") or secrets.token_urlsafe(24)
if "LOCAC_API_KEY" not in os.environ:
    print(f"⚠️  LOCAC_API_KEY não definida; chave desta execução: {API_KEY}")
CORS_ORIGINS = [o.strip() for o in os.environ.get(
    "LOCAC_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",") if o.strip()]

@asynccontextmanager
async def ciclo_de_vida(_app):
    load_models()  # definida abaixo; roda quando o servidor sobe
    yield


app = FastAPI(title="LOCAC API de Precificação", lifespan=ciclo_de_vida)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key"],
)


def exigir_chave(x_api_key: Optional[str] = Header(default=None)):
    if not x_api_key or not secrets.compare_digest(x_api_key, API_KEY):
        raise HTTPException(status_code=401, detail="Chave de API ausente ou inválida (header X-API-Key).")


METODOS = ("rl", "sl", "bandido", "bc")
COBRANCAS = ("venda_unica", "assinatura")
Metodo = Literal["rl", "sl", "bandido", "bc"]
Cobranca = Literal["venda_unica", "assinatura"]

models_state: Dict[str, Any] = {}
retreino_em_andamento = threading.Event()


# --- 2. Modelos de entrada e saída -------------------------------------------
class CampaignInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    Regiao: Literal[tuple(sim.REGIOES)]
    Plataforma: Literal[tuple(sim.PLATAFORMAS)]
    Tier: Literal["Low Ticket", "High Ticket"]
    Orcamento: float = Field(gt=0, le=1_000_000)
    Idade: str = Field(default=sim.CATEGORICAS_FIXAS["Idade"], max_length=16)
    Genero: str = Field(default=sim.CATEGORICAS_FIXAS["Genero"], max_length=16)
    Conteudo: str = Field(default=sim.CATEGORICAS_FIXAS["Conteudo"], max_length=16)
    Tipo_Produto: str = Field(default=sim.CATEGORICAS_FIXAS["Tipo_Produto"], max_length=32)
    Modelo_Cobranca: str = Field(default=sim.CATEGORICAS_FIXAS["Modelo_Cobranca"], max_length=32)
    Complexidade_Oferta: str = Field(default=sim.CATEGORICAS_FIXAS["Complexidade_Oferta"], max_length=16)
    # Memória da assinatura. Ausente -> média do treino (padronizada vira 0).
    dias_desde_ultima_interacao: Optional[float] = Field(default=None, ge=0, le=10_000)
    clv_estimate_percentile: Optional[float] = Field(default=None, ge=0, le=1)
    avg_price_offered_segment_90d: Optional[float] = Field(default=None, ge=0, le=1_000_000)
    price_volatility_30d: Optional[float] = Field(default=None, ge=0, le=1_000)


class Recomendacao(BaseModel):
    metodo: str
    cobranca: str
    preco_recomendado: float
    preco_bruto: float                       # saída do método antes da KBS
    kbs_applied: bool
    lucro_previsto_metodo: Optional[float]   # modelo de lucro do próprio método (BC não tem)
    lucro_estimado_sl: Optional[float]       # avaliação comum: SL no preço final
    var_5_percent: Optional[float]           # avaliação comum: quantis do crítico do RL
    cvar_5_percent: Optional[float]
    latencia_ms: float


class MarketConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lowMin: float = Field(gt=0, le=1_000_000)
    lowMax: float = Field(gt=0, le=1_000_000)
    highMin: float = Field(gt=0, le=1_000_000)
    highMax: float = Field(gt=0, le=1_000_000)
    budgetMin: float = Field(gt=0, le=10_000_000)
    budgetMax: float = Field(gt=0, le=10_000_000)

    @model_validator(mode="after")
    def faixas_validas(self):
        for lo, hi in [("lowMin", "lowMax"), ("highMin", "highMax"), ("budgetMin", "budgetMax")]:
            if getattr(self, lo) >= getattr(self, hi):
                raise ValueError(f"{lo} deve ser menor que {hi}")
        return self


# --- 3. Regra de segurança de preço (KBS), igual para todos os métodos -------
# Faixas por Tier (padrões do painel "Gêmeo Digital"); config_market.json tem prioridade.
DEFAULT_PRICE_RANGES = {
    "Low Ticket": {"min": 10.0, "max": 97.0},
    "High Ticket": {"min": 497.0, "max": 5000.0},
}


def get_price_range(tier: str) -> Dict[str, float]:
    ranges = DEFAULT_PRICE_RANGES
    try:
        with open("config_market.json") as f:
            ranges = {**ranges, **json.load(f).get("price_ranges", {})}
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    if tier not in ranges:
        raise ValueError(f"Tier desconhecido: {tier}")
    return ranges[tier]


def apply_kbs(preco: float, tier: str):
    """Limita o preço à faixa do Tier. Retorna (preço, se a regra alterou o valor)."""
    faixa = get_price_range(tier)
    limitado = float(np.clip(preco, faixa["min"], faixa["max"]))
    return limitado, limitado != float(preco)


# --- 4. Carregamento dos modelos ---------------------------------------------
ARQUIVOS_SL = {"venda_unica": "sl_profit_regressor_model.joblib", "assinatura": "sl_ltv_regressor_model.joblib"}


def load_models():
    """Carrega cada método de forma independente: um arquivo ausente desativa só ele."""
    novo: Dict[str, Any] = {"metodos": {c: {} for c in COBRANCAS}, "erros": {}}
    try:
        # Média da memória da assinatura (valor padrão quando a tela não envia)
        novo["scaler_assinatura_memoria"] = joblib.load("scaler_assinatura_memoria.joblib")
    except FileNotFoundError as e:
        novo["erros"]["base"] = f"arquivo ausente: {e.filename} (rode train_pipeline.py)"
        models_state.clear()
        models_state.update(novo)
        print(f"❌ {novo['erros']['base']}")
        return

    for cob in COBRANCAS:
        carregadores = {
            "rl": lambda: pol.CQL.carregar(f"cql_{cob}"),
            "sl": lambda: joblib.load(ARQUIVOS_SL[cob]),
            "bandido": lambda: joblib.load(f"bandido_{cob}.joblib"),
            "bc": lambda: pol.BC.carregar(f"bc_{cob}"),
        }
        for metodo, carregar in carregadores.items():
            try:
                novo["metodos"][cob][metodo] = carregar()
            except Exception as e:  # arquivo ausente ou incompatível
                novo["erros"][f"{metodo}/{cob}"] = str(e)

    models_state.clear()
    models_state.update(novo)
    carregados = {c: sorted(m) for c, m in novo["metodos"].items()}
    print(f"✅ Métodos carregados: {carregados}")
    for chave, erro in novo["erros"].items():
        print(f"   ⚠️  {chave}: {erro}")


# --- 5. Recomendação ---------------------------------------------------------
def estado_df(input_data: CampaignInput, cobranca: str) -> pd.DataFrame:
    """Estado em uma linha de DataFrame (entrada comum a todos os métodos)."""
    linha = input_data.model_dump()
    if cobranca == "assinatura":
        scaler_memoria = models_state["scaler_assinatura_memoria"]
        for col, media in zip(scaler_memoria.feature_names_in_, scaler_memoria.mean_):
            if linha.get(col) is None:
                linha[col] = float(media)
    return pd.DataFrame([linha])


def politica(metodo: str, cobranca: str):
    p = models_state.get("metodos", {}).get(cobranca, {}).get(metodo)
    if p is None:
        motivo = models_state.get("erros", {}).get(f"{metodo}/{cobranca}") or models_state.get("erros", {}).get("base")
        raise HTTPException(status_code=503, detail=f"Método '{metodo}' indisponível para {cobranca}: {motivo}")
    return p


def risco(cobranca: str, estados: pd.DataFrame, preco: float):
    """VaR/CVaR 5% do lucro no preço dado, pelos quantis do crítico do RL (avaliador comum)."""
    rl = models_state.get("metodos", {}).get(cobranca, {}).get("rl")
    if rl is None:
        return None, None
    quantis = rl.quantis_reais(estados, [preco]).ravel()
    var_5 = float(np.quantile(quantis, 0.05))
    return var_5, float(quantis[quantis <= var_5].mean())


def lucro_previsto(p, cobranca: str, estados: pd.DataFrame, preco: float) -> Optional[float]:
    if hasattr(p, "prever_lucro"):          # SL e bandido
        return float(p.prever_lucro(estados, [preco])[0])
    if isinstance(p, pol.CQL):              # média dos quantis do crítico
        return float(p.quantis_reais(estados, [preco]).mean())
    return None                             # BC não tem modelo de lucro


def recomendar(metodo: str, cobranca: str, input_data: CampaignInput) -> Recomendacao:
    inicio = time.perf_counter()
    p = politica(metodo, cobranca)
    estados = estado_df(input_data, cobranca)
    try:
        preco_bruto = float(p.precos(estados)[0])
        preco, kbs = apply_kbs(preco_bruto, input_data.Tier)
        sl = models_state["metodos"][cobranca].get("sl")
        var_5, cvar_5 = risco(cobranca, estados, preco)
        return Recomendacao(
            metodo=metodo, cobranca=cobranca, preco_recomendado=preco, preco_bruto=preco_bruto,
            kbs_applied=kbs, lucro_previsto_metodo=lucro_previsto(p, cobranca, estados, preco),
            lucro_estimado_sl=float(sl.prever_lucro(estados, [preco])[0]) if sl is not None else None,
            var_5_percent=var_5, cvar_5_percent=cvar_5,
            latencia_ms=(time.perf_counter() - inicio) * 1000,
        )
    except Exception as e:
        print(f"Erro em {metodo}/{cobranca}: {e}")
        raise HTTPException(status_code=500, detail=f"Erro ao recomendar com '{metodo}': {e}")


# --- 6. Rotas ----------------------------------------------------------------
@app.post("/recomendar", response_model=Recomendacao, dependencies=[Depends(exigir_chave)])
def rota_recomendar(input_data: CampaignInput, metodo: Metodo = Query("rl"),
                    cobranca: Cobranca = Query("venda_unica")):
    return recomendar(metodo, cobranca, input_data)


@app.post("/comparar", response_model=List[Recomendacao], dependencies=[Depends(exigir_chave)])
def rota_comparar(input_data: CampaignInput, cobranca: Cobranca = Query("venda_unica")):
    """Uma recomendação por método disponível, para a exibição comparada na tela."""
    disponiveis = [m for m in METODOS if m in models_state.get("metodos", {}).get(cobranca, {})]
    if not disponiveis:
        raise HTTPException(status_code=503, detail="Nenhum método carregado (rode train_pipeline.py).")
    return [recomendar(m, cobranca, input_data) for m in disponiveis]


# Rotas antigas, mantidas por compatibilidade (agora aceitam ?metodo=)
@app.post("/recommend_price", response_model=Recomendacao, dependencies=[Depends(exigir_chave)])
def recommend_price(input_data: CampaignInput, metodo: Metodo = Query("rl")):
    return recomendar(metodo, "venda_unica", input_data)


@app.post("/recommend_subscription_price", response_model=Recomendacao, dependencies=[Depends(exigir_chave)])
def recommend_subscription_price(input_data: CampaignInput, metodo: Metodo = Query("rl")):
    return recomendar(metodo, "assinatura", input_data)


def run_retraining_pipeline():
    """Roda o pipeline de treino e recarrega os modelos ao final."""
    try:
        print("🔄 [BACKGROUND] Iniciando pipeline de atualização...")
        subprocess.check_call([sys.executable, "train_pipeline.py"])
        load_models()
        print("✅ [BACKGROUND] Pipeline concluído e modelos recarregados.")
    except Exception as e:
        print(f"❌ [BACKGROUND] Erro no pipeline: {e}")
    finally:
        retreino_em_andamento.clear()


@app.post("/configure_market", dependencies=[Depends(exigir_chave)])
def configure_market(config: MarketConfig, background_tasks: BackgroundTasks):
    if retreino_em_andamento.is_set():
        raise HTTPException(status_code=409, detail="Já há um retreino em andamento.")
    novas = {
        "price_ranges": {
            "Low Ticket": {"min": config.lowMin, "max": config.lowMax},
            "High Ticket": {"min": config.highMin, "max": config.highMax},
        },
        "budget_range": {"min": config.budgetMin, "max": config.budgetMax},
    }
    try:
        with open("config_market.json", "w") as f:
            json.dump(novas, f, indent=4)
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"Erro ao salvar config: {e}")
    retreino_em_andamento.set()
    background_tasks.add_task(run_retraining_pipeline)
    return {"status": "accepted", "message": "Configuração salva. Re-treinamento iniciado em background."}


@app.get("/")
def read_root():
    return {
        "status": "LOCAC API Online",
        "metodos": {c: sorted(m) for c, m in models_state.get("metodos", {}).items()},
        "retreino_em_andamento": retreino_em_andamento.is_set(),
    }
