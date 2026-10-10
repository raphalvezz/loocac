"""Teste de ponta a ponta da API: artefatos reais (gerador + treinos curtos) e uma
chamada por método e modelo de cobrança. Pulado se o d3rlpy não estiver instalado.
Leva cerca de 1 a 2 minutos em CPU."""
import importlib
import os
import subprocess
import sys

import numpy as np
import pytest

d3rlpy = pytest.importorskip("d3rlpy")
from fastapi.testclient import TestClient  # noqa: E402

PROJETO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHAVE = "chave-de-teste"
ESTADO = {"Regiao": "Europe", "Plataforma": "Instagram", "Tier": "Low Ticket", "Orcamento": 1000}


@pytest.fixture(scope="module")
def cliente(tmp_path_factory):
    pasta = tmp_path_factory.mktemp("artefatos")
    ambiente = {**os.environ, "PYTHONPATH": PROJETO}
    subprocess.run([sys.executable, os.path.join(PROJETO, "Generator_NEW.py")], cwd=pasta,
                   env=ambiente, check=True, capture_output=True)
    anterior = os.getcwd()
    os.chdir(pasta)
    try:
        import joblib
        import pandas as pd

        import avaliar_politicas as av
        import politicas as pol
        import treinar_bandido

        dados = pd.read_csv("sl_dataset_combined.csv")
        cenarios = av.carregar_cenarios()
        joblib.dump(pol.SL("venda_unica", cenarios).treinar(dados, 42), "sl_profit_regressor_model.joblib")
        joblib.dump(pol.SL("assinatura", cenarios).treinar(dados, 42), "sl_ltv_regressor_model.joblib")
        treinar_bandido.main()
        for cob in ["venda_unica", "assinatura"]:
            pol.BC(cob, cenarios, n_passos=300).treinar(dados, 42).salvar(f"bc_{cob}")
            pol.CQL(cob, cenarios, n_passos=200).treinar(dados, 42).salvar(f"cql_{cob}")  # treinar_cql.py, curto

        os.environ["LOCAC_API_KEY"] = CHAVE
        main = importlib.reload(importlib.import_module("main"))
        main.load_models()
        with TestClient(main.app) as c:
            c.main = main
            yield c
    finally:
        os.chdir(anterior)


def _post(cliente, rota, corpo=ESTADO, chave=CHAVE, **params):
    headers = {"X-API-Key": chave} if chave else {}
    return cliente.post(rota, json=corpo, params=params, headers=headers)


@pytest.mark.parametrize("cobranca", ["venda_unica", "assinatura"])
@pytest.mark.parametrize("metodo", ["rl", "sl", "bandido", "bc"])
def test_uma_chamada_por_metodo(cliente, metodo, cobranca):
    r = _post(cliente, "/recomendar", metodo=metodo, cobranca=cobranca)
    assert r.status_code == 200, r.text
    d = r.json()
    assert (d["metodo"], d["cobranca"]) == (metodo, cobranca)
    faixa = cliente.main.get_price_range("Low Ticket")
    assert faixa["min"] <= d["preco_recomendado"] <= faixa["max"]  # KBS igual para todos
    assert d["kbs_applied"] == (d["preco_bruto"] != d["preco_recomendado"])
    assert np.isfinite(d["lucro_estimado_sl"])
    assert d["cvar_5_percent"] <= d["var_5_percent"]
    assert (d["lucro_previsto_metodo"] is None) == (metodo == "bc")


def test_comparar_devolve_todos_os_metodos(cliente):
    r = _post(cliente, "/comparar", cobranca="venda_unica")
    assert r.status_code == 200, r.text
    assert [x["metodo"] for x in r.json()] == ["rl", "sl", "bandido", "bc"]


def test_rota_antiga_continua_funcionando(cliente):
    r = _post(cliente, "/recommend_price", metodo="bandido")
    assert r.status_code == 200 and r.json()["metodo"] == "bandido"


@pytest.mark.parametrize("chave", [None, "errada"])
def test_exige_chave(cliente, chave):
    assert _post(cliente, "/recomendar", chave=chave).status_code == 401
    assert _post(cliente, "/configure_market", corpo={}, chave=chave).status_code == 401


@pytest.mark.parametrize("corpo", [
    {**ESTADO, "Regiao": "Marte"},
    {**ESTADO, "Orcamento": -5},
    {**ESTADO, "campo_extra": 1},
])
def test_valida_entrada(cliente, corpo):
    assert _post(cliente, "/recomendar", corpo=corpo).status_code == 422


def test_configure_market(cliente, monkeypatch):
    main = cliente.main
    monkeypatch.setattr(main, "run_retraining_pipeline", lambda: main.retreino_em_andamento.clear())
    invalida = {"lowMin": 50, "lowMax": 10, "highMin": 497, "highMax": 5000, "budgetMin": 500, "budgetMax": 20000}
    assert _post(cliente, "/configure_market", corpo=invalida).status_code == 422
    valida = {**invalida, "lowMin": 10, "lowMax": 97}
    assert _post(cliente, "/configure_market", corpo=valida).status_code == 200
    main.retreino_em_andamento.set()  # simula um retreino ainda rodando
    try:
        assert _post(cliente, "/configure_market", corpo=valida).status_code == 409
    finally:
        main.retreino_em_andamento.clear()


def test_cors_restrito(cliente):
    def preflight(origem):
        return cliente.options("/recomendar", headers={"Origin": origem, "Access-Control-Request-Method": "POST"})
    assert preflight("http://localhost:5173").headers.get("access-control-allow-origin") == "http://localhost:5173"
    assert "access-control-allow-origin" not in preflight("https://site-qualquer.com").headers


def test_raiz_sem_chave_lista_metodos(cliente):
    r = cliente.get("/")
    assert r.status_code == 200
    assert r.json()["metodos"]["venda_unica"] == ["bandido", "bc", "rl", "sl"]
