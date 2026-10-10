"""Congela o simulador (docs/protocolo_avaliacao.md).

Se algum destes testes falhar, a economia ou a geração de dados mudou e os
resultados do capítulo 5 deixam de ser comparáveis. Só atualize os valores
de referência junto com: VERSAO nova em simulador.py e o motivo registrado
no protocolo.
"""
import numpy as np
import pytest

import simulador as sim

C = sim.CENARIOS_ARTIGO


def test_versao():
    assert sim.VERSAO == "1.0"


@pytest.mark.parametrize("i, regiao, plataforma, modelo, preco, lucro, p_otimo", [
    (0, 'Europe', 'Instagram', 'venda_unica', 15.0, 1026.7652013490215, 15.1887168981453),
    (8, 'Asia', 'LinkedIn', 'venda_unica', 3000.0, 524276.47202903905, 3492.8222984271097),
    (2, 'North America', 'Facebook', 'assinatura', 35.0, 128201.99735718971, 36.73357452893088),
    (11, 'South America', 'Facebook', 'assinatura', 17000.0, 91970707.77774447, 14924.499775282893),
])
def test_economia_congelada(i, regiao, plataforma, modelo, preco, lucro, p_otimo):
    assert sim.lucro_esperado(preco, C[i], regiao, plataforma, modelo) == pytest.approx(lucro, rel=1e-9)
    assert sim.preco_otimo(C[i], regiao, plataforma, modelo)[0] == pytest.approx(p_otimo, rel=1e-9)


def test_dados_congelados():
    d = sim.gerar_dados(C, n_por_cenario=50, semente=7)
    assert len(d) == 600
    primeira = d.iloc[0]
    assert (primeira['Cenario'], primeira['Regiao'], primeira['Plataforma']) == (5, 'Asia', 'LinkedIn')
    assert primeira['Preco_Amostra'] == pytest.approx(62.8490417477251, rel=1e-9)
    assert d['Lucro_Real'].sum() == pytest.approx(280993082.9055091, rel=1e-9)
    assert d['LTV_Real'].sum() == pytest.approx(8976544934.728355, rel=1e-9)
    assert d['Preco_Amostra'].sum() == pytest.approx(1730638.0858002785, rel=1e-9)


def test_gerar_dados_reprodutivel():
    a = sim.gerar_dados(C, 20, 3)
    b = sim.gerar_dados(C, 20, 3)
    assert a.equals(b)
    assert not a.equals(sim.gerar_dados(C, 20, 4))


@pytest.mark.parametrize("modelo", ['venda_unica', 'assinatura'])
def test_otimo_interior(modelo):
    """Propriedade central: o lucro tem ótimo dentro da faixa em todo contexto."""
    for c in C:
        for r in sim.REGIOES:
            for p in sim.PLATAFORMAS:
                p_otimo, _ = sim.preco_otimo(c, r, p, modelo)
                pos = (p_otimo - c['Price_Min']) / (c['Price_Max'] - c['Price_Min'])
                assert 0.02 < pos < 0.98


def test_acao_ida_e_volta():
    faixas = sim.faixas_tier(C)
    for tier, (lo, hi) in faixas.items():
        precos = np.geomspace(lo, hi, 50)
        acoes = sim.preco_para_acao(precos, tier, faixas)
        assert acoes.min() == pytest.approx(-1) and acoes.max() == pytest.approx(1)
        assert sim.acao_para_preco(acoes, tier, faixas) == pytest.approx(precos)


@pytest.fixture
def forma_linear():
    sim.usar_forma('linear')
    yield
    sim.usar_forma('logistica')


@pytest.mark.parametrize("modelo", ['venda_unica', 'assinatura'])
def test_forma_linear_otimo_interior(forma_linear, modelo):
    """A forma alternativa (teste de robustez) também tem ótimo dentro da faixa."""
    for c in C:
        for r in sim.REGIOES:
            for p in sim.PLATAFORMAS:
                p_otimo, _ = sim.preco_otimo(c, r, p, modelo)
                pos = (p_otimo - c['Price_Min']) / (c['Price_Max'] - c['Price_Min'])
                assert 0.02 < pos < 0.98


def test_forma_padrao_e_logistica():
    assert sim.FORMA_DEMANDA == 'logistica'
    with pytest.raises(ValueError):
        sim.usar_forma('cubica')
