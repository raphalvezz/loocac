import numpy as np
import pandas as pd
import pytest

import avaliar_politicas as av
import bandido as ban
import politicas as pol
import simulador as sim

C = sim.CENARIOS_ARTIGO


@pytest.fixture(scope='module')
def estados():
    return sim.contextos(C)


@pytest.mark.parametrize("modelo", ['venda_unica', 'assinatura'])
def test_heuristicas_ficam_na_faixa_observavel(modelo, estados):
    for cls in [pol.Aleatoria, pol.MeioFaixa, pol.MaximoFaixa]:
        p = cls(modelo, C)
        p.treinar(None, 0)
        precos = av.consultar(p, estados)
        for preco, (lo, hi) in zip(precos, p.faixas(estados)):
            assert lo <= preco <= hi


def test_nao_privilegiada_nao_ve_o_cenario(estados):
    class Espia(pol.Politica):
        def precos(self, e):
            assert 'Cenario' not in e.columns
            return np.ones(len(e))
    av.consultar(Espia('venda_unica', C), estados)


def test_chave_estado():
    # Venda única não distingue os cenários 3/4 (índices 2 e 3); assinatura distingue (memória)
    assert pol.chave_cenario('venda_unica', C[2]) == pol.chave_cenario('venda_unica', C[3])
    assert pol.chave_cenario('assinatura', C[2]) != pol.chave_cenario('assinatura', C[3])


@pytest.mark.parametrize("modelo", ['venda_unica', 'assinatura'])
def test_oraculos_sao_tetos(modelo, estados):
    otimo = av.lucros(av.consultar(pol.Oraculo(modelo, C), estados), estados, C, modelo)
    obs = av.lucros(av.consultar(pol.OraculoObservavel(modelo, C), estados), estados, C, modelo)
    assert np.all(obs <= otimo * (1 + 1e-9))
    # Teto da métrica: em cada grupo de estados indistinguíveis, nenhum preço único
    # passa do oraculo_obs em % médio do ótimo
    grupos = pd.Series(list(zip(pol.chaves(modelo, estados), estados['Regiao'], estados['Plataforma'])))
    for cls in [pol.MeioFaixa, pol.MaximoFaixa]:
        v = av.lucros(av.consultar(cls(modelo, C), estados), estados, C, modelo)
        assert np.all(pd.Series(v / otimo).groupby(grupos).mean().to_numpy()
                      <= pd.Series(obs / otimo).groupby(grupos).mean().to_numpy() + 1e-6)


def test_bandido_aprende(estados):
    dados = sim.gerar_dados(C, 1000, 42)
    b = ban.BandidoLGBM('venda_unica', C)
    b.treinar(dados, 42)
    otimo = av.lucros(av.consultar(pol.Oraculo('venda_unica', C), estados), estados, C, 'venda_unica')
    pct = np.mean(av.lucros(av.consultar(b, estados), estados, C, 'venda_unica') / otimo)
    aleatorio = pol.Aleatoria('venda_unica', C)
    aleatorio.treinar(None, 42)
    pct_aleatorio = np.mean(av.lucros(av.consultar(aleatorio, estados), estados, C, 'venda_unica') / otimo)
    assert pct > pct_aleatorio


def test_ic95():
    media, lo, hi = av.ic95([1.0, 2.0, 3.0])
    assert media == 2.0 and lo < 2.0 < hi
    assert av.ic95([5.0]) == (5.0, 5.0, 5.0)


@pytest.mark.parametrize("modelo", ['venda_unica', 'assinatura'])
def test_sl_prever_lucro_e_joblib(modelo, estados, tmp_path):
    import joblib
    dados = sim.gerar_dados(C, 300, 1)
    sl = pol.SL(modelo, C).treinar(dados, 1)
    lucro = sl.prever_lucro(estados, np.full(len(estados), 50.0))
    assert lucro.shape == (len(estados),) and np.isfinite(lucro).all()
    # A API carrega o objeto inteiro (encoder + LightGBM) do .joblib
    caminho = tmp_path / 'sl.joblib'
    joblib.dump(sl, caminho)
    assert np.allclose(joblib.load(caminho).prever_lucro(estados, np.full(len(estados), 50.0)), lucro)
    precos = av.consultar(sl, estados)
    assert all(lo <= p <= hi for p, (lo, hi) in zip(precos, sl.faixas(estados)))
