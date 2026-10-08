import numpy as np
import pytest

import avaliar_politicas as av
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
    meio = av.lucros(av.consultar(pol.MeioFaixa(modelo, C), estados), estados, C, modelo)
    assert np.all(obs <= otimo * (1 + 1e-9))
    assert np.mean(meio / otimo) <= np.mean(obs / otimo) + 1e-9


def test_bandido_aprende(estados):
    dados = sim.gerar_dados(C, 1000, 42)
    b = pol.BandidoLGBM('venda_unica', C)
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
