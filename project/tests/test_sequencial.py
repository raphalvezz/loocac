"""Ambiente sequencial (sequencial.py): calibração e consistência da programação dinâmica."""
import numpy as np
import pytest

import sequencial as sq
import simulador as sim

C = sim.CENARIOS_ARTIGO


@pytest.fixture(scope='module', params=sq.MODELOS)
def avaliador(request):
    return sq.Avaliador(C, request.param)


def test_pd_e_o_teto(avaliador):
    pd_ = avaliador.pct(sq.OraculoPD(C, avaliador.modelo))
    miope = avaliador.pct(sq.OraculoMiope(C, avaliador.modelo))
    assert pd_.mean() == pytest.approx(100, abs=0.1)
    assert np.all(miope <= pd_ + 1e-6)


@pytest.mark.parametrize('modelo, miope_pct', [('venda_unica', 89.7), ('assinatura', 82.7)])
def test_calibracao_registrada(modelo, miope_pct):
    """Planejar vale ~10 p.p. (venda) e ~17 p.p. (assinatura): ver docs/protocolo_avaliacao.md, 9b."""
    av = sq.Avaliador(C, modelo)
    assert av.pct(sq.OraculoMiope(C, modelo)).mean() == pytest.approx(miope_pct, abs=0.1)


def test_dados_por_cobranca():
    venda = sq.gerar_dados(C[:1], 5, 0, 'venda_unica')
    assin = sq.gerar_dados(C[:1], 5, 0, 'assinatura')
    assert len(venda) == len(assin) == 5 * sq.T
    assert (venda['LTV_Real'] == 0).all() and (venda['Lucro_Real'] > 0).all()
    assert (assin['Lucro_Real'] == 0).all() and (assin['LTV_Real'] > 0).all()
    # episódios inteiros e em ordem de período
    assert (venda.groupby('episodio')['passo'].apply(list) == [list(range(sq.T))] * 5).all()
