"""
Conjuntos de cenários para o teste de sensibilidade ao número de contextos
(contexto = cenário × 4 regiões × 3 plataformas).

  padrao     os 12 cenários da tabela do artigo (simulador v1.0) -> 144 contextos
  reduzido   5 cenários (0, 2, 3, 7, 10) -> 60 contextos. Mantém um par de cenários
             que dividem a mesma chave de estado na venda única (2 e 3, orçamento
             1.000), como no conjunto padrão (2/3 e 4/5)
  ampliado   os 12 originais + duas variantes de cada -> 36 cenários, 432 contextos.
             Variante "maior": orçamento × 1,7, faixa de preço e CPA × 1,2; variante
             "menor": orçamento × 0,6, faixa e CPA × 0,8. Os fatores foram escolhidos
             para que nenhum orçamento novo coincida com um original no mesmo Tier;
             os pares que dividem a chave nos originais continuam dividindo nas
             variantes (mesma proporção de estados ambíguos)

A economia (demanda, ruído, multiplicadores de região e plataforma) é a do
simulador v1.0; só a lista de cenários muda. Com 5.000 linhas por cenário, o volume
de dados acompanha o número de contextos.
"""

import simulador as sim

REDUZIDO = (0, 2, 3, 7, 10)
VARIANTES = ((1.7, 1.2), (0.6, 0.8))  # (fator do orçamento, fator da faixa de preço e do CPA)
CONJUNTOS = ('padrao', 'reduzido', 'ampliado')


def conjunto(nome):
    base = sim.CENARIOS_ARTIGO
    if nome == 'padrao':
        return list(base)
    if nome == 'reduzido':
        return [base[i] for i in REDUZIDO]
    if nome == 'ampliado':
        novos = [{**c, 'Budget': c['Budget'] * fo, 'Price_Min': c['Price_Min'] * fp,
                  'Price_Max': c['Price_Max'] * fp, 'CPA_Target': c['CPA_Target'] * fp}
                 for fo, fp in VARIANTES for c in base]
        return list(base) + novos
    raise ValueError(f'conjunto desconhecido: {nome} (use {CONJUNTOS})')
