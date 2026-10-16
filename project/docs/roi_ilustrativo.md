# ROI ilustrativo (hipotético)

> **Aviso:** exemplo numérico para dar escala aos resultados, não uma estimativa de
> retorno. Os ganhos vêm do simulador (efeito intertemporal calibrado por inspeção,
> seção 9f do protocolo) e não foram validados com dados reais.

## Por que não usar os valores em US$ do simulador

O simulador não foi calibrado em dinheiro: os lucros absolutos por campanha são
irreais. Uma campanha High Ticket de assinatura com orçamento médio de US$ 61 mil tem
lucro ótimo de US$ 412 milhões (LTV de 12 períodos); na venda única, US$ 12,8 milhões.
Só as razões entre políticas (% do ótimo) têm significado. Por isso o exemplo abaixo
parte de uma empresa hipotética e aplica as porcentagens medidas.

## Premissa

Uma empresa cujo lucro anual atribuível às campanhas precificadas pelo sistema seria
de **R$ 1.000.000 se ela sempre cobrasse o preço ótimo** (o ótimo da programação
dinâmica, no ambiente sequencial; o oráculo, no estático). O valor é arbitrário:
o ganho escala linearmente com ele.

## Ambiente sequencial (o preço de hoje afeta a demanda de amanhã)

10 sementes, `resultados/sequencial/` (% do ótimo, total):

| a empresa precifica hoje com... | lucro anual | com o CQL γ = 0,95 (84,2% / 98,3%) | ganho |
|---|---|---|---|
| venda única, bandido contextual (75,4%) | R$ 754.000 | R$ 842.000 | **+R$ 88.000 (+11,6%)** |
| venda única, regra fixa do meio da faixa (73,6%) | R$ 736.000 | R$ 842.000 | +R$ 106.000 (+14,4%) |
| assinatura, bandido contextual (84,4%) | R$ 844.000 | R$ 983.000 | **+R$ 139.000 (+16,5%)** |
| assinatura, regra fixa do meio da faixa (79,1%) | R$ 791.000 | R$ 983.000 | +R$ 192.000 (+24,3%) |

O ganho relativo é o número que se sustenta independentemente da premissa: no
ambiente sequencial, trocar o bandido pelo CQL que planeja aumentaria o lucro em
cerca de 12% (venda única) a 17% (assinatura).

## Problema estático (decisões independentes)

10 sementes, `resultados/final_10_sementes/`:

| | bandido | CQL | ganho com a mesma premissa |
|---|---|---|---|
| venda única | 82,7% | 84,9% | +R$ 22.000 (+2,7%) |
| assinatura | 99,0% | 98,9% | −R$ 1.000 (−0,1%) |

Sem efeito intertemporal, o CQL quase não acrescenta ao bandido: é o critério de
decisão de `docs/escopo_produto.md` em números.

## Texto para a monografia

> Para dar escala aos resultados, considere uma empresa hipotética cujo lucro anual
> nas campanhas seria de R$ 1 milhão com preços ótimos. No ambiente sequencial, em
> que o preço afeta a demanda futura, o CQL com γ = 0,95 obteria R$ 842 mil (venda
> única) contra R$ 754 mil do bandido contextual, um ganho de R$ 88 mil (+11,6%); na
> assinatura, R$ 983 mil contra R$ 844 mil (+16,5%). No problema de decisões
> independentes, o ganho cairia para R$ 22 mil (+2,7%) na venda única e seria nulo na
> assinatura. Os valores são ilustrativos: dependem da intensidade do efeito
> intertemporal, calibrada no simulador e não medida em dados reais.
