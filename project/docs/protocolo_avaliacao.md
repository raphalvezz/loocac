# Protocolo de avaliação das políticas de preço

Este documento fixa **antes de rodar** o simulador, a métrica, as comparações e os
critérios de sucesso do capítulo 5. Resultados que contrariem as hipóteses também
são reportados.

> **Status: rascunho para revisão.** Os critérios da seção 6 e as decisões da
> seção 9 precisam ser confirmados antes da primeira rodada com CQL. Depois disso,
> qualquer mudança entra no histórico (seção 10) com justificativa.

## 1. Simulador

- Arquivo: `simulador.py`, `VERSAO = "1.0"`.
- Congelado pelos testes `tests/test_simulador.py` (valores de referência da
  economia e da geração de dados). Se um teste falhar, o simulador mudou.
- Só se altera para corrigir um **erro**. Procedimento: subir `VERSAO`, atualizar
  os valores de referência dos testes, registrar o motivo na seção 10 e refazer
  todas as rodadas (os números de versões diferentes não se misturam).

## 2. Dados

- Política de coleta: preço uniforme na faixa de cada cenário (`simulador.gerar_dados`).
- 12 cenários × 5.000 amostras = 60.000 linhas por semente; ruído lognormal
  (σ = 0,2) nas conversões.
- Treino: primeiros 80% das linhas (embaralhadas). Os 20% restantes não são usados
  na avaliação, que é feita no simulador.

## 3. Grade de avaliação e métrica

- Estados: cada cenário × região × plataforma = **144 estados** (`simulador.contextos`).
- Para cada estado, a política escolhe um preço; mede-se o **lucro esperado**
  (sem ruído) no simulador.
- Métrica principal: **% do lucro do oráculo**, média sobre os estados, reportada
  por tier (Low, High) e no total. Métrica secundária: lucro médio em US$.

## 4. Políticas

Todas implementam `politicas.Politica` (`treinar(dados, semente)`, `precos(estados)`).

| Política | Treina? | Informação |
|---|---|---|
| `aleatorio` | não (estocástica) | estado; é a política de coleta |
| `meio_faixa` | não | estado |
| `maximo_faixa` | não | estado |
| `sl` (regressor do SL_FINAL + grade) | sim | estado |
| `bandido` (LightGBM + grade, alvo em log) | sim | estado |
| `cql` | sim | estado |
| `oraculo_obs` | não | estado + função de lucro do simulador → **teto real** |
| `oraculo` | não | cenário verdadeiro → **referência dos percentuais** |

"Estado" é o mesmo para todas as políticas não privilegiadas e o avaliador remove a
coluna do cenário antes de consultá-las. A faixa observável de cada estado vem do
catálogo por chave de estado (`politicas.chave_estado`):
- venda única: (Tier, Orçamento). Os cenários 3/4 e 5/6 coincidem;
- assinatura: (Tier, Orçamento, `avg_price_offered_segment_90d`), que separa 3/4 e 5/6.

`sl` e `bandido` são a mesma abordagem (regressor de lucro(estado, preço) + escolha
do preço de maior lucro previsto numa grade de 200 preços na faixa observável).
Diferem só na configuração:

Hiperparâmetros fixados a priori, sem ajuste na grade de avaliação:
- sl: os do `SL_FINAL`, `LGBMRegressor(n_estimators=200, learning_rate=0.1, num_leaves=31)`,
  alvo em US$; features = categóricas + Orçamento + Preço (+ memória na assinatura);
- bandido: `LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=63)`,
  alvo log(lucro), grade de 200 preços (geométrica) na faixa observável;
- CQL: os mesmos dos notebooks (`gamma=0`, `conservative_weight=5`, 64 quantis,
  `batch_size=256`, até 50.000 passos, paciência de 20 épocas de 1.000 passos).

## 5. Sementes e intervalos

- Sementes **42 a 46** (5). Cada semente gera um dataset novo e retreina as
  políticas treináveis.
- Intervalo: IC 95% com t de Student sobre as 5 sementes.
- Comparações **pareadas** por semente (mesmo dataset): `cql − aleatorio`,
  `cql − meio_faixa`, `cql − sl`, `cql − bandido`, `sl − aleatorio`, `sl − meio_faixa`,
  `bandido − aleatorio`, `bandido − meio_faixa`, `bandido − sl`.
  Conclusão: "A > B" se o IC 95% da diferença estiver todo acima de 0; "A < B" se
  todo abaixo; senão, "inconclusivo".

## 6. Critérios de sucesso (a confirmar)

Avaliados no total (Todos) e por tier, separadamente para venda única e assinatura:

1. **C1, aprendizado:** `cql − aleatorio` > 0 (o agente supera a política que gerou os dados).
2. **C2, competitividade:** o IC 95% de `cql − bandido` tem limite inferior acima
   de **−1 p.p.** (não-inferioridade com margem de 1 ponto percentual).
3. **C3, utilidade:** `cql − meio_faixa` > 0 (supera a melhor regra fixa).

Se C2 falhar, o resultado da monografia é que um bandido contextual simples é
suficiente neste problema. É um resultado válido e deve ser reportado como tal.

## 7. Generalização (fora da distribuição de treino)

`avaliar_generalizacao.py` treina `sl` e `bandido` **sem** um grupo inteiro e avalia
só nesse grupo:
- uma região de fora (4 rodadas): a categoria vira um one-hot nulo;
- um cenário de fora (12 rodadas): combinação de orçamento e faixa nunca vista.

Métricas no grupo de fora: R² do lucro previsto nas linhas logadas e % do ótimo
nos estados do grupo. Responde à crítica de que o R² do split aleatório é
circular (mesma distribuição no treino e no teste). Referência: o R² máximo
possível com o ruído do simulador é ~0,95 (o lucro esperado sem ruído não
passa disso), então R² perto de 0,95 não diferencia modelos.

## 8. Custo e explicabilidade

`resultados/custo_explicabilidade.csv`: tempo de treino (média por semente),
latência por recomendação (ms, CPU, média sobre os 144 estados), necessidade de
GPU e grau de explicabilidade.

## 9. Decisões em aberto antes da primeira rodada com CQL

1. **Observabilidade na venda única.** O estado não distingue os cenários 3/4 e
   5/6, então o teto (`oraculo_obs`) no Low Ticket é ~67% do ótimo. Opções: incluir
   um preço de referência do produto no estado (muda a API e a tela) ou manter e
   discutir como limitação.
2. **Pouca folga quando o cenário é identificável.** Com a calibração atual,
   `meio_faixa` já obtém ~96% (venda única, High) e ~97,5% (assinatura) do ótimo.
   A variação de contexto (multiplicadores 0,81–1,16) desloca pouco o preço ótimo,
   então a diferença máxima entre qualquer política aprendida e a regra fixa é de
   ~3 p.p. Opções: ampliar a variação de contexto (simulador v1.1) ou manter e
   reportar que o problema tem pouca folga.

## 10. Histórico

| Versão | Data | Mudança | Motivo |
|---|---|---|---|
| 1.0 | 2026-10-08 | Versão inicial (WTP logística, γ = 0, ação por Tier) | Correção da política (seção 5.6) |

### Desvios do protocolo (fora do simulador)

| Data | Mudança | Motivo |
|---|---|---|
| 2026-10-08 | Bandido passa a usar alvo em log(lucro) | Decidido **depois** da primeira avaliação: em US$ ele ficava no piso da faixa no cenário 0. A política `sl` mantém o alvo em US$ e mostra o efeito dessa escolha. |
| 2026-10-08 | `sl_dataset_combined.csv` passa a incluir as colunas de memória | O SL de LTV (assinatura) usa o mesmo estado do CQL. Os dados gerados não mudam. |

## Como reproduzir

```bash
cd project
python -m pytest tests          # confirma que o simulador é o v1.0
python avaliar_politicas.py     # 5 sementes; CQL incluído se o d3rlpy estiver instalado
python avaliar_generalizacao.py # regiões e cenários fora do treino
```

Saídas em `resultados/`: `tabela_artigo.csv`, `comparacoes.csv`, `por_semente.csv`,
`custo_explicabilidade.csv`, `generalizacao.csv`, `generalizacao_resumo.csv` e `manifesto.json`.
