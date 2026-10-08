# Protocolo de avaliação das políticas de preço

Este documento fixa **antes de rodar** o simulador, a métrica, as comparações e os
critérios de sucesso do capítulo 5. Resultados que contrariem as hipóteses também
são reportados.

> **Status: rascunho para revisão.** Os critérios da seção 6 e as decisões da
> seção 8 precisam ser confirmados antes da primeira rodada com CQL. Depois disso,
> qualquer mudança entra no histórico (seção 9) com justificativa.

## 1. Simulador

- Arquivo: `simulador.py`, `VERSAO = "1.0"`.
- Congelado pelos testes `tests/test_simulador.py` (valores de referência da
  economia e da geração de dados). Se um teste falhar, o simulador mudou.
- Só se altera para corrigir um **erro**. Procedimento: subir `VERSAO`, atualizar
  os valores de referência dos testes, registrar o motivo na seção 9 e refazer
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
| `bandido` (LightGBM + grade) | sim | estado |
| `cql` | sim | estado |
| `oraculo_obs` | não | estado + função de lucro do simulador → **teto real** |
| `oraculo` | não | cenário verdadeiro → **referência dos percentuais** |

"Estado" é o mesmo para todas as políticas não privilegiadas e o avaliador remove a
coluna do cenário antes de consultá-las. A faixa observável de cada estado vem do
catálogo por chave de estado (`politicas.chave_estado`):
- venda única: (Tier, Orçamento). Os cenários 3/4 e 5/6 coincidem;
- assinatura: (Tier, Orçamento, `avg_price_offered_segment_90d`), que separa 3/4 e 5/6.

Hiperparâmetros fixados a priori, sem ajuste na grade de avaliação:
- bandido: `LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=63)`,
  alvo log(lucro), grade de 200 preços (geométrica) na faixa observável;
- CQL: os mesmos dos notebooks (`gamma=0`, `conservative_weight=5`, 64 quantis,
  `batch_size=256`, até 50.000 passos, paciência de 20 épocas de 1.000 passos).

## 5. Sementes e intervalos

- Sementes **42 a 46** (5). Cada semente gera um dataset novo e retreina as
  políticas treináveis.
- Intervalo: IC 95% com t de Student sobre as 5 sementes.
- Comparações **pareadas** por semente (mesmo dataset): `cql − aleatorio`,
  `cql − meio_faixa`, `cql − bandido`, `bandido − aleatorio`, `bandido − meio_faixa`.
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

## 7. Custo e explicabilidade

`resultados/custo_explicabilidade.csv`: tempo de treino (média por semente),
latência por recomendação (ms, CPU, média sobre os 144 estados), necessidade de
GPU e grau de explicabilidade.

## 8. Decisões em aberto antes da primeira rodada com CQL

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

## 9. Histórico

| Versão | Data | Mudança | Motivo |
|---|---|---|---|
| 1.0 | 2026-10-08 | Versão inicial (WTP logística, γ = 0, ação por Tier) | Correção da política (seção 5.6) |

## Como reproduzir

```bash
cd project
python -m pytest tests          # confirma que o simulador é o v1.0
python avaliar_politicas.py     # 5 sementes; CQL incluído se o d3rlpy estiver instalado
```

Saídas em `resultados/`: `tabela_artigo.csv`, `comparacoes.csv`, `por_semente.csv`,
`custo_explicabilidade.csv` e `manifesto.json`.
