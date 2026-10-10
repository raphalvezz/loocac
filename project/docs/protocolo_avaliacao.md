# Protocolo de avaliação das políticas de preço

Este documento fixa **antes de rodar** o simulador, a métrica, as comparações e os
critérios de sucesso do capítulo 5. Resultados que contrariem as hipóteses também
são reportados.

> **Status: em vigor desde 2026-10-09.** Critérios da seção 6 confirmados; escala
> da ação do CQL decidida (seção 11, item 3). Os itens 1 e 2 da seção 11 seguem em
> aberto e, se mudarem algo, entram no histórico (seção 12) com justificativa.

## 1. Simulador

- Arquivo: `simulador.py`, `VERSAO = "1.0"`.
- Congelado pelos testes `tests/test_simulador.py` (valores de referência da
  economia e da geração de dados). Se um teste falhar, o simulador mudou.
- Só se altera para corrigir um **erro**. Procedimento: subir `VERSAO`, atualizar
  os valores de referência dos testes, registrar o motivo na seção 12 e refazer
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
| `oraculo_obs` | não | estado + função de lucro do simulador → **teto real** da métrica (maximiza o % do ótimo esperado entre os cenários que o estado não distingue) |
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
- CQL (versão canônica `politicas.CQL` / `treinar_cql.py`): `gamma=0`,
  `conservative_weight=5`, 64 quantis, `batch_size=256`, ação na faixa observável
  do estado (`escala_acao='estado'`), até 50.000 passos, paciência de 20 épocas de
  1.000 passos, checkpoint pelo lucro simulado;
- BC: `politicas.BC` (d3rlpy), mesma escala de ação do CQL, 5.000 passos.

Os notebooks de RL documentam a versão anterior (ação por Tier) e não entram nos
resultados.

## 5. Sementes e intervalos

- Sementes **42 a 46** (5). Cada semente gera um dataset novo e retreina as
  políticas treináveis.
- Intervalo: IC 95% com t de Student sobre as 5 sementes.
- Comparações **pareadas** por semente (mesmo dataset): `cql − aleatorio`,
  `cql − meio_faixa`, `cql − sl`, `cql − bandido`, `sl − aleatorio`, `sl − meio_faixa`,
  `bandido − aleatorio`, `bandido − meio_faixa`, `bandido − sl`.
  Conclusão: "A > B" se o IC 95% da diferença estiver todo acima de 0; "A < B" se
  todo abaixo; senão, "inconclusivo".

## 6. Critérios de sucesso (confirmados em 2026-10-09)

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

## 8. Bandido em operação (simulação online)

`simular_bandido_online.py` (módulo `bandido.py`). O produto começa com poucos dados
logados pela política aleatória (`--n-inicial` por cenário: 10 → 120 linhas, ou
100 → 1.200 linhas) e passa a precificar sozinho:
- 20 lotes de 600 decisões; estados sorteados da grade de avaliação, o MESMO fluxo
  para todas as políticas;
- o simulador devolve o lucro com ruído; as variantes em operação reajustam o
  modelo a cada lote (como um reajuste diário);
- métrica por decisão: % do ótimo (lucro esperado), como na seção 3.

Variantes (hiperparâmetros fixados a priori): ε-greedy com ε = 0,2·0,85^lote
(mínimo 0,02); Thompson por bootstrap com K = 5 modelos. Ambas usam o modelo do
bandido offline. Referências no mesmo fluxo: bandido offline treinado só nos dados
iniciais (congelado), bandido offline com o dataset completo (48 mil linhas),
aleatório e `oraculo_obs`.

Ganho de aprender em operação = variante − offline com os mesmos dados iniciais
(pareado por semente, IC 95%), nos 5 últimos lotes e na média em operação.

## 9. Ablações do CQL e diagnóstico do Low Ticket (seção 5.6)

`ablacoes_cql.py`: treino curto (5.000 passos, seleção pelo lucro simulado),
3 sementes, uma alteração por vez em relação à configuração dos notebooks:
peso conservador (0,5 / 20), ação sem escala por Tier (faixa global), γ = 0,98
com transições independentes, recompensa em log e por estado, e ação na faixa
observável do estado (com e sem recompensa por estado).

`diagnostico_cql.py`: compara, por tier, o preço do ator, o argmax do crítico
e o oráculo observável, e desenha o crítico contra o lucro verdadeiro num
estado Low Ticket.

Resultados em `resultados/ablacoes_cql*.csv`, `resultados/diagnostico_cql.csv`
e `docs/figuras/diagnostico_cql_low_ticket.png`. São de treino curto: servem
para comparar configurações entre si, não como resultado final do CQL.

## 9b. Testes de ambiente (registrados em 2026-10-10, antes dos resultados)

**Ambiente sequencial** (`sequencial.py`, `avaliar_sequencial.py`): campanhas de 12
períodos; o preço muda a reputação ρ (κ = 0,15; preço justo = 0,85·p\*) e a demanda
futura. Ótimo exato por programação dinâmica; métrica = lucro total da campanha em %
desse ótimo; oráculo míope (~90%) mostra quanto vale planejar. Políticas: aleatório,
meio da faixa, SL, bandido, BC, CQL γ = 0 e CQL γ = 0,95 (recompensa por contexto,
20.000 passos, checkpoint pelo lucro total simulado), 3 sementes.
- **S1 (o RL se justifica):** `cql_g095 − bandido` > 0 no total.
- **S2 (o ganho vem de planejar):** `cql_g095 − cql_g0` > 0.
Se S1 falhar, a conclusão do TCC é que, mesmo com efeito intertemporal, o RL
offline não superou o bandido nestes dados.

**Robustez à forma da demanda** (`avaliar_robustez.py`): a mesma comparação da
seção 5 com demanda linear (simulador `usar_forma('linear')`), 3 sementes, CQL com
20.000 passos. A conclusão é robusta se a ordem bandido/BC/CQL e os critérios C1–C3
se repetirem.

**Deslocamento de público** (`avaliar_deslocamento.py`): treino só com North
America/Europe, avaliação em Asia/South America (mais sensíveis a preço) e na
origem; checkpoint do CQL escolhido só na origem. Mede a queda origem − destino.

**FQE** (`avaliar_fqe.py`): estimativa fora da política (d3rlpy FQE, γ = 0) do
lucro do CQL e do BC usando só os 20% de dados não usados no treino, comparada com
o lucro verdadeiro do simulador.

### Resultado do ambiente sequencial (2026-10-10, 3 sementes)

| política | Low Ticket | High Ticket | Todos |
|---|---|---|---|
| aleatorio | 56,3 | 72,0 | 64,2 |
| meio_faixa | 62,9 | 84,4 | 73,6 |
| sl | 65,3 | 84,1 | 74,7 |
| bandido | 61,1 | 89,9 | 75,5 |
| bc | 66,8 | 86,8 | 76,8 |
| cql_g0 | 64,7 | 91,2 | 77,9 |
| **cql_g095** | **70,7** | **96,9** | **83,8** |
| oraculo_miope | 90,5 | 89,0 | 89,7 |

- **S1 confirmado:** `cql_g095 − bandido` = +8,3 p.p. (IC 95% 7,9 a 8,7), positivo nos dois tiers.
- **S2 confirmado:** `cql_g095 − cql_g0` = +5,9 p.p. (5,1 a 6,7). Com γ = 0 o CQL fica
  junto do bandido (+2,4 p.p.); o ganho vem de planejar.
- A trajetória de preço (`docs/figuras/sequencial_trajetorias.png`) mostra o motivo: o
  CQL com γ = 0,95 começa abaixo do preço míope e sobe ao longo da campanha (como a
  programação dinâmica, de forma menos intensa); bandido, SL, BC e CQL γ = 0 cobram
  um preço quase constante.
- Limites: no Low Ticket ainda fica 20 p.p. abaixo do oráculo míope (que conhece o
  cenário); o checkpoint é escolhido com o simulador (mesma regra da seção 4), algo
  que não existe em produção; o efeito intertemporal (κ, λ) é uma hipótese do
  ambiente, não um dado.

### Resultado da robustez à demanda linear (2026-10-10, 3 sementes)

% do ótimo no total (Low / High entre parênteses):

| política | venda única | assinatura |
|---|---|---|
| sl | 80,6 (62,4 / 98,9) | 97,1 |
| bandido | 82,6 (66,1 / 99,1) | 99,2 |
| bc | 87,2 (76,0 / 98,3) | 99,3 |
| cql | 89,0 (79,5 / 98,6) | 99,4 |
| oraculo_obs | 91,6 (83,2 / 100) | 100 |

- A ordem cql > bc > bandido > sl na venda única se repete; a vantagem do CQL está
  toda no Low Ticket (`cql − bandido` = +13,4 p.p.; no High Ticket −0,6, inconclusivo).
- Critérios no total: C1, C2 e C3 passam nos dois modelos.
- Por tier falham por pouco: C2 no High Ticket da venda única (limite inferior
  −1,3 p.p.), C3 no High Ticket da venda única (inconclusivo, +0,4) e no Low Ticket
  da assinatura (−0,09 p.p.). Na assinatura tudo fica entre 97% e 99,4%: o problema
  continua fácil para qualquer método razoável.

## 10. Custo e explicabilidade

`resultados/custo_explicabilidade.csv`: tempo de treino (média por semente),
latência por recomendação (ms, CPU, média sobre os 144 estados), necessidade de
GPU e grau de explicabilidade.

## 11. Decisões em aberto antes da primeira rodada com CQL

1. **Observabilidade na venda única.** O estado não distingue os cenários 3/4 e
   5/6, então o teto (`oraculo_obs`) no Low Ticket é ~78% do ótimo. Opções: incluir
   um preço de referência do produto no estado (muda a API e a tela) ou manter e
   discutir como limitação.
2. **Pouca folga quando o cenário é identificável.** Com a calibração atual,
   `meio_faixa` já obtém ~96% (venda única, High) e ~97,5% (assinatura) do ótimo.
   A variação de contexto (multiplicadores 0,81–1,16) desloca pouco o preço ótimo,
   então a diferença máxima entre qualquer política aprendida e a regra fixa é de
   ~3 p.p. Opções: ampliar a variação de contexto (simulador v1.1) ou manter e
   reportar que o problema tem pouca folga.
3. ~~**Escala da ação do CQL.**~~ **Decidido em 2026-10-09:** `escala_acao='estado'`
   é o padrão no treino, na API e na avaliação; `politicas.CQL`/`treinar_cql.py`
   são a versão canônica e os notebooks só documentam. Registrado como desvio
   (seção 12).

## 12. Histórico

| Versão | Data | Mudança | Motivo |
|---|---|---|---|
| 1.0 | 2026-10-08 | Versão inicial (WTP logística, γ = 0, ação por Tier) | Correção da política (seção 5.6) |

### Desvios do protocolo (fora do simulador)

| Data | Mudança | Motivo |
|---|---|---|
| 2026-10-08 | Bandido passa a usar alvo em log(lucro) | Decidido **depois** da primeira avaliação: em US$ ele ficava no piso da faixa no cenário 0. A política `sl` mantém o alvo em US$ e mostra o efeito dessa escolha. |
| 2026-10-08 | `sl_dataset_combined.csv` passa a incluir as colunas de memória | O SL de LTV (assinatura) usa o mesmo estado do CQL. Os dados gerados não mudam. |
| 2026-10-09 | CQL e BC passam a usar a ação na faixa observável do estado (`escala_acao='estado'`); `politicas.CQL`/`treinar_cql.py` viram a versão canônica | Decidido **depois** das ablações (seção 9): com a ação na faixa do Tier, o crítico extrapola fora da faixa de dados do estado e o ator fica entre duas modas (Low Ticket ~50% do ótimo); com a faixa do estado, ~85% (venda única) e ~98% (assinatura) em treino curto. É a mesma faixa em que `sl` e `bandido` já buscavam o preço, então a comparação fica mais justa. As ablações publicadas usam a versão anterior (`tier`) na base. |
| 2026-10-10 | FQE aprende a recompensa relativa à média da chave de estado (lucro / média − 1), não o lucro padronizado globalmente | Decidido **depois** da primeira rodada do FQE: com a padronização global a ordem entre estados saía certa (correlação ~1), mas o % do ótimo errava em centenas de p.p. nos estados de lucro pequeno. Com a escala por estado: CQL venda única 85,0% (simulador) contra 85,1% (FQE); BC erra +5,9 p.p.; assinatura ±1,2 p.p. |
| 2026-10-08 | `oraculo_obs` passa a maximizar o % do ótimo, não o lucro médio em US$ | Erro de definição: maximizando US$, o cenário de lucro maior dominava e uma política observável (ε-greedy) passou do "teto". Na venda única o teto sobe de 83,6% para 88,9% (Low: 67% → 78%). |

## Como reproduzir

```bash
cd project
python -m pytest tests          # confirma que o simulador é o v1.0
python avaliar_politicas.py     # 5 sementes; CQL incluído se o d3rlpy estiver instalado
python avaliar_generalizacao.py # regiões e cenários fora do treino
python simular_bandido_online.py --n-inicial 10   # bandido aprendendo em operação
python figura_bandido_online.py
```

Saídas em `resultados/`: `tabela_artigo.csv`, `comparacoes.csv`, `por_semente.csv`,
`custo_explicabilidade.csv`, `generalizacao.csv`, `generalizacao_resumo.csv` e `manifesto.json`.
