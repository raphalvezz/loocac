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

## 5b. Extensão para 10 sementes (registrada em 2026-10-14, antes de rodar)

As sementes 47–51 são acrescentadas às 42–46 na comparação principal (todas as
abordagens, CQL com 50.000 passos) e no ambiente sequencial (duas cobranças, CQL
com 50.000 passos). Nada mais muda: mesmos dados por semente, mesmas políticas,
mesmos critérios. As tabelas finais passam a usar as 10 sementes; as de 5 sementes
ficam como estão em `resultados/final_5_sementes/` e no histórico do git, para mostrar
se alguma conclusão (C1–C3, S1–S2) muda com a amostra maior. Se mudar, a conclusão
reportada é a de 10 sementes e a mudança fica registrada aqui.

### Por que 5 (e depois 10) sementes — nota de 2026-10-15

As 5 sementes iniciais foram uma **escolha prática** (tempo de CPU: cada semente da
comparação principal leva ~2,5 h só nos dois CQL), **não** o resultado de uma
análise formal de poder feita antes. A extensão para 10 sementes (seção 5b) também
foi decidida por custo. A análise de poder abaixo é **retrospectiva**: usa o
desvio-padrão entre sementes observado (`analise_estatistica.py`,
`resultados/estatistica/poder.csv`), teste t pareado unilateral, α = 2,5%, poder 80%.

| comparação (total) | dp entre sementes (p.p.) | menor efeito detectável, 5 sementes | idem, 10 sementes | efeito observado (distância ao limiar) |
|---|---|---|---|---|
| C1 venda / assinatura | 1,47 / 1,17 | 2,4 / 1,9 | 1,5 / 1,2 | 13,6 / 10,9 |
| C2 venda / assinatura | 0,13 / 0,10 | 0,22 / 0,16 | 0,13 / 0,10 | 3,2 / 0,92 |
| C3 venda / assinatura | 0,08 / 0,04 | 0,13 / 0,07 | 0,08 / 0,04 | 4,4 / 1,5 |
| S1 venda / assinatura | 1,30 / 0,84 | 2,2 / 1,4 | 1,3 / 0,83 | 8,8 / 13,9 |
| S2 venda / assinatura | 1,42 / 0,52 | 2,4 / 0,87 | 1,4 / 0,52 | 6,1 / 11,6 |

Leitura: todos os efeitos primários ficam muito acima do menor efeito detectável já
com 5 sementes, então a amostra bastou **para estas hipóteses**. Não basta para
diferenças finas: detectar 0,5 p.p. em S1 na venda única exigiria ~55 sementes. A
limitação fica registrada: o número de sementes não foi escolhido por poder.

### Correção para múltiplas comparações (2026-10-15)

`analise_estatistica.py` refaz C1–C3 e S1–S2 como testes t pareados unilaterais
(α = 2,5%, o mesmo que o IC 95% bilateral) e aplica Holm e Bonferroni em duas
famílias: as 10 hipóteses primárias (5 critérios × 2 cobranças, no total) e as 30
com os recortes por tier (`resultados/estatistica/testes.csv`).
- Todas as 10 primárias são rejeitadas com Holm e com Bonferroni (maior p = 1,3×10⁻⁷,
  S2 venda única; limiar de Bonferroni 0,0025).
- Nas 30, 29 são rejeitadas com as duas correções. A única que não passa é a que já
  não passava sem correção: C2 na assinatura, Low Ticket (p = 0,0252). **As
  conclusões não mudam.**

### Margem de −1 p.p. do C2 (2026-10-15)

A margem foi fixada **por bom senso** antes dos resultados, não derivada de um
cálculo. Para dar escala a ela:
- A variação entre sementes de `cql − bandido` é de 0,10–0,13 p.p. (dp), então 1 p.p.
  não é uma margem "de ruído": é uma diferença de importância prática.
- A vantagem do bandido sobre a melhor regra fixa (`meio_faixa`) é de 2,15 p.p. na
  venda única e 1,55 p.p. na assinatura. Pela regra usual de não-inferioridade
  (preservar ao menos 50% do efeito do comparador sobre a referência), a margem seria
  −1,07 p.p. na venda única e −0,77 p.p. na assinatura. A margem de −1 p.p. é, então,
  mais frouxa que essa regra na assinatura.
- Refeito com a margem derivada (−0,77 p.p. na assinatura; −0,76 a −1,31 p.p. por
  tier), C2 dá o mesmo resultado em todos os recortes: passa no total das duas
  cobranças e em todos os tiers, exceto o Low Ticket da assinatura, que já falhava.

## 6. Critérios de sucesso (confirmados em 2026-10-09)

Avaliados no total (Todos) e por tier, separadamente para venda única e assinatura:

1. **C1, aprendizado:** `cql − aleatorio` > 0 (o agente supera a política que gerou os dados).
2. **C2, competitividade:** o IC 95% de `cql − bandido` tem limite inferior acima
   de **−1 p.p.** (não-inferioridade com margem de 1 ponto percentual).
3. **C3, utilidade:** `cql − meio_faixa` > 0 (supera a melhor regra fixa).

Se C2 falhar, o resultado da monografia é que um bandido contextual simples é
suficiente neste problema. É um resultado válido e deve ser reportado como tal.

### Resultado final (2026-10-11, 5 sementes, CQL com 50.000 passos)

Rodada de 5 sementes; a referência para o capítulo 5 passou a ser a de 10 sementes
(abaixo, `resultados/final_10_sementes/`). % do ótimo:

| política | venda Low | venda High | venda Todos | assin. Low | assin. High | assin. Todos |
|---|---|---|---|---|---|---|
| aleatorio | 61,5 | 83,1 | 72,3 | 88,8 | 88,5 | 88,6 |
| meio_faixa | 64,8 | 96,3 | 80,6 | 97,5 | 97,5 | 97,5 |
| sl | 58,6 | 99,0 | 78,8 | 88,7 | 98,9 | 93,8 |
| bandido | 66,4 | 98,9 | 82,6 | 99,0 | 99,1 | 99,0 |
| bc | 69,0 | 96,6 | 82,8 | 97,7 | 97,8 | 97,7 |
| **cql** | **70,6** | **99,2** | **84,9** | 98,2 | **99,7** | 98,9 |
| oraculo_obs | 77,7 | 100 | 88,9 | 100 | 100 | 100 |

- **C1** passa em todos os recortes (`cql − aleatorio` de +9 a +16 p.p.).
- **C2** passa no total dos dois modelos (venda única +2,3 p.p., IC 2,1 a 2,5;
  assinatura −0,1, IC −0,21 a 0,02) e em todos os tiers, exceto o Low Ticket da
  assinatura: −0,84 p.p., IC −1,06 a −0,62, limite inferior 0,06 p.p. abaixo da margem.
- **C3** passa em todos os recortes (`cql − meio_faixa` de +0,7 a +5,9 p.p.).
- Leitura: no problema estático o CQL empata com o bandido (assinatura) ou o supera
  por pouco (venda única, +2,3 p.p., todo no Low Ticket). A vantagem clara do RL
  aparece no ambiente sequencial (seção 9b).

### Resultado com 10 sementes (2026-10-15, sementes 42–51, `resultados/final_10_sementes/`)

| política | venda Low | venda High | venda Todos | assin. Low | assin. High | assin. Todos |
|---|---|---|---|---|---|---|
| aleatorio | 60,7 | 82,0 | 71,3 | 88,3 | 87,8 | 88,0 |
| meio_faixa | 64,8 | 96,3 | 80,6 | 97,5 | 97,5 | 97,5 |
| sl | 58,5 | 98,4 | 78,4 | 88,9 | 98,9 | 93,9 |
| bandido | 66,4 | 99,0 | 82,7 | 99,1 | 99,0 | 99,0 |
| bc | 69,0 | 96,6 | 82,8 | 97,7 | 97,8 | 97,7 |
| **cql** | **70,4** | **99,4** | **84,9** | 98,2 | **99,7** | 98,9 |

- **C1 e C3** continuam passando em todos os recortes.
- **C2** passa no total dos dois modelos (venda única +2,2 p.p., IC 2,1 a 2,3;
  assinatura −0,08, IC −0,15 a −0,01) e em todos os tiers, exceto o Low Ticket da
  assinatura, que segue falhando por uma margem desprezível: −0,88 p.p., limite
  inferior −1,0003 (com 5 sementes era −1,06).
- Com 10 sementes os intervalos ficam mais estreitos e duas diferenças passam a ser
  significativas: na venda única, High Ticket, o CQL supera o bandido (+0,42, IC 0,03
  a 0,81); na assinatura, no total, o bandido supera o CQL por 0,08 p.p. (dentro da
  margem de não-inferioridade). **Nenhuma conclusão muda.**

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

**Extensão do sequencial (registrada em 2026-10-11, antes de rodar):** a rodada
final usa 5 sementes (42–46), CQL com 50.000 passos (o mesmo da comparação
principal) e as duas cobranças. Na assinatura o lucro do período é o LTV da
coorte e a calibração (κ, λ) é a mesma, sem ajuste: o oráculo míope fica em 82,7%
(planejar vale ~17 p.p.). S1 e S2 são avaliados separadamente em cada cobrança,
no total e por tier. A rodada de 3 sementes e 20.000 passos (venda única) fica
como resultado preliminar.

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

### Resultado final do ambiente sequencial (2026-10-11, 5 sementes, CQL com 50.000 passos)

`resultados/sequencial/` (tabela legível: `tabela_legivel.csv`). % do ótimo da
programação dinâmica, média [IC 95%]:

| política | venda Low | venda High | venda Todos | assin. Low | assin. High | assin. Todos |
|---|---|---|---|---|---|---|
| aleatorio | 56,0 | 72,4 | 64,2 | 72,4 | 71,8 | 72,1 |
| meio_faixa | 62,9 | 84,4 | 73,6 | 79,3 | 78,9 | 79,1 |
| sl | 65,3 | 84,7 | 75,0 [72,1; 77,9] | 93,2 | 85,5 | 89,3 [88,7; 90,0] |
| bandido | 61,1 | 89,8 | 75,5 [75,0; 76,0] | 84,6 | 84,3 | 84,4 [83,9; 84,9] |
| bc | 65,0 | 87,2 | 76,1 [73,6; 78,6] | 81,3 | 81,6 | 81,4 [81,0; 81,9] |
| cql_g0 | 64,7 | 91,5 | 78,1 [77,6; 78,6] | 86,9 | 86,6 | 86,7 [86,4; 87,1] |
| **cql_g095** | **72,5** | **96,5** | **84,5** [82,6; 86,3] | **98,5** | **98,2** | **98,4** [98,1; 98,7] |
| oraculo_miope | 90,5 | 89,0 | 89,7 | 82,8 | 82,5 | 82,7 |

Comparações pareadas (p.p., IC 95%):

| | venda única | assinatura |
|---|---|---|
| **S1** `cql_g095 − bandido` | +9,0 (7,5 a 10,5) | +13,9 (13,4 a 14,5) |
| **S2** `cql_g095 − cql_g0` | +6,4 (4,5 a 8,3) | +11,6 (11,0 a 12,2) |
| `cql_g0 − bandido` | +2,6 (1,9 a 3,3) | +2,3 (1,9 a 2,7) |
| `cql_g095 − oraculo_miope` | −5,3 (−7,1 a −3,4) | +15,7 (15,4 a 16,0) |

- **S1 e S2 confirmados nas duas cobranças, no total e em cada tier** (todos os
  limites inferiores acima de zero). O ganho do γ = 0,95 sobre o γ = 0 (6 a 12 p.p.)
  é maior que o do CQL míope sobre o bandido (~2,5 p.p.): a vantagem vem de planejar.
- Na assinatura o CQL com γ = 0,95 chega a 98,4% do ótimo e supera o oráculo míope
  (que conhece o cenário) em 15,7 p.p.; a trajetória de preço quase coincide com a
  da programação dinâmica (`docs/figuras/sequencial_trajetorias.png`). Na venda única
  fica abaixo do oráculo míope no Low Ticket (−18 p.p.) e acima no High Ticket.
- O SL na assinatura (89,3%) supera o bandido e o oráculo míope sem planejar: cobra
  um preço constante ~19% abaixo do míope, o que por acaso constrói reputação.
- Os limites da rodada preliminar continuam valendo (checkpoint escolhido pelo
  simulador; κ e λ são hipóteses). A rodada preliminar está em
  `resultados/sequencial_preliminar/`.

### Resultado com 10 sementes (2026-10-15, sementes 42–51, `resultados/sequencial/`)

A pasta `resultados/sequencial/` passa a ter as 10 sementes (a versão de 5 sementes
está no histórico do git, commit `2e55bdf`). % do ótimo da programação dinâmica:

| política | venda Low | venda High | venda Todos | assin. Low | assin. High | assin. Todos |
|---|---|---|---|---|---|---|
| sl | 63,2 | 83,1 | 73,1 | 93,5 | 85,6 | 89,5 |
| bandido | 60,9 | 90,0 | 75,4 | 84,4 | 84,4 | 84,4 |
| bc | 64,2 | 87,0 | 75,6 | 81,3 | 81,4 | 81,3 |
| cql_g0 | 64,9 | 91,4 | 78,1 | 86,8 | 86,5 | 86,7 |
| **cql_g095** | **71,2** | **97,3** | **84,2** | **98,4** | **98,1** | **98,3** |
| oraculo_miope | 90,5 | 89,0 | 89,7 | 82,8 | 82,5 | 82,7 |

| | venda única | assinatura |
|---|---|---|
| **S1** `cql_g095 − bandido` | +8,8 (7,9 a 9,7) | +13,9 (13,3 a 14,5) |
| **S2** `cql_g095 − cql_g0` | +6,1 (5,1 a 7,1) | +11,6 (11,3 a 12,0) |

S1 e S2 continuam confirmados nas duas cobranças, no total e em cada tier (todos os
limites inferiores acima de zero; o menor é +3,9 p.p., S2 na venda única, Low
Ticket). Os números mudam menos de 0,5 p.p. em relação às 5 sementes. **Nenhuma
conclusão muda.**

### Distância exata ao ótimo da programação dinâmica (10 sementes)

`resultados/estatistica/distancia_oraculo.csv` (p.p. abaixo de 100%, média [IC 95%],
mínimo–máximo entre as sementes):

| CQL γ = 0,95 | Low Ticket | High Ticket | Todos |
|---|---|---|---|
| venda única | 28,8 [26,6; 31,1], 22,6–33,7 | 2,7 [2,0; 3,5], 1,3–4,3 | **15,8** [14,9; 16,7], 13,2–17,7 |
| assinatura | 1,6 [1,3; 1,8], 1,0–2,0 | 1,9 [1,6; 2,1], 1,5–2,6 | **1,7** [1,5; 1,9], 1,3–2,3 |

Para comparação, no total: oráculo míope a 10,3 p.p. (venda) e 17,3 p.p.
(assinatura); bandido a 24,6 e 15,6 p.p. Na venda única quase toda a distância do
CQL está no Low Ticket, onde o estado não identifica o cenário.

### Resultado preliminar do ambiente sequencial (2026-10-10, venda única, 3 sementes, 20.000 passos)

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

### Resultado do deslocamento de público (2026-10-10, 3 sementes)

% do ótimo na origem → no destino; queda pareada em p.p. (IC 95%):

| política | venda única | queda | assinatura | queda |
|---|---|---|---|---|
| meio_faixa | 80,5 → 80,6 | 0,0 | 98,1 → 96,8 | 1,3 |
| sl | 77,8 → 73,3 | 4,5 (3,0 a 6,0) | 94,4 → 92,1 | 2,3 (−1,4 a 6,1) |
| bandido | 82,6 → 70,5 | **12,0** (10,5 a 13,5) | 98,0 → 84,0 | **13,9** (4,0 a 23,9) |
| bc | 83,4 → 82,4 | 1,0 (0,8 a 1,2) | 97,6 → 97,9 | −0,3 |
| cql | 85,3 → 83,1 | 2,2 (0,7 a 3,8) | 98,0 → 98,0 | 0,0 |
| oraculo_obs | 89,1 → 88,6 | 0,5 | 100 → 100 | 0,0 |

- O bandido é o mais frágil: perde 12–14 p.p. quando o público muda e cai abaixo da
  regra fixa `meio_faixa`. O CQL e o BC quase não perdem.
- **Sensibilidade (posterior, 3 sementes, `resultados/bandido_robusto/`):** um
  LightGBM regularizado (`num_leaves` 15, `min_child_samples` 200, L2 = 10) reduz a
  queda do bandido para 5,9 p.p. (venda; IC 5,5 a 6,3) e 6,3 p.p. (assinatura; 1,3 a
  11,3) **sem custo** dentro da distribuição (82,5% e 99,0%, contra 82,6% e 99,0%).
  Uma regularização forte (`num_leaves` 7, `min_child_samples` 500, L2 = 50) elimina
  a queda (−1,6 e −1,5 p.p.), mas custa 4 p.p. dentro da distribuição (78,3% e
  95,2%). O CQL perde 2,2 e 0,0 p.p. com 84,9% e 98,9% dentro da distribuição: a
  robustez dele não depende de ajuste. A fragilidade é, portanto, em boa parte da
  configuração do LightGBM, não do bandido como método.
- Hipótese (não testada): o CQL e o BC ficam presos à faixa de preço observada nos
  dados (o termo conservador e a imitação puxam para a coleta), enquanto o bandido
  extrapola a curva de lucro aprendida com a região como variável, e a região do
  destino é uma categoria nunca vista.

## 9d. Seleção de checkpoint sem o simulador: FQE contra WIS (2026-10-15)

`avaliar_selecao_ope.py`: um treino de 50.000 passos do CQL por semente e cobrança
(3 sementes), com checkpoints a cada 5.000 passos (10 por treino). Cada checkpoint
recebe três notas: o % do ótimo verdadeiro (simulador), o estimado pelo FQE (seção
9b) e o estimado por **importance sampling ponderado (WIS)** com núcleo gaussiano
(h = 0,1 na escala da ação) e as propensões conhecidas da coleta (preço uniforme na
faixa do cenário). Os dois estimadores usam só os 20% de dados que o agente não viu.

| | Spearman com o verdadeiro | arrependimento (p.p.) | escolhe o mesmo checkpoint |
|---|---|---|---|
| FQE, venda única | −0,49 a −0,27 | 1,0 a 1,6 | — |
| WIS, venda única | 0,68 a 0,83 | 0,0 a 0,17 | — |
| FQE, assinatura | 0,22 a 0,26 | 0,22 a 0,30 | — |
| WIS, assinatura | 0,90 a 0,98 | 0,01 a 0,13 | — |
| FQE × WIS | −0,37 a 0,42 | — | 0 de 6 |

Arrependimento = % do ótimo do melhor checkpoint verdadeiro − % do checkpoint
escolhido pelo estimador (`resultados/selecao_ope/resumo.csv`).

- **O WIS ordena os checkpoints quase como o simulador** e escolhe um que fica a
  no máximo 0,17 p.p. do melhor. Superestima o nível em ~2–3 p.p. na venda única
  (ESS de ~15 linhas por contexto), mas o nível não importa para escolher.
- **O FQE não serve para escolher checkpoint aqui:** na venda única a ordem sai
  invertida (Spearman negativo) e ele escolhe sempre o último checkpoint, que é dos
  piores (arrependimento de até 1,6 p.p.); na assinatura a ordem é quase aleatória.
  Os dois estimadores nunca escolheram o mesmo checkpoint.
- Por que o WIS vai bem: a política de coleta é conhecida e cobre toda a faixa
  (preço uniforme), exatamente a condição em que importance sampling funciona. O FQE
  depende de um modelo Q aprendido em ~10 mil linhas, cujo erro é maior que as
  diferenças (décimos de p.p.) entre checkpoints. Com dados reais de uma coleta
  menos aleatória, o WIS perderia essa vantagem (pesos maiores, ESS menor).
- Consequência: um sistema sem simulador poderia escolher o checkpoint do CQL pelo
  WIS sem perda relevante; o resultado da seção 9b (FQE acerta o nível agregado de
  um modelo pronto) não se estende à comparação fina entre checkpoints.

## 9c. Risco (VaR/CVaR 5%) de cada método (2026-10-14)

SL e bandido ganharam regressão quantílica (`treinar_quantis.py`: LightGBM com
`objective='quantile'` nos quantis 1%, 2,5%, 5%, 10%, 50% e 90%, mesmas features e
escala do alvo). Assim os três métodos com modelo de lucro têm um VaR/CVaR 5%
próprio: crítico quantílico (CQL) ou regressão quantílica (SL, bandido); o BC não
tem. A API devolve esse risco em `var_5_metodo`/`cvar_5_metodo`/`fonte_risco_metodo`,
além da avaliação comum pelo crítico (`var_5_percent`/`cvar_5_percent`).

`avaliar_risco.py` compara cada estimativa com o valor exato do simulador (o ruído é
lognormal conhecido: VaR exato = E(p)·exp(−σ²/2 + σ·z₀,₀₅)), no preço recomendado e
numa grade de 5 preços por estado. Erro em % do lucro ótimo do estado; cobertura =
probabilidade real de o lucro ficar abaixo do VaR estimado (ideal 5%).
Modelos da verificação em máquina limpa (`resultados/risco.csv`):

| | erro abs. VaR (grade) | cobertura (grade) | erro abs. VaR (preço recomendado) | cobertura (preço rec.) |
|---|---|---|---|---|
| CQL venda única | 175% | 9,6% | 105% | 6,6% |
| SL venda única | 5,4% | 6,8% | 25,5% | 20,1% |
| Bandido venda única | 5,1% | 6,7% | 27,7% | 19,0% |
| CQL assinatura | 73% | 48,8% | 83% | 55,1% |
| SL assinatura | 2,6% | 6,9% | 2,1% | 5,2% |
| Bandido assinatura | 3,0% | 6,9% | 2,7% | 4,0% |

- A regressão quantílica é bem calibrada: erro de 2–6% do lucro ótimo e cobertura
  de 5–7% na grade; no conjunto de teste os quantis 1%, 5%, 50% e 90% cobrem 1,1%,
  5,0–5,4%, 50% e 90–92%.
- O crítico do CQL **não** serve como medida de risco: na venda única subestima o VaR
  (fica negativo no Low Ticket) e na assinatura o superestima (o lucro fica abaixo
  do "VaR 5%" em ~50% dos casos). O termo conservador do CQL empurra os valores Q
  para baixo fora dos dados e a escala da recompensa distorce a cauda; o crítico
  serve para escolher o preço, não para medir a distribuição do lucro.
- No preço recomendado da venda única, Low Ticket, SL e bandido também erram (~45%):
  ali o estado não identifica o cenário, e o quantil aprendido é o da mistura dos
  cenários que dividem a chave (limitação de observabilidade, seção 11). No High
  Ticket o erro é de 5–6%.
- Consequência para a API: a coluna "VaR 5% (RL)" da avaliação comum herda esses
  erros; o risco próprio do SL e do bandido é o mais confiável.

## 10. Custo e explicabilidade

`resultados/custo_explicabilidade.csv`: tempo de treino (média por semente),
latência por recomendação (ms, CPU, média sobre os 144 estados), necessidade de
GPU e grau de explicabilidade.

## 11. Decisões que estavam em aberto antes da primeira rodada com CQL

1. ~~**Observabilidade na venda única.**~~ **Decidido em 2026-10-11: manter o
   simulador v1.0 e declarar como limitação** (`docs/limitacoes.md`). Incluir um
   preço de referência no estado mudaria o simulador, a API e a tela depois de
   todas as rodadas; o teto observável (`oraculo_obs`) já é reportado junto.
   Texto original: O estado não distingue os cenários 3/4 e
   5/6, então o teto (`oraculo_obs`) no Low Ticket é ~78% do ótimo. Opções: incluir
   um preço de referência do produto no estado (muda a API e a tela) ou manter e
   discutir como limitação.
2. ~~**Pouca folga quando o cenário é identificável.**~~ **Decidido em 2026-10-11:
   manter o simulador v1.0 e declarar como limitação.** Ampliar a variação de
   contexto agora invalidaria as rodadas finais; a pouca folga é reportada e é um
   dos motivos de o problema estático não separar bem os métodos (seção 6).
   Texto original: Com a calibração atual,
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
| 2026-10-11 | Seção 11, itens 1 e 2 (preço de referência no estado; folga do simulador): manter o simulador v1.0 e declarar como limitação | Mudar o estado ou a calibração depois das rodadas finais invalidaria a comparação; os dois pontos entram em `docs/limitacoes.md`. |
| 2026-10-11 | Análise de sensibilidade **posterior** do bandido regularizado no deslocamento de público (`avaliar_bandido_robusto.py`) | Feita depois de ver a queda de 12–14 p.p. É exploratória: o bandido da comparação principal continua o registrado (`bandido.BandidoLGBM`). |
| 2026-10-11 | Tela mostra "indisponível" (não "—" nem número) quando a API devolve `null` em lucro do método ou VaR/CVaR | O risco vem do crítico do RL para todos os métodos; sem o RL carregado, não há valor a mostrar. Conferido no navegador (`docs/figuras/tela_sem_rl.png`). |
| 2026-10-14 | Regressão quantílica no SL e no bandido; API com risco próprio por método; rotas legadas `/recommend_price` e `/recommend_subscription_price` removidas | Comparação simétrica de risco (seção 9c). Não muda as decisões de preço nem a comparação principal. |
| 2026-10-08 | `oraculo_obs` passa a maximizar o % do ótimo, não o lucro médio em US$ | Erro de definição: maximizando US$, o cenário de lucro maior dominava e uma política observável (ε-greedy) passou do "teto". Na venda única o teto sobe de 83,6% para 88,9% (Low: 67% → 78%). |

## Como reproduzir

Tudo na pasta `project/`, depois de `pip install -r ../requirements.txt`. Os
tempos são de uma máquina de 4 núcleos sem GPU; os scripts com CQL dominam o
tempo. Os resultados de referência já estão versionados em `resultados/`.

```bash
cd project
python -m pytest tests                   # simulador v1.0, sequencial e API (~1 min com 4 threads)

# Modelos usados pela API (ou baixe a release: docs/artefatos.md)
python train_pipeline.py                 # ~4 h 30 (dois CQL de 50 mil passos)

# Comparação principal (seção 6)
python avaliar_politicas.py --saida resultados/final_10_sementes  # 10 sementes, ~24 h em série

# Testes de ambiente (seção 9b)
python avaliar_sequencial.py             # 10 sementes x 2 cobranças, ~40 h em série
python figura_sequencial.py
python avaliar_robustez.py               # demanda linear, 3 sementes, ~4 h 30
python avaliar_deslocamento.py           # público novo, 3 sementes, ~4 h 30
python avaliar_bandido_robusto.py        # sensibilidade do bandido, ~20 min
python avaliar_fqe.py                    # FQE; precisa dos modelos do train_pipeline.py, ~15 min
python avaliar_risco.py                  # calibração do VaR/CVaR; idem (com treinar_quantis.py), ~5 min
python avaliar_selecao_ope.py --sementes 3   # FQE x WIS na escolha do checkpoint, ~6 h

# Análises complementares
python avaliar_generalizacao.py           # regiões e cenários fora do treino, 1 semente, ~2 h
# Bandido em operação: lento (refaz o bandido e os 5 modelos do Thompson a cada lote);
# com 1 thread passou de 9 h sem terminar a 1ª semente de 5. Use várias threads.
for n in 10 100; do   # 120 e 1.200 linhas iniciais
  python simular_bandido_online.py --n-inicial $n --saida resultados/bandido_online_$((n*12))
  python figura_bandido_online.py --entrada resultados/bandido_online_$((n*12))/bandido_online_curva.csv \
      --saida docs/figuras/bandido_online_$((n*12))_iniciais.png
done
python ablacoes_cql.py && python diagnostico_cql.py
```

O sequencial roda em paralelo, uma semente e cobrança por processo, e depois junta:

```bash
for m in venda_unica assinatura; do for s in $(seq 42 51); do
  OMP_NUM_THREADS=1 python avaliar_sequencial.py --modelos $m --semente-inicial $s --sementes 1 \
      --saida tmp/seq/${m}_s$s &
done; done; wait
python avaliar_sequencial.py --agregar tmp/seq/* --saida resultados/sequencial
```

| Script | Saída em `resultados/` | Seção |
|---|---|---|
| `avaliar_politicas.py` | `final_10_sementes/{tabela_artigo,comparacoes,por_semente,custo_explicabilidade}.csv`, `manifesto.json` | 6 |
| `avaliar_sequencial.py` | `sequencial/{tabela,tabela_legivel,comparacoes,por_semente,trajetorias}.csv`, `manifesto.json` | 9b |
| `figura_sequencial.py` | `docs/figuras/sequencial_trajetorias.png` | 9b |
| `avaliar_robustez.py` | `robustez_linear/` | 9b |
| `avaliar_deslocamento.py` | `deslocamento/{tabela,por_semente}.csv` | 9b |
| `avaliar_bandido_robusto.py` | `bandido_robusto/{tabela,por_semente}.csv` | 9b |
| `avaliar_fqe.py` | `fqe.csv`, `fqe_por_estado.csv` | 9b |
| `treinar_quantis.py` + `avaliar_risco.py` | `risco.csv`, `risco_por_estado.csv` | 9c |
| `avaliar_selecao_ope.py` | `selecao_ope/{checkpoints,resumo}.csv` | 9d |
| `avaliar_generalizacao.py` | `generalizacao.csv`, `generalizacao_resumo.csv` | 7 |
| `simular_bandido_online.py` + `figura_bandido_online.py` | `bandido_online_{120,1200}/`, `docs/figuras/bandido_online_*_iniciais.png` | 8 |
| `ablacoes_cql.py`, `diagnostico_cql.py` | `ablacoes_cql*.csv`, `diagnostico_cql.csv` | 9 |
| `verificar_api.py` (API no ar) | `verificacao_api.json` | `docs/verificacao.md` |
