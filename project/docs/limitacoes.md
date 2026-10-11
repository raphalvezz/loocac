# Limitações (texto-base para a monografia)

Cada item traz o fato, o número de referência e onde ele está no repositório.
Os textos entre aspas podem ser adaptados diretamente.

## 1. Observabilidade do estado (venda única)

O estado da venda única (Tier, orçamento, região, plataforma) não distingue os
cenários 3/4 e 5/6, que têm faixas de preço diferentes. Nenhuma política que vê só
o estado passa do teto `oraculo_obs`: 88,9% do ótimo no total e 77,7% no Low Ticket
(`resultados/final_5_sementes/`). Incluir um preço de referência do produto no
estado resolveria, mas mudaria o simulador, a API e a tela depois das rodadas
finais (decisão em `protocolo_avaliacao.md`, seção 11).

> Parte da distância até o ótimo não é do método, e sim da informação disponível:
> com o estado usado, o melhor resultado possível é 88,9% do ótimo na venda única.

## 2. Pouca folga no problema estático

Quando o cenário é identificável, a regra fixa `meio_faixa` já obtém ~96% (venda
única, High Ticket) e 97,5% (assinatura). A diferença máxima entre um método
aprendido e a regra fixa fica em poucos pontos percentuais, o que limita o quanto o
problema de passo único consegue separar os métodos.

> No problema de passo único, os métodos aprendidos ganham da melhor regra fixa por
> 1,5 a 4,4 p.p. no total (CQL); as diferenças entre eles são da mesma ordem.

## 3. O efeito intertemporal é uma hipótese de simulação

A vantagem clara do RL (CQL γ = 0,95 sobre o bandido: +9,0 p.p. na venda única e
+13,9 p.p. na assinatura) aparece só no ambiente sequencial, em que o preço muda a
reputação e a demanda futura. A intensidade desse efeito (κ = 0,15, λ = 0,85) foi
calibrada, não medida. O produto serve a decisão de passo único
(`docs/escopo_produto.md`).

## 4. Seleção de checkpoint com o simulador

O checkpoint do CQL (e do BC) é escolhido pelo lucro no simulador, informação que
um sistema real não teria. A avaliação fora da política (FQE, `resultados/fqe.csv`)
acerta o % do ótimo no agregado (85,1% estimado contra 85,0% no simulador para o CQL
da venda única), mas erra em média 26% por estado: serve como verificação do total,
não como substituto do simulador.

## 5. Generalização para público novo

Treinado só com North America e Europe, o bandido perde 12,0 p.p. (venda única) e
13,9 p.p. (assinatura) em Asia e South America; o CQL perde 2,2 e 0,0
(`resultados/deslocamento/`). Uma análise posterior mostra que boa parte disso é da
configuração do LightGBM: com regularização moderada a queda cai para ~6 p.p. sem
custo dentro da distribuição (`resultados/bandido_robusto/`).

> O bandido contextual, na configuração usada, foi o método mais sensível a uma
> mudança de público; uma regularização moderada reduz essa perda pela metade sem
> perder desempenho no público original. O CQL se manteve estável sem ajuste.

## 6. Risco (VaR/CVaR) depende do crítico do RL

O VaR e o CVaR de 5% exibidos para todos os métodos vêm dos quantis do crítico do
CQL, avaliados no preço de cada método. É um avaliador comum, não uma medida
própria de cada método, e herda os erros do crítico. Sem o modelo de RL carregado,
a API devolve `null` e a tela mostra "indisponível" (`docs/figuras/tela_sem_rl.png`).
O BC não tem modelo de lucro próprio (também "indisponível").

## 7. Segurança

Chave de API única e visível no navegador, sem autenticação por usuário, sem
limitação de taxa e com o retreino no mesmo servidor. Texto pronto em
`docs/seguranca.md` (seção "Texto para a seção de Limitações").

## 8. Dados simulados

Todos os resultados vêm de dados gerados pelo simulador v1.0 (demanda logística de
disposição a pagar; a demanda linear foi usada como teste de robustez e manteve a
ordem dos métodos). Nenhum resultado foi validado com campanhas reais.
