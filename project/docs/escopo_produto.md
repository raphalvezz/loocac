# Escopo do produto: decisão de passo único

**Decisão (2026-10-11):** a API e a tela servem a decisão de **passo único**: dado
o estado da campanha, recomendam um preço. O ambiente sequencial
(`sequencial.py`) **não** está integrado ao produto. Ele é a evidência
experimental de quando vale migrar para RL que planeja.

## Por que não integrar o agente sequencial agora

1. **O estado sequencial não existe no produto.** O agente sequencial precisa da
   reputação ρ da marca naquele público e do período da campanha. O sistema não
   mede reputação. Para servir esse agente seria preciso definir um proxy
   observável (por exemplo, conversão recente contra a esperada) e validá-lo com
   dados reais, o que está fora do escopo do TCC.
2. **A dinâmica é uma hipótese.** κ e λ foram calibrados para que planejar valha
   cerca de 10 p.p. (venda única) e 17 p.p. (assinatura). Isso mostra **quando** o
   RL se justifica, não que o mercado real tenha esse efeito.
3. **No problema de passo único o CQL não tem vantagem decisiva.** Na comparação
   principal ele empata com o bandido contextual (assinatura) ou o supera por
   pouco (venda única, +2,3 p.p.). Servir o CQL como método padrão (`metodo=rl`)
   é coerente com esses resultados, mas a API também oferece `sl`, `bandido` e
   `bc` para comparação lado a lado.

## Texto para a monografia (Limitações / Trabalhos futuros)

> O protótipo serve a decisão de preço de passo único: para cada estado da
> campanha, a API recomenda um preço com o método escolhido (CQL, bandido
> contextual, SL ou BC). Nesse problema, o CQL ficou empatado com o bandido
> contextual na assinatura e 2,3 p.p. acima dele na venda única (5 sementes).
> A vantagem clara do RL apareceu apenas no ambiente sequencial, em que o preço
> de hoje altera a reputação e a demanda futura: com γ = 0,95, o CQL superou o
> bandido em 9,0 p.p. (venda única) e 13,9 p.p. (assinatura) e a sua própria versão
> míope (γ = 0) em 6,4 e 11,6 p.p. (5 sementes, IC 95% acima de zero; seção 5.x). Esse ambiente não foi
> integrado ao produto porque o sistema não observa a reputação da marca e
> porque a intensidade do efeito intertemporal (κ, λ) é uma hipótese de
> simulação. O resultado deve ser lido como um critério de decisão: o RL offline
> passa a se justificar quando houver evidência, nos dados reais, de que o preço
> praticado afeta a demanda dos períodos seguintes e quando houver uma medida
> observável desse efeito. Até lá, um bandido contextual entrega resultado
> equivalente com menor custo de treino (segundos contra cerca de 40 minutos em
> CPU) e maior explicabilidade.

Os números de referência estão em `docs/protocolo_avaliacao.md` (seção 6 e 9b) e
em `resultados/final_5_sementes/` e `resultados/sequencial/`.

## O que mudaria para a opção (b)

Uma rota `POST /recomendar_campanha` receberia também `reputacao` (ou o proxy) e
`periodo`, e usaria um `pol.CQL` treinado com `extras=sequencial.EXTRAS` sobre
dados logados de campanhas em vários períodos. O código de treino e avaliação já
existe em `avaliar_sequencial.py`; faltam o proxy de reputação, os dados reais e
a validação.
