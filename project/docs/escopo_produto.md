# Escopo do produto: decisão de passo único

**Decisão (2026-10-11):** a API e a tela servem a decisão de **passo único**: dado
o estado da campanha, recomendam um preço. O ambiente sequencial
(`sequencial.py`) **não** está integrado ao produto. Ele é a evidência
experimental de quando vale migrar para RL que planeja.

**Atualização (2026-10-14): rota experimental `/recomendar_sequencial`.** A API passou
a servir também o CQL com γ = 0,95 do ambiente sequencial, com o estado da campanha
persistido por cliente (detalhes abaixo, em "Rota experimental"). A decisão acima
continua valendo para o produto principal: a rota sequencial é uma demonstração de
como o agente seria servido, e a reputação que ela usa é a do Gêmeo Digital, não uma
medida real. Os motivos abaixo explicam por que ela não substitui `/recomendar`.

## Por que o agente sequencial não é o padrão do produto

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

## Rota experimental (opção b, implementada em 2026-10-14)

```
POST   /recomendar_sequencial?cobranca=venda_unica   corpo: estado da campanha + cliente_id
GET    /sequencial/{cliente_id}?cobranca=...         estado atual e histórico
DELETE /sequencial/{cliente_id}?cobranca=...         reinicia a campanha
```

- **Modelo:** `cql_seq_{cobrança}` (`treinar_cql_sequencial.py`, ou
  `train_pipeline.py --sequencial`), a mesma configuração do `cql_g095` da avaliação:
  estado com reputação e período, recompensa por contexto, 50.000 passos. Sem o
  arquivo, a rota devolve 503 e o resto da API funciona.
- **Estado persistido:** SQLite (`estado_sequencial.py`, arquivo `LOCAC_ESTADO_DB`,
  padrão `estado_sequencial.sqlite`). Escolhido por ser da biblioteca padrão,
  sobreviver a reinícios da API e não exigir servidor; cada chamada lê e grava a
  campanha numa transação exclusiva, então duas chamadas simultâneas do mesmo cliente
  não avançam o mesmo período duas vezes. Uma campanha é (cliente_id, cobrança): a
  primeira chamada fixa o contexto (região, plataforma, tier, orçamento) e começa com
  ρ = 1 no período 0; chamadas com outro contexto recebem 409; depois de 12 períodos a
  campanha se encerra (409) até um DELETE.
- **Cada chamada** recomenda o preço do período atual, supõe que ele foi praticado
  e avança o estado: ρ' = clip(ρ + κ (p_ref − p) / p_ref), com p_ref do cenário de
  treino mais parecido (mesmo Tier, orçamento mais próximo), e devolve o lucro
  esperado do Gêmeo Digital naquele período.
- **O que falta para uso real:** trocar a reputação simulada por um proxy observável
  (conversão recente contra a esperada, por exemplo), receber o preço efetivamente
  praticado e o resultado observado em vez de supor o recomendado, e autenticar o
  cliente (hoje qualquer portador da chave da API lê ou reinicia qualquer cliente_id;
  ver `docs/seguranca.md`).
