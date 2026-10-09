# LOCAC — Precificação de campanhas com RL offline

Protótipo do TCC: recomenda o preço de um produto digital (venda única ou
assinatura) para uma campanha, a partir de região, plataforma, tier e orçamento.
Compara quatro abordagens sobre os mesmos dados simulados: RL offline (CQL),
bandido contextual, aprendizado supervisionado (SL) e imitação do histórico (BC).
A interface é a página **Campaign Simulator** da rede social ConnectPro.

## Estrutura

```
start.py                    sobe a API (e roda o pipeline se faltar algum modelo)
requirements.txt            dependências Python (iguais às de project/)
project/
  simulador.py              Gêmeo Digital econômico (congelado, v1.0)
  Generator_NEW.py          gera o dataset logado e os encoders
  politicas.py              interface única de política (SL, BC, CQL, heurísticas, oráculos)
  bandido.py                bandido contextual (offline, ε-greedy, Thompson)
  treinar_bandido.py / treinar_bc.py / treinar_cql.py   treino e exportação
  SL_FINAL (1).ipynb        SL: métricas, SHAP e exportação
  train_pipeline.py         gera dados e treina tudo, em ordem
  main.py                   API FastAPI (/recomendar, /comparar, /configure_market)
  avaliar_politicas.py      comparação com baselines (sementes, IC 95%)
  avaliar_generalizacao.py  regiões e cenários fora do treino
  simular_bandido_online.py bandido aprendendo em operação
  ablacoes_cql.py / diagnostico_cql.py   ablações e diagnóstico do CQL
  resultados/               CSVs para o capítulo 5
  docs/                     protocolo, artefatos, segurança e figuras
  src/                      front-end React (ConnectPro + Campaign Simulator)
  tests/                    pytest (simulador, políticas, API de ponta a ponta)
```

Os notebooks `código_final_RL_OFF (25).ipynb` e `RL_assinatura (5).ipynb`
documentam a versão anterior do agente; a versão canônica é `politicas.CQL`.

## Instalação

```bash
pip install -r requirements.txt     # Python 3.11+; torch em CPU basta
cd project && npm ci                # front-end (node_modules não é versionado)
```

## Gerar os modelos

```bash
cd project
python -m pytest tests              # confirma o simulador v1.0 (cerca de 1 min)
python train_pipeline.py            # dados + SL + bandido + BC + CQL (algumas horas em CPU)
```

Os modelos não ficam no git; ver [`project/docs/artefatos.md`](project/docs/artefatos.md)
para o que cada passo gera e como publicar/baixar uma versão como release.

## Rodar o sistema

```bash
# 1. API (na pasta project/)
export LOCAC_API_KEY="uma-chave-longa"
uvicorn main:app --port 8000        # ou: python ../start.py

# 2. Tela (outro terminal, na pasta project/)
cp .env.example .env.local          # coloque a mesma chave em VITE_LOCAC_API_KEY
npm run dev                         # http://localhost:5173 -> Campaign Simulator
```

Exemplo de chamada direta:

```bash
curl -X POST "http://127.0.0.1:8000/comparar?cobranca=venda_unica" \
  -H "X-API-Key: $LOCAC_API_KEY" -H "Content-Type: application/json" \
  -d '{"Regiao": "Europe", "Plataforma": "Instagram", "Tier": "Low Ticket", "Orcamento": 1000}'
```

## Resultados (capítulo 5)

```bash
cd project
python avaliar_politicas.py         # 5 sementes, todas as abordagens
python avaliar_generalizacao.py
python simular_bandido_online.py --n-inicial 10 && python figura_bandido_online.py
python ablacoes_cql.py && python diagnostico_cql.py
```

Métrica, comparações, critérios de sucesso e desvios registrados:
[`project/docs/protocolo_avaliacao.md`](project/docs/protocolo_avaliacao.md).

## Segurança

Chave de API, CORS restrito e validação de entrada; sem autenticação de
usuários. Detalhes e texto para a seção de Limitações em
[`project/docs/seguranca.md`](project/docs/seguranca.md).

## Histórico

Os guias anteriores (`INICIO_RAPIDO.md`, `README_EXECUCAO_COMPLETA.md`,
`SOLUCAO_ValueError.md`, `RESUMO_SOLUCOES.txt`, `INDEX.txt`) documentavam
correções da primeira versão: geração de dados dentro dos notebooks, preço
normalizado sem conversão e o `ValueError` de `region_metrics`. Essas correções
foram incorporadas e depois substituídas pelo gerador, pelo simulador e pelos
scripts acima; os guias continuam no histórico do git.
