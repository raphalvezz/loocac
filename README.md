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

O RL é treinado só por `treinar_cql.py` (`politicas.CQL`). Os notebooks de RL da
versão anterior (`código_final_RL_OFF (25).ipynb`, `RL_assinatura (5).ipynb`) e o
backup do gerador antigo saíram do repositório; continuam no histórico do git
(`git show 0f1d302:"project/RL_assinatura (5).ipynb"`). O único notebook que
resta, `SL_FINAL (1).ipynb`, faz parte do pipeline.

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
python avaliar_politicas.py --saida resultados/final_10_sementes  # comparação principal, 10 sementes
python avaliar_sequencial.py && python figura_sequencial.py         # ambiente sequencial
python avaliar_robustez.py && python avaliar_deslocamento.py && python avaliar_fqe.py
```

Lista completa, tempos e saídas: seção "Como reproduzir" do protocolo.

Métrica, comparações, critérios de sucesso e desvios registrados:
[`project/docs/protocolo_avaliacao.md`](project/docs/protocolo_avaliacao.md).

## Escopo

A API e a tela servem a decisão de preço de **passo único**. O ambiente sequencial
(`project/sequencial.py`, o preço de hoje muda a demanda de amanhã) mostra quando o
RL que planeja passa a valer a pena; a rota experimental `POST /recomendar_sequencial`
serve esse agente com o estado da campanha (reputação do Gêmeo Digital e período)
persistido por `cliente_id` em SQLite. Ela exige `python train_pipeline.py --sequencial`.
Motivos, limites e texto para a monografia em
[`project/docs/escopo_produto.md`](project/docs/escopo_produto.md).

Verificar a API em execução com os modelos treinados (uma chamada por método e
cobrança): `LOCAC_API_KEY=... python verificar_api.py` (na pasta `project/`).
Resultado em máquina limpa: [`project/docs/verificacao.md`](project/docs/verificacao.md).
Limitações reunidas para a monografia: [`project/docs/limitacoes.md`](project/docs/limitacoes.md).
Exemplo de retorno (hipotético): [`project/docs/roi_ilustrativo.md`](project/docs/roi_ilustrativo.md).

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
