# Verificação em máquina limpa (2026-10-11)

Clone novo do repositório (commit `a7674f4`) e ambiente Python novo
(`python -m venv` + `pip install -r requirements.txt`, sem pacotes do sistema).
Linux, 4 CPUs, sem GPU; Python 3.13.16, Node 22.22.0, npm 10.9.4;
torch 2.14.1, d3rlpy 2.8.1, scikit-learn 1.7.2, LightGBM 4.7.0.

| Passo | Comando | Resultado |
|---|---|---|
| Testes | `python -m pytest tests` | **48 passaram**, 0 falhas (16 min com 1 thread) |
| Front | `npm ci && npm run build` | build ok (370 kB de JS) |
| Lint | `npm run lint` | 0 erros, 1 aviso (`react-refresh/only-export-components` em `UserContext.tsx`) |
| Pipeline | `python train_pipeline.py` | ok em 4 h 35 min (CQL: 2 h 10 min por cobrança, com 1 thread) |
| API | `uvicorn main:app` + `python verificar_api.py` | **30 chamadas, 0 falhas** |

Modelos gerados pelo pipeline (% do ótimo na grade de avaliação, semente 42):

| | Venda única | Assinatura |
|---|---|---|
| Bandido | 82,8 | 99,0 |
| BC | 82,6 | 97,7 |
| CQL (melhor checkpoint) | 85,0 | 99,0 |

Os números coincidem com a rodada anterior: o pipeline é reprodutível.

`verificar_api.py` chama `/recomendar` para cada método (`rl`, `sl`, `bandido`,
`bc`), cobrança e três casos (Low Ticket com orçamento 1.000; High Ticket com 50
mil e com 200 mil) e `/comparar` para cada cobrança e caso. Confere status 200,
preço dentro da faixa observável do estado, lucro SL e VaR preenchidos, e a
rejeição (401) sem a chave. Latência de 17 a 68 ms por chamada. Resposta
completa em `resultados/verificacao_api.json`.

## Problemas encontrados antes desta rodada (corrigidos em `a7674f4`)

- A regra de segurança da API limitava o High Ticket a 497–5.000 e cortava as
  recomendações dos orçamentos de 100 e 200 mil (faixas até 25.000). Agora usa
  as faixas de treino; teste em `tests/test_api_e2e.py`.
- Sem memória informada, a assinatura usava a média global do treino e o SL
  previa LTV de 11 milhões num Low Ticket de orçamento 1.000. Agora usa a
  memória do cenário de treino mais parecido (114 mil).
- Num checkout antigo, `npm run build` falhava com `vite: Permission denied`
  (`node_modules` desatualizado). Resolve com `rm -rf node_modules && npm ci`.

## Como repetir

```bash
git clone <repo> && cd loocac
python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
cd project
python -m pytest tests
npm ci && npm run build && npm run lint
python train_pipeline.py
LOCAC_API_KEY=teste uvicorn main:app --port 8000 &
LOCAC_API_KEY=teste python verificar_api.py
```
