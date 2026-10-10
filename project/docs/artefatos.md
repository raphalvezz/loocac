# Artefatos treinados (fora do repositório)

Os modelos treinados não são versionados no git (`.gitignore`: `*.d3`, `*.pt`,
`project/d3rlpy_logs/`). Eles são **gerados** por um comando e, quando for
preciso compartilhar uma versão (banca, artigo), **publicados como anexos de uma
release** do GitHub.

## Gerar

```bash
cd project
python -m pytest tests        # confirma o simulador v1.0 (dados reproduzíveis)
python train_pipeline.py      # ou: python ../start.py (gera se faltar e sobe a API)
```

O pipeline executa, em ordem:

| Passo | Comando | Gera |
|---|---|---|
| 1 | `Generator_NEW.py` | `sl_dataset_combined.csv`, buffers `.h5`, encoders/scalers `.joblib`, `faixas_tier.json`, `cenarios_treino.json`, `data_manifest.json` |
| 2 | `SL_FINAL (1).ipynb` | `sl_profit_regressor_model.joblib`, `sl_ltv_regressor_model.joblib`, `docs/figuras/shap_sl_venda_unica.png` |
| 2b | `treinar_bandido.py` | `bandido_venda_unica.joblib`, `bandido_assinatura.joblib` |
| 2c | `treinar_bc.py` | `bc_venda_unica.d3/.joblib`, `bc_assinatura.d3/.joblib` |
| 3 | `treinar_cql.py` (versão canônica do CQL) | `cql_venda_unica.d3/.joblib`, `cql_assinatura.d3/.joblib` |

Os notebooks de RL (`código_final_RL_OFF (25).ipynb`, `RL_assinatura (5).ipynb`)
documentam a versão anterior e não são executados pelo pipeline.

Os dados são determinísticos (semente 42): o `data_manifest.json` traz o sha256
de cada arquivo de dados, para conferir que duas máquinas geraram o mesmo
dataset. O treino do CQL depende de torch/CPU e não é bit a bit reproduzível;
por isso a comparação entre abordagens usa várias sementes
(`avaliar_politicas.py`, ver `docs/protocolo_avaliacao.md`).

Tempo aproximado em CPU de 4 núcleos: gerador e SL em poucos minutos; cada CQL
com 50 mil passos, de 40 a 60 minutos (menos se o critério de parada por lucro
simulado encerrar antes).

## Publicar uma versão (release)

Release em vez de Git LFS: não consome cota de LFS, cada versão dos modelos fica
presa a uma tag (o commit que a gerou) e o download não exige configuração.

```bash
cd project
TAG=modelos-$(date +%Y%m%d)
gh release create "$TAG" --title "Modelos treinados $TAG" \
  --notes "Gerados por train_pipeline.py no commit $(git rev-parse --short HEAD). Simulador v1.0." \
  cql_venda_unica.d3 cql_venda_unica.joblib cql_assinatura.d3 cql_assinatura.joblib \
  bc_venda_unica.d3 bc_venda_unica.joblib bc_assinatura.d3 bc_assinatura.joblib \
  sl_profit_regressor_model.joblib sl_ltv_regressor_model.joblib \
  bandido_venda_unica.joblib bandido_assinatura.joblib \
  scaler_assinatura_memoria.joblib cenarios_treino.json faixas_tier.json data_manifest.json
```

## Usar uma versão publicada

```bash
cd project
gh release download modelos-AAAAMMDD     # baixa os anexos para a pasta atual
python ../start.py                       # encontra os modelos e só sobe a API
```

Os anexos acima bastam para a API: cada modelo leva o próprio encoder dentro do
`.joblib`, e `scaler_assinatura_memoria.joblib` dá o valor padrão da memória da
assinatura. Não misture arquivos de releases diferentes.
