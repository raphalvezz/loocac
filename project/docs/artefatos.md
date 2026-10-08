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
| 3 | `código_final_RL_OFF (25).ipynb` | `modelo_rl_final.d3` (CQL, venda única) |
| 4 | `RL_assinatura (5).ipynb` | `modelo_rl_assinatura.d3` (CQL, assinatura) |

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
  modelo_rl_final.d3 modelo_rl_assinatura.d3 \
  sl_profit_regressor_model.joblib sl_ltv_regressor_model.joblib \
  bandido_venda_unica.joblib bandido_assinatura.joblib \
  data_manifest.json
```

## Usar uma versão publicada

```bash
cd project
gh release download modelos-AAAAMMDD     # baixa os anexos para a pasta atual
python ../start.py                       # encontra os modelos e só sobe a API
```

Os encoders/scalers e o dataset continuam vindo do `Generator_NEW.py` (ou podem
ser anexados à mesma release). A API precisa dos artefatos da MESMA execução do
pipeline: não misture modelos de uma release com encoders de outra geração.
