#!/usr/bin/env python3
"""
Avaliação fora da política (FQE) como segunda forma de medir as políticas d3rlpy.

O lucro no simulador usa o "mundo verdadeiro". Na prática ele não existe: só há
os dados logados. O FQE (Fitted Q Evaluation, d3rlpy.ope) aprende Q^π(s, a) da
política a partir desses dados e estima o lucro dela em cada estado como
Q^π(s, π(s)). Aqui comparamos essa estimativa com o lucro esperado verdadeiro
do simulador nos 144 estados de avaliação.

O FQE é ajustado nos 20% finais do dataset, que os agentes NÃO usaram no treino.
Com γ = 0 (decisões independentes), Q^π(s, a) é o lucro esperado do par.

Entradas: cql_<modelo>.* e bc_<modelo>.* (treinar_cql.py / treinar_bc.py) e
sl_dataset_combined.csv, na pasta atual.
Saída:    resultados/fqe.csv (+ fqe_por_estado.csv)
Uso:      python avaliar_fqe.py [--passos 10000]
"""

import argparse
import os

import numpy as np
import pandas as pd

import avaliar_politicas as av
import politicas as pol
import simulador as sim


def ajustar_fqe(agente, teste, passos):
    import d3rlpy
    from d3rlpy.dataset import Episode, FIFOBuffer, ReplayBuffer
    from d3rlpy.ope import FQE, FQEConfig

    lucro = teste[pol.ALVO[agente.modelo]].to_numpy(dtype=float)
    media, desvio = float(lucro.mean()), float(lucro.std())
    episodio = Episode(agente.prep.transform(teste),
                       agente.preco_para_acao(teste['Preco_Amostra'].to_numpy(), teste).reshape(-1, 1).astype(np.float32),
                       ((lucro - media) / desvio).reshape(-1, 1).astype(np.float32), True)
    buffer = ReplayBuffer(FIFOBuffer(limit=len(teste)), episodes=[episodio])
    fqe = FQE(algo=agente.algo, config=FQEConfig(gamma=0.0, learning_rate=1e-4), device='cpu')
    fqe.fit(buffer, n_steps=passos, n_steps_per_epoch=min(1000, passos),
            logger_adapter=d3rlpy.logging.NoopAdapterFactory(), show_progress=False)
    return fqe, media, desvio


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--passos', type=int, default=10000)
    ap.add_argument('--saida', default='resultados')
    args = ap.parse_args()

    dados = pd.read_csv('sl_dataset_combined.csv')
    teste = dados.iloc[int(len(dados) * pol.FRACAO_TREINO):].reset_index(drop=True)
    cenarios = av.carregar_cenarios()
    estados = sim.contextos(cenarios)
    linhas, por_estado = [], []
    for modelo in av.MODELOS:
        otimo = av.lucros(av.consultar(pol.Oraculo(modelo, cenarios), estados), estados, cenarios, modelo)
        for classe, prefixo in [(pol.CQL, f'cql_{modelo}'), (pol.BC, f'bc_{modelo}')]:
            agente = classe.carregar(prefixo)
            fqe, media, desvio = ajustar_fqe(agente, teste, args.passos)
            sem_cenario = estados.drop(columns='Cenario')
            precos = agente.precos(sem_cenario)
            obs = agente.prep.transform(sem_cenario)
            acoes = agente.preco_para_acao(precos, sem_cenario).reshape(-1, 1).astype(np.float32)
            estimado = fqe.predict_value(obs, acoes) * desvio + media
            verdadeiro = av.lucros(precos, estados, cenarios, modelo)
            pct_est, pct_ver = 100 * np.mean(estimado / otimo), 100 * np.mean(verdadeiro / otimo)
            linhas.append({
                'modelo': modelo, 'politica': agente.nome,
                'pct_do_otimo_simulador': pct_ver, 'pct_do_otimo_fqe': pct_est,
                'erro_pp': pct_est - pct_ver,
                'erro_relativo_medio': float(np.mean(np.abs(estimado - verdadeiro) / verdadeiro)),
                'correlacao_por_estado': float(np.corrcoef(estimado, verdadeiro)[0, 1]),
            })
            por_estado += [{'modelo': modelo, 'politica': agente.nome, 'cenario': int(c), 'tier': t,
                            'lucro_simulador': v, 'lucro_fqe': e}
                           for c, t, v, e in zip(estados['Cenario'], estados['Tier'], verdadeiro, estimado)]
            print(f"  {modelo:12s} {agente.nome}: simulador {pct_ver:.1f}% | FQE {pct_est:.1f}% do ótimo", flush=True)
    os.makedirs(args.saida, exist_ok=True)
    pd.DataFrame(linhas).to_csv(os.path.join(args.saida, 'fqe.csv'), index=False)
    pd.DataFrame(por_estado).to_csv(os.path.join(args.saida, 'fqe_por_estado.csv'), index=False)
    with pd.option_context('display.float_format', '{:,.3f}'.format, 'display.width', 140):
        print(pd.DataFrame(linhas).to_string(index=False))


if __name__ == '__main__':
    main()
