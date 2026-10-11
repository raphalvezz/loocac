#!/usr/bin/env python3
"""
Robustez à forma funcional da demanda: repete a comparação de avaliar_politicas.py
com a demanda LINEAR (simulador.usar_forma('linear')) no lugar da logística.

Se a ordem das abordagens e as conclusões (critérios C1-C3) se mantêm com uma
curva diferente, o resultado não depende da fórmula escolhida para o simulador.

Saídas: resultados/robustez_linear/ (mesmos arquivos de avaliar_politicas.py)
Uso:    python avaliar_robustez.py [--sementes 3] [--passos-cql 20000] [--sem-cql]
"""

import argparse
import json
import os
from types import SimpleNamespace

import avaliar_politicas as av
import simulador as sim


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--forma', default='linear', choices=[f for f in sim.FORMAS_DEMANDA if f != 'logistica'])
    ap.add_argument('--sementes', type=int, default=3)
    ap.add_argument('--passos-cql', type=int, default=20000)
    ap.add_argument('--sem-cql', action='store_true')
    ap.add_argument('--saida', default=None)
    a = ap.parse_args()
    saida = a.saida or f'resultados/robustez_{a.forma}'

    sim.usar_forma(a.forma)
    args = SimpleNamespace(sementes=list(range(42, 42 + a.sementes)), n_por_cenario=5000,
                           passos_cql=a.passos_cql, sem_cql=a.sem_cql)
    por_semente, custos = av.avaliar(args)
    os.makedirs(saida, exist_ok=True)
    tabela = av.resumir(por_semente)
    por_semente.to_csv(os.path.join(saida, 'por_semente.csv'), index=False)
    tabela.to_csv(os.path.join(saida, 'tabela_artigo.csv'), index=False)
    av.comparar(por_semente).to_csv(os.path.join(saida, 'comparacoes.csv'), index=False)
    with open(os.path.join(saida, 'manifesto.json'), 'w') as f:
        json.dump({'forma_demanda': a.forma, 'simulador_versao': sim.VERSAO, 'sementes': args.sementes,
                   'passos_cql': None if a.sem_cql else a.passos_cql}, f, indent=2)
    print(tabela.pivot_table(index=['modelo', 'politica'], columns='tier', values='pct_do_otimo',
                             sort=False).round(1).to_string())


if __name__ == '__main__':
    main()
