#!/usr/bin/env python3
"""
Bandido mais robusto ao público novo? Análise de sensibilidade (não faz parte da
comparação principal, que fica com o bandido registrado em bandido.BandidoLGBM).

No teste de deslocamento (avaliar_deslocamento.py) o bandido perdeu 12–14 p.p.
ao trocar de público. Aqui comparamos o bandido original com duas versões
regularizadas do mesmo LightGBM (árvores menores, mais amostras por folha,
penalidade L2, amostragem de colunas):

  1. deslocamento: treino em North America/Europe, avaliação na origem e em
     Asia/South America (mesmo protocolo do avaliar_deslocamento.py);
  2. custo dentro da distribuição: treino com todas as regiões, avaliação nos
     144 estados (mesmo protocolo do avaliar_politicas.py).

Saída: resultados/bandido_robusto/{por_semente,tabela}.csv
Uso:   python avaliar_bandido_robusto.py [--sementes 3]
"""

import argparse
import os

import pandas as pd

import avaliar_deslocamento as desl
import avaliar_politicas as av
import bandido as ban
import simulador as sim


class BandidoRegularizado(ban.BandidoLGBM):
    nome = 'bandido_reg'
    parametros = {'n_estimators': 300, 'learning_rate': 0.05, 'num_leaves': 15, 'min_child_samples': 200,
                  'reg_lambda': 10.0, 'colsample_bytree': 0.8}


class BandidoRegularizadoForte(ban.BandidoLGBM):
    nome = 'bandido_reg_forte'
    parametros = {'n_estimators': 300, 'learning_rate': 0.05, 'num_leaves': 7, 'min_child_samples': 500,
                  'reg_lambda': 50.0, 'colsample_bytree': 0.6}


VARIANTES = [ban.BandidoLGBM, BandidoRegularizado, BandidoRegularizadoForte]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--sementes', type=int, default=3)
    ap.add_argument('--saida', default='resultados/bandido_robusto')
    args = ap.parse_args()

    cenarios = av.carregar_cenarios()
    todos = sim.contextos(cenarios)
    grupos = {'origem': todos[todos['Regiao'].isin(desl.ORIGEM)].reset_index(drop=True),
              'destino': todos[todos['Regiao'].isin(desl.DESTINO)].reset_index(drop=True),
              'todas_regioes': todos}
    os.makedirs(args.saida, exist_ok=True)
    linhas = []
    for semente in range(42, 42 + args.sementes):
        completos = sim.gerar_dados(cenarios, 5000, semente)
        so_origem = completos[completos['Regiao'].isin(desl.ORIGEM)].reset_index(drop=True)
        for modelo in av.MODELOS:
            for Classe in VARIANTES:
                p = Classe(modelo, cenarios).treinar(so_origem, semente)
                for grupo in ('origem', 'destino'):
                    linhas.append({'modelo': modelo, 'politica': p.nome, 'semente': semente, 'treino': 'origem',
                                   'avaliado_em': grupo,
                                   'pct_do_otimo': 100 * desl.metrica(modelo, cenarios, grupos[grupo])(p)})
                p = Classe(modelo, cenarios).treinar(completos, semente)
                linhas.append({'modelo': modelo, 'politica': p.nome, 'semente': semente, 'treino': 'todas_regioes',
                               'avaliado_em': 'todas_regioes',
                               'pct_do_otimo': 100 * desl.metrica(modelo, cenarios, todos)(p)})
                ult = {r['avaliado_em']: r['pct_do_otimo'] for r in linhas[-3:]}
                print(f"  s{semente} {modelo:12s} {p.nome:18s} origem {ult['origem']:5.1f}%  destino "
                      f"{ult['destino']:5.1f}%  | todas as regiões {ult['todas_regioes']:5.1f}%", flush=True)
                pd.DataFrame(linhas).to_csv(os.path.join(args.saida, 'por_semente.csv'), index=False)

    det = pd.DataFrame(linhas)
    tabela = []
    for (modelo, nome), g in det.groupby(['modelo', 'politica'], sort=False):
        piv = g.pivot_table(index='semente', columns='avaliado_em', values='pct_do_otimo')
        linha = {'modelo': modelo, 'politica': nome, 'n_sementes': len(piv)}
        for col in ('origem', 'destino', 'todas_regioes'):
            linha[f'pct_{col}'], linha[f'{col}_ic95_inf'], linha[f'{col}_ic95_sup'] = av.ic95(piv[col])
        linha['queda_pp'], linha['queda_ic95_inf'], linha['queda_ic95_sup'] = av.ic95(piv['origem'] - piv['destino'])
        tabela.append(linha)
    tabela = pd.DataFrame(tabela)
    tabela.to_csv(os.path.join(args.saida, 'tabela.csv'), index=False)
    with pd.option_context('display.float_format', '{:,.1f}'.format, 'display.width', 160):
        print(tabela[['modelo', 'politica', 'pct_origem', 'pct_destino', 'queda_pp', 'pct_todas_regioes']]
              .to_string(index=False))


if __name__ == '__main__':
    main()
