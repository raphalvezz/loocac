#!/usr/bin/env python3
"""Verificação de ponta a ponta da API em execução, com os modelos treinados.

Faz uma chamada por método, cobrança e caso em /recomendar, uma em /comparar por
cobrança e caso e confere: status 200, preço dentro da faixa observável do estado
(Tier + orçamento; o caso de 200 mil pega um corte indevido da regra de segurança),
campos da avaliação comum preenchidos e rejeição sem a chave de API.

Pré-requisito: python train_pipeline.py e a API no ar
    LOCAC_API_KEY=... uvicorn main:app --port 8000
Uso:   LOCAC_API_KEY=... python verificar_api.py [--url http://127.0.0.1:8000]
Saída: resultados/verificacao_api.json (e código de saída 1 se algo falhar)
"""
import argparse
import json
import os
import platform
import sys
from datetime import datetime, timezone

import httpx

import simulador as sim

CASOS = [
    {'Regiao': 'Europe', 'Plataforma': 'Instagram', 'Tier': 'Low Ticket', 'Orcamento': 1000},
    {'Regiao': 'Asia', 'Plataforma': 'LinkedIn', 'Tier': 'High Ticket', 'Orcamento': 50000},
    {'Regiao': 'North America', 'Plataforma': 'Facebook', 'Tier': 'High Ticket', 'Orcamento': 200000},
]


def faixa_observavel(caso):
    """União das faixas dos cenários com o mesmo Tier e orçamento (o que os dados mostram)."""
    cs = [c for c in sim.CENARIOS_ARTIGO if c['Tier'] == caso['Tier'] and c['Budget'] == caso['Orcamento']]
    return min(c['Price_Min'] for c in cs), max(c['Price_Max'] for c in cs)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--url', default='http://127.0.0.1:8000')
    ap.add_argument('--saida', default='resultados/verificacao_api.json')
    args = ap.parse_args()
    chave = os.environ.get('LOCAC_API_KEY', '')
    cli = httpx.Client(base_url=args.url, headers={'X-API-Key': chave}, timeout=120)
    falhas, chamadas = [], []

    def checar(cond, msg):
        if not cond:
            falhas.append(msg)
            print('  FALHOU:', msg)

    raiz = cli.get('/').json()
    print('API:', raiz)
    for cobranca in ('venda_unica', 'assinatura'):
        for caso in CASOS:
            lo, hi = faixa_observavel(caso)
            for metodo in ('rl', 'sl', 'bandido', 'bc'):
                r = cli.post('/recomendar', params={'metodo': metodo, 'cobranca': cobranca}, json=caso)
                corpo = r.json()
                chamadas.append({'rota': '/recomendar', 'metodo': metodo, 'cobranca': cobranca,
                                 'entrada': caso, 'status': r.status_code, 'resposta': corpo})
                rotulo = f'{metodo}/{cobranca}/{caso["Tier"]}/{caso["Orcamento"]:g}'
                checar(r.status_code == 200, f'{rotulo}: status {r.status_code} {corpo}')
                if r.status_code != 200:
                    continue
                p = corpo['preco_recomendado']
                checar(lo * 0.999 <= p <= hi * 1.001, f'{rotulo}: preço {p:.2f} fora de [{lo:.2f}, {hi:.2f}]')
                checar(corpo['lucro_estimado_sl'] is not None, f'{rotulo}: sem lucro_estimado_sl')
                checar(corpo['var_5_percent'] is not None, f'{rotulo}: sem VaR')
                print(f'  {rotulo:32s} preço {p:12,.2f}  lucro SL {corpo["lucro_estimado_sl"] or 0:14,.0f}  '
                      f'{corpo["latencia_ms"]:.0f} ms')
            r = cli.post('/comparar', params={'cobranca': cobranca}, json=caso)
            chamadas.append({'rota': '/comparar', 'cobranca': cobranca, 'entrada': caso,
                             'status': r.status_code, 'resposta': r.json()})
            checar(r.status_code == 200 and {x['metodo'] for x in r.json()} == {'rl', 'sl', 'bandido', 'bc'},
                   f'/comparar {cobranca}: {r.status_code}')
    sem_chave = httpx.post(f'{args.url}/recomendar', json=CASOS[0], timeout=30)
    checar(sem_chave.status_code == 401, f'sem chave: esperado 401, veio {sem_chave.status_code}')

    os.makedirs(os.path.dirname(args.saida) or '.', exist_ok=True)
    with open(args.saida, 'w') as f:
        json.dump({'gerado_em': datetime.now(timezone.utc).isoformat(timespec='seconds'),
                   'python': platform.python_version(), 'api': raiz, 'falhas': falhas,
                   'chamadas': chamadas}, f, indent=2, ensure_ascii=False)
    print(f'{len(chamadas)} chamadas, {len(falhas)} falhas -> {args.saida}')
    sys.exit(1 if falhas else 0)


if __name__ == '__main__':
    main()
