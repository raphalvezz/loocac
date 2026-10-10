# Segurança do protótipo

## O que está implementado (`main.py`)

| Medida | Como configurar |
|---|---|
| Chave de API no header `X-API-Key`, exigida em todas as rotas exceto `GET /` | `LOCAC_API_KEY` (sem ela, a API gera uma chave aleatória e a imprime ao subir) |
| CORS restrito às origens da tela | `LOCAC_CORS_ORIGINS` (padrão: `http://localhost:5173,http://127.0.0.1:5173`) |
| Validação de entrada: região, plataforma e Tier em listas fechadas; limites numéricos; campos extras recusados (422) | — |
| `/configure_market` protegida pela chave, com faixas validadas (mín. < máx.) e um único retreino por vez (409 se já houver um) | — |
| Métodos carregados de forma independente; um modelo ausente desativa só ele (503 com o motivo) | — |

A tela lê a URL e a chave de `VITE_LOCAC_API_URL` e `VITE_LOCAC_API_KEY`
(arquivo `project/.env.local`, não versionado; modelo em `project/.env.example`).

```bash
# API
export LOCAC_API_KEY="uma-chave-longa"
cd project && uvicorn main:app --port 8000
# Tela (outro terminal), com a mesma chave em .env.local
npm run dev
```

## Texto para a seção de Limitações

> O protótipo adota apenas medidas mínimas de segurança: uma chave de API
> compartilhada, verificada em cada requisição, restrição das origens aceitas
> (CORS) e validação dos dados de entrada. Não há autenticação nem autorização
> por usuário: a chave é a mesma para todos e, na interface web, fica exposta no
> código entregue ao navegador, servindo apenas para impedir chamadas
> acidentais ou de origens não previstas durante o desenvolvimento local. Também
> não há proteção de dados pessoais (criptografia em repouso, registro de
> acesso, política de retenção), limitação de taxa de requisições nem
> isolamento do processo de retreino, que roda no mesmo servidor da API. A rede
> social ConnectPro usa um login simulado no navegador. Antes de qualquer uso
> real, sobretudo com dados de usuários da rede social, esses pontos se tornam
> requisitos: autenticação individual (por exemplo, OAuth 2.0/OpenID Connect),
> autorização por perfil, transporte apenas por HTTPS, armazenamento seguro de
> segredos e adequação à LGPD.
