"""
Estado persistido das campanhas sequenciais (rota /recomendar_sequencial da API).

Cada campanha é identificada por (cliente_id, cobrança) e guarda o contexto
fixo (região, plataforma, tier, orçamento), a reputação atual ρ e o período.
Cada recomendação grava uma linha no histórico.

Armazenamento: SQLite (biblioteca padrão), um arquivo local
(LOCAC_ESTADO_DB, padrão estado_sequencial.sqlite). Sobrevive a reinícios da
API, não exige servidor e aguenta o volume de um protótipo. As operações de
leitura e escrita de uma campanha são feitas numa transação única
(BEGIN IMMEDIATE), então duas chamadas simultâneas do mesmo cliente não
avançam o mesmo período duas vezes.
"""

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

ESQUEMA = """
CREATE TABLE IF NOT EXISTS campanhas (
    cliente_id  TEXT NOT NULL,
    cobranca    TEXT NOT NULL,
    contexto    TEXT NOT NULL,
    reputacao   REAL NOT NULL,
    periodo     INTEGER NOT NULL,
    criado_em   TEXT NOT NULL,
    atualizado_em TEXT NOT NULL,
    PRIMARY KEY (cliente_id, cobranca)
);
CREATE TABLE IF NOT EXISTS historico (
    cliente_id  TEXT NOT NULL,
    cobranca    TEXT NOT NULL,
    periodo     INTEGER NOT NULL,
    preco       REAL NOT NULL,
    reputacao_antes  REAL NOT NULL,
    reputacao_depois REAL NOT NULL,
    lucro_esperado   REAL NOT NULL,
    registrado_em    TEXT NOT NULL
);
"""


def caminho_padrao():
    return os.environ.get("LOCAC_ESTADO_DB", "estado_sequencial.sqlite")


def _agora():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def transacao(caminho=None):
    """Conexão própria por chamada (seguro entre threads), em transação exclusiva de escrita."""
    con = sqlite3.connect(caminho or caminho_padrao(), timeout=30, isolation_level=None)
    con.row_factory = sqlite3.Row
    try:
        con.executescript(ESQUEMA)
        con.execute("BEGIN IMMEDIATE")
        yield con
        con.execute("COMMIT")
    except BaseException:
        con.execute("ROLLBACK")
        raise
    finally:
        con.close()


def ler(con, cliente_id, cobranca):
    linha = con.execute("SELECT * FROM campanhas WHERE cliente_id = ? AND cobranca = ?",
                        (cliente_id, cobranca)).fetchone()
    if linha is None:
        return None
    d = dict(linha)
    d["contexto"] = json.loads(d["contexto"])
    return d


def criar(con, cliente_id, cobranca, contexto, reputacao):
    agora = _agora()
    con.execute("INSERT INTO campanhas VALUES (?, ?, ?, ?, 0, ?, ?)",
                (cliente_id, cobranca, json.dumps(contexto, sort_keys=True), reputacao, agora, agora))
    return ler(con, cliente_id, cobranca)


def avancar(con, cliente_id, cobranca, periodo, preco, reputacao_antes, reputacao_depois, lucro_esperado):
    agora = _agora()
    con.execute("UPDATE campanhas SET reputacao = ?, periodo = ?, atualizado_em = ? "
                "WHERE cliente_id = ? AND cobranca = ?",
                (reputacao_depois, periodo + 1, agora, cliente_id, cobranca))
    con.execute("INSERT INTO historico VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (cliente_id, cobranca, periodo + 1, preco, reputacao_antes, reputacao_depois,
                 lucro_esperado, agora))


def historico(con, cliente_id, cobranca):
    return [dict(r) for r in con.execute(
        "SELECT periodo, preco, reputacao_antes, reputacao_depois, lucro_esperado, registrado_em "
        "FROM historico WHERE cliente_id = ? AND cobranca = ? ORDER BY periodo", (cliente_id, cobranca))]


def apagar(con, cliente_id, cobranca):
    n = con.execute("DELETE FROM campanhas WHERE cliente_id = ? AND cobranca = ?",
                    (cliente_id, cobranca)).rowcount
    con.execute("DELETE FROM historico WHERE cliente_id = ? AND cobranca = ?", (cliente_id, cobranca))
    return n > 0
