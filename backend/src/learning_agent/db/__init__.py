"""Postgres access: connection and schema migrations (D2: psycopg 3 + raw SQL)."""

import os

import psycopg
from psycopg.rows import TupleRow


def database_url() -> str:
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError(
            "DATABASE_URL is not set; run with `uv run --env-file .env ...` "
            "(see backend/.env.example)"
        )
    return url


def connect(url: str | None = None) -> psycopg.Connection[TupleRow]:
    """A connection in autocommit mode: group statements with `with conn.transaction():`."""
    return psycopg.connect(url or database_url(), autocommit=True)
