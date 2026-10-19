"""Postgres fixture: each test module gets a fresh database, dropped afterwards.

DATABASE_URL comes from the environment (tests/conftest.py also reads backend/.env). Tests that
use the fixture are skipped when it is missing or the server is down (`docker compose up -d`).
"""

import os
import uuid
from collections.abc import Iterator

import psycopg
import pytest
from psycopg import sql
from psycopg.conninfo import make_conninfo


@pytest.fixture(scope="module")
def database_url() -> Iterator[str]:
    """Connection string of a throwaway database (the server's own database is never touched)."""
    url = os.environ.get("DATABASE_URL")
    if url is None:
        pytest.skip("DATABASE_URL is not set (see backend/.env.example)")
    try:
        admin = psycopg.connect(url, autocommit=True, connect_timeout=3)
    except psycopg.OperationalError as e:
        pytest.skip(f"Postgres is not reachable ({e}); run `docker compose up -d`")
    name = f"test_{uuid.uuid4().hex[:12]}"
    with admin:
        admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
        try:
            yield make_conninfo(url, dbname=name)
        finally:
            admin.execute(sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(name)))
