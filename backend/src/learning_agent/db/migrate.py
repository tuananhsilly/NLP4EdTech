"""Apply the numbered SQL migrations in db/migrations/ (D2: raw SQL, no migration framework).

    uv run --env-file .env python -m learning_agent.db.migrate

Files are named NNNN_description.sql, numbered 0001, 0002, ... without gaps. Each pending file
runs in its own transaction together with its row in `schema_migrations`, so a failing file
leaves no trace. The table stores a checksum of every applied file: an applied migration must
never be edited (write a new one instead), and the runner refuses to continue if it was.
"""

import hashlib
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import LiteralString, cast

import psycopg
from psycopg.rows import TupleRow

from learning_agent.db import connect

MIGRATIONS_DIR = Path(__file__).parent / "migrations"
_FILE_NAME = re.compile(r"^(\d{4})_[a-z0-9_]+\.sql$")

_CREATE_TABLE = """
    CREATE TABLE IF NOT EXISTS schema_migrations (
        version    integer PRIMARY KEY,
        name       text NOT NULL,
        checksum   text NOT NULL,
        applied_at timestamptz NOT NULL DEFAULT now()
    )
"""


class MigrationError(RuntimeError):
    """The migration files or the recorded history are inconsistent."""


@dataclass(frozen=True)
class Migration:
    version: int
    name: str
    sql: str

    @property
    def checksum(self) -> str:
        return hashlib.sha256(self.sql.encode("utf-8")).hexdigest()


def discover(directory: Path = MIGRATIONS_DIR) -> list[Migration]:
    """Migration files in version order; names must match NNNN_*.sql and versions run 1..n."""
    found: list[Migration] = []
    for path in sorted(directory.glob("*.sql")):
        match = _FILE_NAME.match(path.name)
        if match is None:
            raise MigrationError(f"{path.name}: expected a name like 0001_description.sql")
        found.append(Migration(int(match[1]), path.name, path.read_text(encoding="utf-8")))
    versions = [m.version for m in found]
    if versions != list(range(1, len(found) + 1)):
        raise MigrationError(f"versions must be 1..{len(found)} without gaps, got {versions}")
    return found


def pending(available: list[Migration], applied: dict[int, str]) -> list[Migration]:
    """The migrations still to run, in version order.

    `available` comes from discover(); `applied` maps each recorded version to its checksum.
    Raises MigrationError if a recorded migration no longer exists as a file, if its file
    changed since it was applied, or if a migration older than an applied one is still pending
    (history out of order). All checks run before anything is returned.
    """
    by_version = {m.version: m for m in available}
    missing = sorted(v for v in applied if v not in by_version)
    if missing:
        raise MigrationError(f"applied migrations have no file any more: versions {missing}")
    changed = [
        m.name for m in available if m.version in applied and applied[m.version] != m.checksum
    ]
    if changed:
        raise MigrationError(f"applied migrations were edited: {changed}; write a new one instead")
    todo = [m for m in available if m.version not in applied]
    if todo and applied and todo[0].version < max(applied):
        raise MigrationError(f"{todo[0].name} is older than an applied migration (out of order)")
    return todo


def applied_checksums(conn: psycopg.Connection[TupleRow]) -> dict[int, str]:
    rows = conn.execute("SELECT version, checksum FROM schema_migrations").fetchall()
    return {int(version): str(checksum) for version, checksum in rows}


def migrate(
    conn: psycopg.Connection[TupleRow], directory: Path = MIGRATIONS_DIR
) -> list[Migration]:
    """Apply the pending migrations and return them (empty when the schema is up to date)."""
    conn.execute(_CREATE_TABLE)
    todo = pending(discover(directory), applied_checksums(conn))
    for m in todo:
        with conn.transaction():
            # The file comes from this repository, not from user input; psycopg's types accept
            # only literal SQL strings to keep runtime strings out of queries by default.
            conn.execute(cast(LiteralString, m.sql))
            conn.execute(
                "INSERT INTO schema_migrations (version, name, checksum) VALUES (%s, %s, %s)",
                (m.version, m.name, m.checksum),
            )
    return todo


def main() -> int:
    with connect() as conn:
        done = migrate(conn)
    if not done:
        print("schema is up to date")
    for m in done:
        print(f"applied {m.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
