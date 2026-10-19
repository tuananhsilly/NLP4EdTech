from pathlib import Path

import pytest

from learning_agent.db.migrate import MIGRATIONS_DIR, Migration, MigrationError, discover, pending

A = Migration(1, "0001_a.sql", "CREATE TABLE a ();")
B = Migration(2, "0002_b.sql", "CREATE TABLE b ();")


def write(directory: Path, *names: str) -> Path:
    for n in names:
        (directory / n).write_text(f"-- {n}\n", encoding="utf-8")
    return directory


# --- discover ----------------------------------------------------------------------------


def test_discover_returns_files_in_version_order(tmp_path: Path) -> None:
    found = discover(write(tmp_path, "0002_b.sql", "0001_a.sql"))
    assert [(m.version, m.name) for m in found] == [(1, "0001_a.sql"), (2, "0002_b.sql")]


@pytest.mark.parametrize("bad", ["1_a.sql", "0001-a.sql", "0001_A.sql", "0001_a.SQL.sql"])
def test_discover_rejects_badly_named_files(tmp_path: Path, bad: str) -> None:
    with pytest.raises(MigrationError, match="0001_description"):
        discover(write(tmp_path, bad))


@pytest.mark.parametrize("names", [("0001_a.sql", "0003_c.sql"), ("0002_b.sql",)])
def test_discover_rejects_gaps(tmp_path: Path, names: tuple[str, ...]) -> None:
    with pytest.raises(MigrationError, match="without gaps"):
        discover(write(tmp_path, *names))


def test_checksum_changes_when_the_file_changes() -> None:
    assert A.checksum != Migration(1, "0001_a.sql", A.sql + " ").checksum


def test_repository_migrations_are_valid() -> None:
    assert [m.name for m in discover(MIGRATIONS_DIR)][:1] == ["0001_kb.sql"]


# --- pending -----------------------------------------------------------------------------


def test_nothing_applied_returns_all_in_order() -> None:
    assert pending([A, B], {}) == [A, B]


def test_applied_migrations_are_skipped() -> None:
    assert pending([A, B], {1: A.checksum}) == [B]


def test_up_to_date_returns_nothing() -> None:
    assert pending([A, B], {1: A.checksum, 2: B.checksum}) == []


def test_edited_applied_migration_raises() -> None:
    with pytest.raises(MigrationError, match="0001_a.sql"):
        pending([A, B], {1: "0" * 64})


def test_pending_migration_older_than_an_applied_one_raises() -> None:
    with pytest.raises(MigrationError, match="0001_a.sql.*out of order"):
        pending([A, B], {2: B.checksum})


def test_recorded_migration_without_file_raises() -> None:
    with pytest.raises(MigrationError, match="2"):
        pending([A], {1: A.checksum, 2: B.checksum})
