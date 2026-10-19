"""Make backend/.env visible to tests (DATABASE_URL, HF_HOME), like `uv run --env-file .env`.

Variables already set in the environment win. Lines are KEY=VALUE; blank lines and lines starting
with "#" are ignored.
"""

import os
from pathlib import Path

ENV_FILE = Path(__file__).resolve().parents[1] / ".env"

if ENV_FILE.exists():
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        key, sep, value = line.partition("=")
        if sep and key.strip() and not key.lstrip().startswith("#"):
            os.environ.setdefault(key.strip(), value.strip())
