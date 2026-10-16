"""Enforce one-way dependencies between packages (07 §5.2, invariant principle 1 in CLAUDE.md).

Imports are read with `ast`, so source files are inspected without being executed.
"""

import ast
from pathlib import Path

import pytest

ROOT = "learning_agent"
SRC_PACKAGE_DIR = Path(__file__).resolve().parents[1] / "src" / ROOT

# 07 §5.2 rule 1: deterministic core must not import `llm`. `spec.generator` is the exception.
NO_LLM_PACKAGES = {"domain", "mathx", "spec", "grading", "learner", "policy", "itembank"}
NO_LLM_EXCEPTIONS = {"spec.generator"}


def module_name(path: Path, package_dir: Path) -> str:
    parts = list(path.relative_to(package_dir.parent).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def imported_modules(path: Path, package_dir: Path) -> set[str]:
    """Absolute names of all modules imported by `path`, with relative imports resolved."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    parts = module_name(path, package_dir).split(".")
    package = parts if path.name == "__init__.py" else parts[:-1]
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package[: len(package) - (node.level - 1)]
                prefix = ".".join(base + ([node.module] if node.module else []))
            else:
                prefix = node.module or ""
            # `from learning_agent import llm` imports the module learning_agent.llm
            found.add(prefix)
            found.update(f"{prefix}.{alias.name}" for alias in node.names)
    return found


def _within(name: str, target: str) -> bool:
    return name == target or name.startswith(target + ".")


def find_violations(package_dir: Path) -> list[str]:
    violations: list[str] = []
    for path in sorted(package_dir.rglob("*.py")):
        name = module_name(path, package_dir)
        sub = name.removeprefix(ROOT + ".")
        top = sub.split(".")[0]
        imports = imported_modules(path, package_dir)

        def imports_any(target: str, imports: set[str] = imports) -> bool:
            return any(_within(m, target) for m in imports)

        exempt = any(_within(sub, e) for e in NO_LLM_EXCEPTIONS)
        if top in NO_LLM_PACKAGES and not exempt and imports_any(f"{ROOT}.llm"):
            violations.append(f"{name} imports {ROOT}.llm")
        # 07 §5.2 rule 2: no package may import `api`.
        if top != "api" and imports_any(f"{ROOT}.api"):
            violations.append(f"{name} imports {ROOT}.api")
    return violations


def make_package(tmp_path: Path, files: dict[str, str]) -> Path:
    package_dir = tmp_path / ROOT
    for rel, source in {"__init__.py": "", **files}.items():
        path = package_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
    return package_dir


def test_scans_real_source_tree():
    assert (SRC_PACKAGE_DIR / "__init__.py").is_file()


def test_real_source_tree_has_no_violations():
    assert find_violations(SRC_PACKAGE_DIR) == []


@pytest.mark.parametrize(
    "import_line",
    [
        "from ..llm import client",
        "from learning_agent.llm.client import LLMClient",
        "import learning_agent.llm",
        "from learning_agent import llm",
    ],
)
def test_detects_llm_import_in_deterministic_core(tmp_path: Path, import_line: str):
    package_dir = make_package(
        tmp_path,
        {
            "llm/__init__.py": "",
            "grading/__init__.py": "",
            "grading/grader.py": import_line + "\n",
        },
    )
    assert find_violations(package_dir) == [
        "learning_agent.grading.grader imports learning_agent.llm"
    ]


def test_spec_generator_may_import_llm(tmp_path: Path):
    package_dir = make_package(
        tmp_path,
        {
            "llm/__init__.py": "",
            "spec/__init__.py": "",
            "spec/generator.py": "from ..llm import client\n",
        },
    )
    assert find_violations(package_dir) == []


def test_detects_api_import(tmp_path: Path):
    package_dir = make_package(
        tmp_path,
        {
            "api/__init__.py": "",
            "orchestrator/__init__.py": "from learning_agent.api import app\n",
        },
    )
    assert find_violations(package_dir) == [
        "learning_agent.orchestrator imports learning_agent.api"
    ]
