"""Every third-party import must be declared in the `train`/`colab` extras.

This is the check that would have caught `pip install -e .` producing a package that
imported fine but could not train (bitsandbytes missing).
"""

import ast
import re
import sys
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# import name -> distribution name, where they differ
DISTRIBUTIONS = {
    "yaml": "PyYAML",
    "huggingface_hub": "huggingface-hub",
}


def _distribution(spec: str) -> str:
    """'PyYAML>=6.0' -> 'pyyaml'; 'torch==2.11.0' -> 'torch'."""
    return re.split(r"[<>=!~;\s\[]", spec.strip())[0].lower()


def declared() -> set[str]:
    project = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    extras = project["project"]["optional-dependencies"]
    # `train` is the full runtime; `colab` is train-minus-torch for Colab T4.
    return {_distribution(spec) for extra in ("train", "colab") for spec in extras[extra]}


def third_party_imports() -> set[str]:
    names: set[str] = set()
    for path in (REPO / "src" / "cyberft").glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                names.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names.add(node.module.split(".")[0])
    stdlib = sys.stdlib_module_names
    return {DISTRIBUTIONS.get(name, name).lower() for name in names
            if name not in stdlib and name != "cyberft"}


def test_all_runtime_imports_are_declared():
    missing = third_party_imports() - declared()
    assert not missing, f"undeclared runtime dependencies: {sorted(missing)}"


def test_train_extra_is_not_empty():
    assert len(declared()) >= 8


def test_colab_extra_is_train_minus_torch():
    project = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    extras = project["project"]["optional-dependencies"]
    train = {_distribution(spec) for spec in extras["train"]}
    colab = {_distribution(spec) for spec in extras["colab"]}
    assert colab == train - {"torch"}, f"colab extra drifted: {sorted(train ^ (colab | {'torch'}))}"
