"""Guard against accidental edits to the copied competition scorer."""

from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_NAMES = {"HostVisibleError", "single_lab_f1", "mouse_fbeta", "score"}


def _official_notebook_tree() -> ast.Module:
    notebook = json.loads((ROOT / "mabe-f-beta.ipynb").read_text(encoding="utf-8"))
    source = "".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )
    return ast.parse(source)


def _definitions(tree: ast.Module) -> dict[str, ast.AST]:
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in OFFICIAL_NAMES
    }


def test_production_metric_definitions_match_notebook_ast_exactly() -> None:
    official = _definitions(_official_notebook_tree())
    production = _definitions(
        ast.parse((ROOT / "src" / "mabe" / "metric.py").read_text(encoding="utf-8"))
    )

    assert official.keys() == OFFICIAL_NAMES
    assert production.keys() == OFFICIAL_NAMES
    for name in OFFICIAL_NAMES:
        assert ast.dump(production[name], include_attributes=False) == ast.dump(
            official[name], include_attributes=False
        ), f"Production definition {name!r} diverged from mabe-f-beta.ipynb"
