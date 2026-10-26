"""The CLI surface must stay minimal: --config everywhere, --smoke/--force only."""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_cli_surface_is_minimal():

    sys.path.insert(0, str(REPO / "src"))
    from cyberft.args import build_parser

    actions = {}
    for action in build_parser()._actions:
        if action.dest == "command":
            for name, sub in action.choices.items():
                actions[name] = sorted(a.dest for a in sub._actions if a.dest != "help")
    assert actions == {
        "data": ["config", "force", "smoke"],
        "train": ["config", "smoke"],
        "eval": ["config", "smoke"],
        "export": ["config"],
        "info": ["config"],
    }
