"""Loaders are the only place dataset quirks are handled; keep them pinned by tests."""

import pytest

pytest.importorskip("datasets")
pytest.importorskip("torch")  # loaders imports model.py, which needs the training stack

from cyberft.config import SourceConfig
from cyberft.loaders import KimiLoader, LinuxCommandsLoader, RedTeamLoader, get_loader

MAX_CHARS = 4096
SEED = 42
ROLES = ["system", "user", "assistant"]


def _loader(cls, name="x"):
    return cls(SourceConfig(name=name, repo="org/dataset"), max_chars=MAX_CHARS, seed=SEED)


def test_redteam_drops_echo_rows():
    loader = _loader(RedTeamLoader, "redteam")
    row = {"instruction": "do a thing", "input": "same text", "output": "same text"}
    assert loader.map_row(row) is None


def test_redteam_drops_bare_json_dumps():
    loader = _loader(RedTeamLoader, "redteam")
    row = {"instruction": "do a thing", "input": "", "output": '{"key": "value"}'}
    assert loader.map_row(row) is None


def test_redteam_keeps_a_normal_row():
    loader = _loader(RedTeamLoader, "redteam")
    row = {"instruction": "How do I enumerate SMB?", "input": "", "output": "Use nmap --script smb-enum." * 2}
    messages = loader.map_row(row)
    assert [m["role"] for m in messages] == ROLES
    assert messages[1]["content"] == "How do I enumerate SMB?"


def test_kimi_rejects_conversations_without_assistant_turn():
    loader = _loader(KimiLoader, "kimi")
    row = {"messages": [{"role": "user", "content": "hi"}, {"role": "user", "content": "hello"}]}
    assert loader.map_row(row) is None


def test_kimi_inserts_system_prompt_and_keeps_turns():
    loader = _loader(KimiLoader, "kimi")
    row = {"messages": [
        {"role": "user", "content": "explain CVE-2024-1"},
        {"role": "assistant", "content": "It is a remote code execution flaw."},
    ]}
    messages = loader.map_row(row)
    assert [m["role"] for m in messages] == ROLES
    assert len(messages) == 3


def test_linuxcmd_turns_invocation_into_a_request():
    loader = _loader(LinuxCommandsLoader, "linuxcmd")
    row = {"invocation": "list all files including hidden", "cmd": "ls -la"}
    messages = loader.map_row(row)
    assert "list all files including hidden" in messages[1]["content"]
    assert messages[2]["content"] == "ls -la"


def test_linuxcmd_rejects_missing_fields():
    loader = _loader(LinuxCommandsLoader, "linuxcmd")
    assert loader.map_row({"invocation": "", "cmd": "ls"}) is None
    assert loader.map_row({"invocation": "list", "cmd": ""}) is None


def test_registry_resolves_every_known_source():
    for name in ("redteam", "kali", "kimi", "cti", "exploitdb", "linuxcmd", "debian"):
        assert name == get_loader(name).DATASET_NAME


def test_loader_rejects_mismatched_spec():
    with pytest.raises(Exception, match="handles source"):
        RedTeamLoader(SourceConfig(name="kali", repo="org/dataset"),
                      max_chars=MAX_CHARS, seed=SEED)
