"""Normalisers are the only place dataset quirks are handled; keep them pinned by tests."""

import pytest

pytest.importorskip("datasets")

from cyberft.data import build_kimi, build_linuxcmd, build_redteam

MAX_CHARS = 4096
ROLES = ["system", "user", "assistant"]


def test_redteam_drops_echo_rows():
    row = {"instruction": "do a thing", "input": "same text", "output": "same text"}
    assert build_redteam(row, MAX_CHARS) is None


def test_redteam_drops_bare_json_dumps():
    row = {"instruction": "do a thing", "input": "", "output": '{"key": "value"}'}
    assert build_redteam(row, MAX_CHARS) is None


def test_redteam_keeps_a_normal_row():
    row = {"instruction": "How do I enumerate SMB?", "input": "", "output": "Use nmap --script smb-enum." * 2}
    messages = build_redteam(row, MAX_CHARS)
    assert [m["role"] for m in messages] == ROLES
    assert messages[1]["content"] == "How do I enumerate SMB?"


def test_kimi_rejects_conversations_without_assistant_turn():
    row = {"messages": [{"role": "user", "content": "hi"}, {"role": "user", "content": "hello"}]}
    assert build_kimi(row, MAX_CHARS) is None


def test_kimi_inserts_system_prompt_and_keeps_turns():
    row = {"messages": [
        {"role": "user", "content": "explain CVE-2024-1"},
        {"role": "assistant", "content": "It is a remote code execution flaw."},
    ]}
    messages = build_kimi(row, MAX_CHARS)
    assert [m["role"] for m in messages] == ROLES
    assert len(messages) == 3


def test_linuxcmd_turns_invocation_into_a_request():
    row = {"invocation": "list all files including hidden", "cmd": "ls -la"}
    messages = build_linuxcmd(row, MAX_CHARS)
    assert "list all files including hidden" in messages[1]["content"]
    assert messages[2]["content"] == "ls -la"


def test_linuxcmd_rejects_missing_fields():
    assert build_linuxcmd({"invocation": "", "cmd": "ls"}, MAX_CHARS) is None
    assert build_linuxcmd({"invocation": "list", "cmd": ""}, MAX_CHARS) is None
