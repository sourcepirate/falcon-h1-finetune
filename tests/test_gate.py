"""Generation gate: reject empty, short, and repetitive post-train output.

Torch-free so it runs in the CPU test job; the training hook is covered by
test_push.py (torch-gated).
"""

import pytest

from cyberft.errors import DegenerateOutputError
from cyberft.gate import (
    assert_generations,
    degenerate_reason,
    distinct_bigram_ratio,
    max_ngram_repeats,
)


def test_good_answer_passes():
    text = "Use tcpdump -n port 53 to capture DNS traffic on all interfaces."
    assert degenerate_reason(text) is None


def test_empty_and_short_rejected():
    assert degenerate_reason("") == "empty output"
    assert degenerate_reason("   \n ") == "empty output"
    assert "characters" in degenerate_reason("tcpdump")


def test_too_few_words_rejected():
    assert "words" in (degenerate_reason("ipconfig example.com") or "")


def test_markup_soup_rejected():
    text = "curl https/Example URLs | <file>.gz <txt><p> <html><body>"
    assert "markup" in (degenerate_reason(text) or "")


def test_short_valid_command_passes():
    assert degenerate_reason("Use ss -lntp.") is None


def test_looping_command_rejected():
    text = "tcp -d 1 -d 1 -d 1 -d 1 -d 1"
    reason = degenerate_reason(text)
    assert reason is not None
    assert "repetitive" in reason or "repeated" in reason


def test_repeated_ngram_rejected_even_when_long_enough():
    text = " ".join(["tcpdump -i 10.0.0.1 -nn"] * 4)
    reason = degenerate_reason(text)
    assert reason is not None
    assert "repetitive" in reason or "repeated" in reason


def test_long_varied_answer_not_rejected():
    text = (
        "SQL injection happens when user input is concatenated into a query. "
        "Use parameterised statements and validate every input on the server."
    )
    assert degenerate_reason(text) is None


def test_helpers():
    assert distinct_bigram_ratio("one two three four") == 1.0
    assert distinct_bigram_ratio("a a a a") < 0.5
    assert max_ngram_repeats("x x x x x", size=2) == 4


def test_assert_generations_names_the_bad_prompt():
    prompts = ["good prompt", "bad prompt"]
    answers = [
        "A perfectly reasonable and varied answer that keeps going for a while.",
        "",
    ]
    with pytest.raises(DegenerateOutputError, match="bad prompt"):
        assert_generations(prompts, answers)


def test_assert_generations_passes_when_all_good():
    assert_generations(
        ["a", "b"],
        [
            "Use tcpdump -n port 53 to capture DNS traffic.",
            "Open ports with ss -lntp or lsof -i -P -n.",
        ],
    )
