"""Post-train generation gate.

A finite loss proves the adapter did not NaN; it does not prove the model still
answers. A tiny model trained at the wrong schedule can collapse into repeating
one malformed command while every loss stays finite (this actually shipped
once). The gate generates greedily on a few prompts and refuses empty, very
short, or repetitive output.

Torch-free: the checks are pure string/n-gram functions so they run in the CPU
test job.
"""

from __future__ import annotations

import itertools
import re
from collections.abc import Sequence

from .errors import DegenerateOutputError

MIN_CHARS = 8
MIN_WORDS = 3
MIN_WORDS_FOR_BIGRAMS = 8
MIN_DISTINCT_BIGRAMS = 0.35
MAX_MARKUP_TAGS = 2
# "tcp -d x -d x -d x" is only ~10 words; distinct-bigram alone can miss it,
# so also reject a 4-gram repeated three or more times.
MAX_REPEATED_NGRAM = 3
NGRAM_SIZE = 4

_WORD = re.compile(r"\S+")
_MARKUP = re.compile(r"</?[a-zA-Z][^>\s]*>")


def words(text: str) -> list[str]:
    return [w.lower() for w in _WORD.findall(text)]


def distinct_bigram_ratio(text: str) -> float:
    """Unique adjacent word pairs / total pairs. 0.0 when there are no pairs."""
    tokens = words(text)
    if len(tokens) < 2:
        return 0.0
    bigrams = list(itertools.pairwise(tokens))
    return len(set(bigrams)) / len(bigrams)


def max_ngram_repeats(text: str, size: int = NGRAM_SIZE) -> int:
    tokens = words(text)
    if len(tokens) < size:
        return 1
    counts: dict[tuple[str, ...], int] = {}
    for i in range(len(tokens) - size + 1):
        gram = tuple(tokens[i : i + size])
        counts[gram] = counts.get(gram, 0) + 1
    return max(counts.values())


def degenerate_reason(
    text: str,
    *,
    min_chars: int = MIN_CHARS,
    min_words: int = MIN_WORDS,
    min_words_for_bigrams: int = MIN_WORDS_FOR_BIGRAMS,
    min_distinct_bigrams: float = MIN_DISTINCT_BIGRAMS,
    max_markup_tags: int = MAX_MARKUP_TAGS,
    max_repeats: int = MAX_REPEATED_NGRAM,
) -> str | None:
    """Return why ``text`` is unusable, or None when it looks like an answer."""
    stripped = text.strip()
    if not stripped:
        return "empty output"
    if len(stripped) < min_chars:
        return f"only {len(stripped)} characters"
    tokens = words(stripped)
    if len(tokens) < min_words:
        return f"only {len(tokens)} words"
    markup = len(_MARKUP.findall(stripped))
    if markup >= max_markup_tags:
        return f"markup soup ({markup} tags)"
    if len(tokens) >= min_words_for_bigrams and distinct_bigram_ratio(stripped) < min_distinct_bigrams:
        return f"distinct-bigram ratio {distinct_bigram_ratio(stripped):.2f} (repetitive)"
    repeats = max_ngram_repeats(stripped)
    if repeats >= max_repeats:
        return f"{NGRAM_SIZE}-gram repeated {repeats}x"
    return None


def assert_generations(prompts: Sequence[str], answers: Sequence[str]) -> None:
    """Raise :class:`DegenerateOutputError` naming every bad prompt."""
    if len(prompts) != len(answers):
        raise ValueError("prompts and answers must have the same length")
    problems = []
    for prompt, answer in zip(prompts, answers, strict=True):
        reason = degenerate_reason(answer)
        if reason is not None:
            excerpt = answer.strip().replace("\n", " ")[:80]
            problems.append(f"{prompt!r}: {reason} ({excerpt!r})")
    if problems:
        raise DegenerateOutputError(
            "post-train generation gate failed, refusing to publish a collapsed adapter:\n  "
            + "\n  ".join(problems)
        )
