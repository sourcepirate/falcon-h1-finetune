"""Build the seven-dataset corpus once, deterministically, and cache it to disk.

Where data comes from is configuration (``data.sources``); how a row becomes a chat
message is code (``BUILDERS``). The cache is fingerprinted, so a config change invalidates
it instead of silently training on the previous corpus.
"""

from __future__ import annotations

import array
import hashlib
import json
import logging
import shutil
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset, load_from_disk
from huggingface_hub import HfApi, hf_hub_download

from .config import Config, SourceConfig
from .errors import ConfigError, NotBuiltError

log = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a cybersecurity assistant. You help with defensive security, authorized "
    "penetration testing, threat intelligence, Linux administration and vulnerability "
    "analysis. Be precise, technical and practical, and include commands or code when useful. "
    "Only help with authorized testing and educational purposes."
)
MIN_ASSISTANT_CHARS = 20


def _clean(value: object, max_chars: int) -> str:
    if value is None:
        return ""
    text = value if isinstance(value, str) else str(value)
    text = text.strip()
    return text[:max_chars] if len(text) > max_chars else text


def _messages(user: str, assistant: str, max_chars: int, *,
              min_chars: int = MIN_ASSISTANT_CHARS) -> list[dict] | None:
    user, assistant = _clean(user, max_chars), _clean(assistant, max_chars)
    if len(user) < 3 or len(assistant) < min_chars:
        return None
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]


# ---------------------------------------------------------------- normalisers
def build_redteam(ex: dict, max_chars: int) -> list[dict] | None:
    instruction = _clean(ex.get("instruction"), max_chars)
    extra = _clean(ex.get("input"), max_chars)
    out = _clean(ex.get("output"), max_chars)
    if not out or (extra and extra == out):           # echo rows
        return None
    if out[:1] in "[{" and out[-1:] in "]}":          # bare JSON dumps
        return None
    user = f"{instruction}\n\n{extra}".strip() if extra else instruction
    return _messages(user, out, max_chars)


def build_kali(ex: dict, max_chars: int) -> list[dict] | None:
    return _messages(ex.get("instruction", ""), ex.get("response", ""), max_chars)


def build_kimi(ex: dict, max_chars: int) -> list[dict] | None:
    raw = ex.get("messages")
    if not isinstance(raw, list) or len(raw) < 2:
        return None
    clean: list[dict] = []
    for message in raw:
        role = message.get("role")
        content = _clean(message.get("content"), max_chars)
        if role not in ("system", "user", "assistant") or not content:
            return None
        clean.append({"role": role, "content": content})
    if clean[-1]["role"] != "assistant" or len(clean[-1]["content"]) < MIN_ASSISTANT_CHARS:
        return None
    if clean[0]["role"] != "system":
        clean.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
    return clean


def build_cti(ex: dict, max_chars: int) -> list[dict] | None:
    instruction = _clean(ex.get("instruction"), max_chars)
    extra = _clean(ex.get("input"), max_chars)
    if not instruction:
        return None
    user = f"{instruction}\n\nContext:\n{extra}" if extra else instruction
    return _messages(user, ex.get("output", ""), max_chars)


def build_exploitdb(ex: dict, max_chars: int) -> list[dict] | None:
    return _messages(ex.get("input", ""), ex.get("output", ""), max_chars)


def build_linuxcmd(ex: dict, max_chars: int) -> list[dict] | None:
    invocation = _clean(ex.get("invocation"), max_chars)
    command = _clean(ex.get("cmd"), max_chars)
    if not invocation or not command:
        return None
    # Commands are legitimately short ("ls -la"), so do not apply the prose minimum.
    return _messages(f"Write the Linux shell command for this task:\n{invocation}", command,
                     max_chars, min_chars=1)


def build_debian(ex: dict, max_chars: int) -> list[dict] | None:
    return _messages(ex.get("input", ""), ex.get("output", ""), max_chars)


BUILDERS: dict[str, Callable[[dict, int], list[dict] | None]] = {
    "redteam": build_redteam,
    "kali": build_kali,
    "kimi": build_kimi,
    "cti": build_cti,
    "exploitdb": build_exploitdb,
    "linuxcmd": build_linuxcmd,
    "debian": build_debian,
}


# ---------------------------------------------------------------- loading
def _load_source(spec: SourceConfig) -> Dataset:
    if spec.parquet:
        local = hf_hub_download(
            repo_id=spec.repo, filename=spec.parquet, repo_type="dataset",
            revision=spec.revision or "main",
        )
        return load_dataset("parquet", data_files=local, split="train")

    parts = []
    for split in spec.splits:
        kwargs: dict = {"path": spec.repo, "split": split}
        if spec.subset:
            kwargs["name"] = spec.subset
        if spec.revision:
            kwargs["revision"] = spec.revision
        parts.append(load_dataset(**kwargs))
    return parts[0] if len(parts) == 1 else concatenate_datasets(parts)


def _normalise(dataset: Dataset, spec: SourceConfig, max_chars: int, seed: int) -> Dataset:
    total = len(dataset)
    if spec.cap and total > spec.cap:
        dataset = dataset.shuffle(seed=seed).select(range(spec.cap))

    builder = BUILDERS[spec.name]
    rows = [msgs for ex in dataset if (msgs := builder(ex, max_chars)) is not None]
    log.info("%-10s loaded %7d  kept %7d", spec.name, total, len(rows))
    return Dataset.from_list([{"messages": m} for m in rows])


def build_messages(cfg: Config) -> Dataset:
    configured = {source.name for source in cfg.data.sources}
    if unknown := configured - set(BUILDERS):
        raise ConfigError(f"data.sources has no normaliser for: {sorted(unknown)}")
    if missing := set(BUILDERS) - configured:
        raise ConfigError(f"data.sources is missing dataset(s): {sorted(missing)}")

    max_chars = cfg.data.max_len * 4          # cheap guard; one source row is ~9.7M chars
    parts = [
        _normalise(_load_source(source), source, max_chars, cfg.data.seed)
        for source in cfg.data.sources
    ]
    return concatenate_datasets(parts).shuffle(seed=cfg.data.seed)


# ---------------------------------------------------------------- rendering and split
def render_and_split(cfg: Config, tokenizer, dataset: Dataset, *, eval_size: int) -> DatasetDict:
    def render(ex: dict) -> dict:
        text = tokenizer.apply_chat_template(ex["messages"], tokenize=False, add_generation_prompt=False)
        if not text.endswith(tokenizer.eos_token):
            text += tokenizer.eos_token
        return {"text": text}

    dataset = dataset.map(render, remove_columns=["messages"], desc="chat template")

    seen: set[str] = set()
    unique: list[int] = []
    for i, text in enumerate(dataset["text"]):
        if text not in seen:
            seen.add(text)
            unique.append(i)
    dataset = dataset.select(unique)

    # Token-accurate length filter: drop, never truncate, so answers stay whole.
    texts = list(dataset["text"])
    keep_idx: list[int] = []
    keep_len: list[int] = []
    for start in range(0, len(texts), 1000):
        batch = texts[start:start + 1000]
        for offset, ids in enumerate(tokenizer(batch, add_special_tokens=False)["input_ids"]):
            if len(ids) <= cfg.data.max_len:
                keep_idx.append(start + offset)
                keep_len.append(len(ids))
    dataset = dataset.select(keep_idx)
    log.info("deduped %d | length-filtered %d", len(unique), len(dataset))

    # Eval is drawn only from rows that fit eval_max_len: evaluation runs without loss
    # scaling, and one overflowing long sample poisons the whole validation number.
    eval_pos = [i for i, n in enumerate(keep_len) if n <= cfg.data.eval_max_len][:eval_size]
    eval_set = set(eval_pos)
    eval_ds = dataset.select(eval_pos)
    train_ds = dataset.select([i for i in range(len(dataset)) if i not in eval_set])

    def tokenize(ex: dict) -> dict:
        return {"input_ids": tokenizer(ex["text"], add_special_tokens=False)["input_ids"]}

    return DatasetDict({
        "train": train_ds.map(tokenize, desc="tokenize train"),
        "eval": eval_ds.map(tokenize, desc="tokenize eval"),
    })


def build(cfg: Config, tokenizer, *, smoke: bool = False) -> DatasetDict:
    dataset = build_messages(cfg)
    if smoke:
        dataset = dataset.select(range(min(400, len(dataset))))
    eval_size = min(cfg.data.eval_size, 32) if smoke else cfg.data.eval_size
    splits = render_and_split(cfg, tokenizer, dataset, eval_size=eval_size)
    log.info("train %d | eval %d", len(splits["train"]), len(splits["eval"]))
    return splits


# ---------------------------------------------------------------- identity and cache
def fingerprint(cfg: Config, *, smoke: bool) -> str:
    """Everything that changes corpus content. Used to invalidate the cache."""
    payload = {
        "sources": [asdict(source) for source in cfg.data.sources],
        "max_len": cfg.data.max_len,
        "eval_max_len": cfg.data.eval_max_len,
        "eval_size": cfg.data.eval_size,
        "seed": cfg.data.seed,
        "smoke": smoke,
        "model_id": cfg.model.id,
        "system_prompt": SYSTEM_PROMPT,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def content_hash(splits: DatasetDict) -> str:
    """Hash the actual token ids, so two corpora with equal lengths are still distinct."""
    digest = hashlib.sha256()
    for split_name in sorted(splits):
        for ids in splits[split_name]["input_ids"]:
            digest.update(array.array("i", ids).tobytes())
    return digest.hexdigest()


def _resolved_revisions(cfg: Config) -> dict[str, str]:
    api = HfApi()
    resolved: dict[str, str] = {}
    for source in cfg.data.sources:
        try:
            resolved[source.name] = api.dataset_info(source.repo).sha or "unknown"
        except Exception as exc:                     # offline or rate-limited: still build
            resolved[source.name] = f"unresolved ({type(exc).__name__})"
    return resolved


def manifest(cfg: Config, splits: DatasetDict, *, smoke: bool) -> dict:
    lengths = sorted(len(x) for x in splits["train"]["input_ids"]) or [0]

    def percentile(q: float) -> int:
        return lengths[min(len(lengths) - 1, int(len(lengths) * q))]

    return {
        "fingerprint": fingerprint(cfg, smoke=smoke),
        "content_sha256": content_hash(splits),
        "model_id": cfg.model.id,
        "sources": {source.name: source.repo for source in cfg.data.sources},
        "revisions": _resolved_revisions(cfg),
        "max_len": cfg.data.max_len,
        "eval_max_len": cfg.data.eval_max_len,
        "rows": {split: len(splits[split]) for split in splits},
        "tokens": {"p50": percentile(0.50), "p90": percentile(0.90), "max": lengths[-1]},
    }


def prepare_data(cfg: Config, *, smoke: bool = False, force: bool = False,
                 tokenizer=None) -> DatasetDict:
    """Return the tokenised corpus, building it on first use.

    Cached under ``data.output_dir/{full,smoke}``. ``force=True`` rebuilds from source;
    a config change makes the cache invalid, and the caller is told to rebuild rather
    than silently training on the previous corpus.
    """
    out = Path(cfg.data.output_dir) / ("smoke" if smoke else "full")
    expected = fingerprint(cfg, smoke=smoke)

    if not force and (out / "dataset_dict.json").exists():
        try:
            stored = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            stored = {}
        if stored.get("fingerprint") != expected:
            raise NotBuiltError(
                f"{out} was built with a different configuration "
                f"(fingerprint {stored.get('fingerprint')} != {expected}). "
                "Call prepare_data(cfg, force=True) to rebuild."
            )
        return load_from_disk(str(out))

    from .model import build_tokenizer  # local import keeps import-time deps light

    tokenizer = tokenizer or build_tokenizer(cfg)
    splits = build(cfg, tokenizer, smoke=smoke)
    shutil.rmtree(out, ignore_errors=True)     # save_to_disk does not clear stale shards
    splits.save_to_disk(str(out))
    (out / "manifest.json").write_text(
        json.dumps(manifest(cfg, splits, smoke=smoke), indent=2), encoding="utf-8"
    )
    return splits


def data_manifest(cfg: Config, *, smoke: bool = False) -> dict:
    path = Path(cfg.data.output_dir) / ("smoke" if smoke else "full") / "manifest.json"
    if not path.exists():
        raise NotBuiltError(f"{path} not found - call prepare_data() first")
    return json.loads(path.read_text(encoding="utf-8"))
