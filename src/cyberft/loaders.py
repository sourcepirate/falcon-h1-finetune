"""Corpus loaders: one class per dataset, sharing a common base.

Where data comes from is configuration (``data.sources``); how a row becomes a chat
message is code (one :class:`LoaderBase` subclass per source, registered in
``LOADERS``). The built corpus is cached to disk fingerprinted, so a config change
invalidates it instead of silently training on the previous corpus.
"""

from __future__ import annotations

import array
import hashlib
import json
import logging
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset, load_from_disk
from huggingface_hub import HfApi, hf_hub_download

from .config import Config, SourceConfig
from .errors import ConfigError, NotBuiltError
from .model import build_tokenizer

log = logging.getLogger(__name__)

# Bump whenever a loader's map_row or SYSTEM_PROMPT logic changes the shape
# of the corpus. fingerprint() hashes this so editing loaders.py invalidates the
# Arrow cache instead of silently training on stale rows.
LOADERS_VERSION = 1

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


class LoaderBase:
    """Base class for all loaders. Subclasses declare their source and map rows."""

    DATASET_NAME: str | None = None  # key into LOADERS; must match SourceConfig.name

    def __init__(self, spec: SourceConfig, *, max_chars: int, seed: int):
        if self.DATASET_NAME is None:
            raise TypeError(f"{type(self).__name__} must set DATASET_NAME")
        if spec.name != self.DATASET_NAME:
            raise ConfigError(
                f"{type(self).__name__} handles source {self.DATASET_NAME!r}, "
                f"got SourceConfig {spec.name!r}"
            )
        self.spec = spec
        self.max_chars = max_chars
        self.seed = seed

    def map_row(self, row: dict[str, Any]) -> list[dict] | None:
        """Map one raw row to chat messages, or None to drop the row."""
        raise NotImplementedError

    def load(self) -> Dataset:
        """Load the source, cap it, and return the kept rows as a messages Dataset."""
        raw = self._load_ds()
        total = len(raw)
        if self.spec.cap and total > self.spec.cap:
            raw = raw.shuffle(seed=self.seed).select(range(self.spec.cap))
        rows = [msgs for ex in raw if (msgs := self.map_row(ex)) is not None]
        log.info("%-10s loaded %7d  kept %7d", self.DATASET_NAME, total, len(rows))
        return Dataset.from_list([{"messages": m} for m in rows])

    def _load_ds(self) -> Dataset:
        """Fetch the raw source from the Hub (parquet file or repo splits)."""
        spec = self.spec
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


class RedTeamLoader(LoaderBase):
    """WNT3D/Ultimate-Offensive-Red-Team: instruction/input/output red-team rows."""

    DATASET_NAME = "redteam"

    def map_row(self, row: dict[str, Any]) -> list[dict] | None:
        instruction = _clean(row.get("instruction"), self.max_chars)
        extra = _clean(row.get("input"), self.max_chars)
        out = _clean(row.get("output"), self.max_chars)
        if not out or (extra and extra == out):           # echo rows
            return None
        if out[:1] in "[{" and out[-1:] in "]}":          # bare JSON dumps
            return None
        user = f"{instruction}\n\n{extra}".strip() if extra else instruction
        return _messages(user, out, self.max_chars)


class KaliLoader(LoaderBase):
    """suryanshp1/kali-linux-pentesting-data: instruction/response Kali how-tos."""

    DATASET_NAME = "kali"

    def map_row(self, row: dict[str, Any]) -> list[dict] | None:
        return _messages(row.get("instruction", ""), row.get("response", ""), self.max_chars)


class KimiLoader(LoaderBase):
    """echel0nn1881/kimi-cyber-reasoning: pre-built multi-turn conversations."""

    DATASET_NAME = "kimi"

    def map_row(self, row: dict[str, Any]) -> list[dict] | None:
        raw = row.get("messages")
        if not isinstance(raw, list) or len(raw) < 2:
            return None
        clean: list[dict] = []
        for message in raw:
            role = message.get("role")
            content = _clean(message.get("content"), self.max_chars)
            if role not in ("system", "user", "assistant") or not content:
                return None
            clean.append({"role": role, "content": content})
        if clean[-1]["role"] != "assistant" or len(clean[-1]["content"]) < MIN_ASSISTANT_CHARS:
            return None
        if clean[0]["role"] != "system":
            clean.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
        return clean


class CTILoader(LoaderBase):
    """reloading0101/threat-intelligence-dataset: instruction/input/output CTI rows."""

    DATASET_NAME = "cti"

    def map_row(self, row: dict[str, Any]) -> list[dict] | None:
        instruction = _clean(row.get("instruction"), self.max_chars)
        extra = _clean(row.get("input"), self.max_chars)
        if not instruction:
            return None
        user = f"{instruction}\n\nContext:\n{extra}" if extra else instruction
        return _messages(user, row.get("output", ""), self.max_chars)


class ExploitDBLoader(LoaderBase):
    """Waiper/ExploitDB_DataSet: input/output exploit descriptions."""

    DATASET_NAME = "exploitdb"

    def map_row(self, row: dict[str, Any]) -> list[dict] | None:
        return _messages(row.get("input", ""), row.get("output", ""), self.max_chars)


class LinuxCommandsLoader(LoaderBase):
    """Romit2004/LinuxCommands: invocation/cmd pairs (commands may be short)."""

    DATASET_NAME = "linuxcmd"

    def map_row(self, row: dict[str, Any]) -> list[dict] | None:
        invocation = _clean(row.get("invocation"), self.max_chars)
        command = _clean(row.get("cmd"), self.max_chars)
        if not invocation or not command:
            return None
        # Commands are legitimately short ("ls -la"), so do not apply the prose minimum.
        return _messages(f"Write the Linux shell command for this task:\n{invocation}", command,
                         self.max_chars, min_chars=1)


class DebianLoader(LoaderBase):
    """2etatg/Debian_Hacking_Networking: input/output Debian how-tos."""

    DATASET_NAME = "debian"

    def map_row(self, row: dict[str, Any]) -> list[dict] | None:
        return _messages(row.get("input", ""), row.get("output", ""), self.max_chars)


LOADERS: dict[str, type[LoaderBase]] = {
    "redteam": RedTeamLoader,
    "kali": KaliLoader,
    "kimi": KimiLoader,
    "cti": CTILoader,
    "exploitdb": ExploitDBLoader,
    "linuxcmd": LinuxCommandsLoader,
    "debian": DebianLoader,
}


def get_loader(name: str) -> type[LoaderBase]:
    """Return the loader class for a configured source name."""
    try:
        return LOADERS[name]
    except KeyError:
        raise ConfigError(
            f"data.sources has no loader for: {name!r} (known: {sorted(LOADERS)})"
        ) from None


def build_messages(cfg: Config) -> Dataset:
    configured = {source.name for source in cfg.data.sources}
    if unknown := configured - set(LOADERS):
        raise ConfigError(f"data.sources has no loader for: {sorted(unknown)}")
    if missing := set(LOADERS) - configured:
        raise ConfigError(f"data.sources is missing dataset(s): {sorted(missing)}")

    max_chars = cfg.data.max_len * 4          # cheap guard; one source row is ~9.7M chars
    parts = [
        get_loader(source.name)(source, max_chars=max_chars, seed=cfg.data.seed).load()
        for source in cfg.data.sources
    ]
    return concatenate_datasets(parts).shuffle(seed=cfg.data.seed)


# ---------------------------------------------------------------- rendering and split
class RenderSplitPipeline:
    """Render → dedupe → length-filter → split → tokenize, as an iterable of stages.

    Each ``next()`` call advances exactly one stage and yields its name;
    :meth:`run` drains the iterator and returns the final ``DatasetDict``.
    ``render_row`` and ``tokenize_row`` are plain methods rather than closures,
    so every stage stays independently testable.
    """

    STAGES = ("render", "dedupe", "filter", "split", "tokenize")

    def __init__(self, cfg: Config, tokenizer, dataset: Dataset, *, eval_size: int) -> None:
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.eval_size = eval_size
        self._stage_index = 0
        self.lengths: list[int] = []
        self._train: Dataset | None = None
        self._eval: Dataset | None = None
        self.splits: DatasetDict | None = None

    def __iter__(self) -> RenderSplitPipeline:
        return self

    def __next__(self) -> str:
        if self._stage_index >= len(self.STAGES):
            raise StopIteration
        name = self.STAGES[self._stage_index]
        self._stage_index += 1
        getattr(self, f"_run_{name}")()
        return name

    def run(self) -> DatasetDict:
        for _ in self:
            pass
        assert self.splits is not None, "pipeline drained without producing splits"
        return self.splits

    def render_row(self, example: dict) -> dict:
        text = self.tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False)
        if not text.endswith(self.tokenizer.eos_token):
            text += self.tokenizer.eos_token
        return {"text": text}

    def tokenize_row(self, example: dict) -> dict:
        return {"input_ids": self.tokenizer(example["text"], add_special_tokens=False)["input_ids"]}

    def _run_render(self) -> None:
        self.dataset = self.dataset.map(
            self.render_row, remove_columns=["messages"], desc="chat template")

    def _run_dedupe(self) -> None:
        seen: set[str] = set()
        unique: list[int] = []
        for i, text in enumerate(self.dataset["text"]):
            if text not in seen:
                seen.add(text)
                unique.append(i)
        self.dataset = self.dataset.select(unique)
        self._deduped = len(unique)

    def _run_filter(self) -> None:
        # Token-accurate length filter: drop, never truncate, so answers stay whole.
        texts = list(self.dataset["text"])
        keep_idx: list[int] = []
        self.lengths = []
        for start in range(0, len(texts), 1000):
            batch = texts[start:start + 1000]
            encoded = self.tokenizer(batch, add_special_tokens=False)["input_ids"]
            for offset, ids in enumerate(encoded):
                if len(ids) <= self.cfg.data.max_len:
                    keep_idx.append(start + offset)
                    self.lengths.append(len(ids))
        self.dataset = self.dataset.select(keep_idx)
        log.info("deduped %d | length-filtered %d", self._deduped, len(self.dataset))

    def _run_split(self) -> None:
        # Eval is drawn only from rows that fit eval_max_len: evaluation runs without
        # loss scaling, and one overflowing long sample poisons the whole number.
        eval_pos = [i for i, n in enumerate(self.lengths)
                    if n <= self.cfg.data.eval_max_len][:self.eval_size]
        eval_set = set(eval_pos)
        self._eval = self.dataset.select(eval_pos)
        self._train = self.dataset.select(
            [i for i in range(len(self.dataset)) if i not in eval_set])

    def _run_tokenize(self) -> None:
        assert self._train is not None and self._eval is not None, "split stage did not run"
        self.splits = DatasetDict({
            "train": self._train.map(self.tokenize_row, desc="tokenize train"),
            "eval": self._eval.map(self.tokenize_row, desc="tokenize eval"),
        })


def render_and_split(cfg: Config, tokenizer, dataset: Dataset, *, eval_size: int) -> DatasetDict:
    """Render chat messages to text, filter, split and tokenize via RenderSplitPipeline."""
    return RenderSplitPipeline(cfg, tokenizer, dataset, eval_size=eval_size).run()


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
        "loaders_version": LOADERS_VERSION,
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
    """Resolve every source repo to its current commit SHA.

    Raises on the first failure: a manifest must never silently record an
    unresolved revision and pretend the corpus is fingerprinted.
    """
    api = HfApi()
    return {
        source.name: api.dataset_info(source.repo).sha or "unknown"
        for source in cfg.data.sources
    }


def _percentile(sorted_lengths: list[int], q: float) -> int:
    return sorted_lengths[min(len(sorted_lengths) - 1, int(len(sorted_lengths) * q))]


def manifest(cfg: Config, splits: DatasetDict, *, smoke: bool) -> dict:
    lengths = sorted(len(x) for x in splits["train"]["input_ids"]) or [0]

    return {
        "fingerprint": fingerprint(cfg, smoke=smoke),
        "content_sha256": content_hash(splits),
        "model_id": cfg.model.id,
        "sources": {source.name: source.repo for source in cfg.data.sources},
        "revisions": _resolved_revisions(cfg),
        "max_len": cfg.data.max_len,
        "eval_max_len": cfg.data.eval_max_len,
        "rows": {split: len(splits[split]) for split in splits},
        "tokens": {"p50": _percentile(lengths, 0.50),
                   "p90": _percentile(lengths, 0.90), "max": lengths[-1]},
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
