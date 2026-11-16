"""Publish a merged model to the Hugging Face Hub as a complete repository.

A repo that holds only weights is not usable: the Hub shows no model card, no
licence, no base-model link, and llama.cpp users have nothing to download. So
``push`` builds all three artifacts before uploading and verifies them remotely:

1. the merged weights (unless repairing an existing repo with ``from_hub``);
2. ``README.md`` with a real model card and Hub metadata;
3. the configured GGUF quantization, generated from the merged weights.

Failure is loud: a missing GGUF or an upload that does not land raises instead
of leaving another half-populated repo behind.
"""

from __future__ import annotations

import json
import logging
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml
from huggingface_hub import HfApi, snapshot_download

from .config import Config
from .errors import ConfigError, NotBuiltError

log = logging.getLogger(__name__)

BASE_MODEL_LICENSE = "falcon-llm-license"
BASE_MODEL_LICENSE_URL = "https://huggingface.co/tiiuae/falcon-llm-license"
CARD_FILE = "README.md"
TRAINING_CONFIG_FILE = "training_config.json"


def render_model_card(cfg: Config, gguf_names: list[str]) -> str:
    """Return the Hub model card (YAML front matter + Markdown body).

    Pure so the card is testable without network, torch or a Hub token.
    """
    datasets = [source.repo for source in cfg.data.sources]
    front: dict[str, Any] = {
        "base_model": cfg.model.id,
        "library_name": "transformers",
        "pipeline_tag": "text-generation",
        "license": "other",
        "license_name": BASE_MODEL_LICENSE,
        "license_link": BASE_MODEL_LICENSE_URL,
        "language": ["en"],
        "tags": [
            "falcon-h1",
            "qlora",
            "peft",
            "text-generation",
            "cyber-security",
            "gguf",
        ],
        "datasets": datasets,
    }
    front_matter = yaml.safe_dump(front, sort_keys=False, allow_unicode=True).strip()
    gguf_list = "\n".join(f"- `{name}`" for name in gguf_names) or "- (none)"
    first_gguf = gguf_names[0] if gguf_names else "model.gguf"
    dataset_rows = "\n".join(
        f"| `{source.name}` | [`{source.repo}`](https://huggingface.co/datasets/{source.repo}) "
        f"| {source.cap or 'all'} |"
        for source in cfg.data.sources
    )
    return f"""---
{front_matter}
---

# {cfg.model.id.split('/')[-1]} — cyber-security QLoRA

A QLoRA fine-tune of [`{cfg.model.id}`](https://huggingface.co/{cfg.model.id})
on public red-team and cyber-security datasets. Weights are merged into the base
model, so this repository loads as a normal causal language model.

## Training

| Setting | Value |
|---|---|
| Method | QLoRA ({cfg.model.quant} base) + LoRA |
| LoRA rank / alpha / dropout | {cfg.lora.r} / {cfg.lora.alpha} / {cfg.lora.dropout} |
| LoRA targets | {", ".join(f"`{m}`" for m in cfg.lora.target_modules)} |
| Learning rate | {cfg.train.lr} |
| Epochs | {cfg.train.epochs} |
| Micro batch x grad accum | {cfg.train.micro_batch} x {cfg.train.grad_accum} |
| Max sequence length | {cfg.data.max_len} (longer rows dropped, not truncated) |
| Optimizer | `{cfg.train.optim}` |
| Seed | {cfg.train.seed} |

## Data

| Loader | Source | Rows (cap) |
|---|---|---|
{dataset_rows}

`reloading0101/threat-intelligence-dataset` is **CC-BY-NC-SA-4.0**
(non-commercial); check every source before commercial use.

## Files

- `model.safetensors` — merged weights (bf16) with the tokenizer and chat template.
- `training_config.json` — resolved configuration used for this run.
{gguf_list}

## Usage

Transformers:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "{cfg.export.hf_repo or 'your-org/your-model'}"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
messages = [{{"role": "user", "content": "Write a tcpdump command for DNS only"}}]
inputs = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
).to(model.device)
print(tokenizer.decode(model.generate(**inputs, max_new_tokens=128)[0][inputs["input_ids"].shape[-1]:],
                       skip_special_tokens=True))
```

llama.cpp (CPU or GPU, no Python):

```bash
llama-cli -m {first_gguf} -p "Write a tcpdump command for DNS only" --n-predict 128
```

## Evaluation

Training and per-row evaluation gate on finite loss; no benchmark scores are
claimed here. A NaN/inf loss aborts the run instead of publishing an adapter.

## Limitations and authorised use

This model is intended for **authorised** security testing, research and
education. It can produce offensive commands and may be wrong; review every
output before running it. The base model ships under the Falcon-LLM licence
([link]({BASE_MODEL_LICENSE_URL})), and the merged model inherits it.
"""


def write_repo_metadata(upload_dir: Path, cfg: Config, gguf_names: list[str]) -> list[Path]:
    """Write the model card and resolved training config into ``upload_dir``."""
    upload_dir.mkdir(parents=True, exist_ok=True)
    card = upload_dir / CARD_FILE
    card.write_text(render_model_card(cfg, gguf_names), encoding="utf-8")
    config = upload_dir / TRAINING_CONFIG_FILE
    config.write_text(json.dumps(asdict(cfg), indent=2, default=str) + "\n", encoding="utf-8")
    return [card, config]


def push(
    cfg: Config,
    *,
    adapter: str | Path | None = None,
    skip_merge: bool = False,
    from_hub: bool = False,
) -> str:
    """Merge (or reuse) the model, build card + GGUF, upload, and verify.

    ``from_hub`` repairs an existing repo without the original adapter: it
    downloads the published weights, regenerates card and GGUF locally, and
    uploads only those (the weights stay untouched).
    """
    if not cfg.export.hf_repo:
        raise ConfigError("export.hf_repo is not set: set the Hub repo id to push to")
    if from_hub and skip_merge:
        raise ConfigError("from_hub and skip_merge are mutually exclusive")
    # Imported lazily: torch/transformers stay off `import cyberft.hub`, so the
    # model card can be rendered and tested without the training stack.
    from .export import merge

    api = HfApi()
    if from_hub:
        with tempfile.TemporaryDirectory(prefix="cyberft-push-") as tmp:
            merged = Path(tmp)
            log.info("downloading %s for repair", cfg.export.hf_repo)
            snapshot_download(
                repo_id=cfg.export.hf_repo, local_dir=merged, ignore_patterns=["*.gguf"]
            )
            return _publish(cfg, api, merged, upload_weights=False)
    adapter_dir = Path(adapter) if adapter else Path(cfg.train.output_dir)
    if skip_merge:
        merged = Path(cfg.export.merged_dir)
        if not merged.exists():
            raise NotBuiltError(f"skip_merge=True given but {merged} does not exist")
    else:
        merged = merge(cfg, adapter_dir)
    return _publish(cfg, api, merged, upload_weights=True)


def _publish(cfg: Config, api: HfApi, merged: Path, *, upload_weights: bool) -> str:
    from .export import build_gguf  # local: keeps torch out of module import

    repo_id = cfg.export.hf_repo
    assert repo_id is not None  # push() validated it
    # Build the GGUF first: a conversion failure must stop the upload, not
    # publish a repo that is missing the file users actually download.
    gguf = build_gguf(cfg, merged)
    metadata = write_repo_metadata(merged, cfg, [gguf.name])
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    if upload_weights:
        api.upload_folder(repo_id=repo_id, folder_path=str(merged))
    else:
        for path in metadata:
            api.upload_file(repo_id=repo_id, path_or_fileobj=str(path), path_in_repo=path.name)
    api.upload_file(repo_id=repo_id, path_or_fileobj=str(gguf), path_in_repo=gguf.name)
    _verify_remote(api, repo_id, required={CARD_FILE, TRAINING_CONFIG_FILE, "config.json", gguf.name},
                   upload_weights=upload_weights)
    log.info("pushed %s: card + %s to %s", merged, gguf.name, repo_id)
    return repo_id


def _verify_remote(
    api: HfApi,
    repo_id: str,
    *,
    required: set[str],
    upload_weights: bool,
) -> None:
    remote = set(api.list_repo_files(repo_id=repo_id, repo_type="model"))
    missing = required - remote
    # Sharded or single-file, the pushed repo must carry weights.
    if upload_weights and not any(name.endswith(".safetensors") for name in remote):
        missing.add("*.safetensors")
    if missing:
        raise RuntimeError(
            f"push to {repo_id} incomplete, missing {sorted(missing)}; "
            "refusing to report success for a partial repository"
        )
