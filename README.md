# Falcon-H1-3B cyber QLoRA — production pipeline

Fine-tunes [`tiiuae/Falcon-H1-3B-Instruct`](https://huggingface.co/tiiuae/Falcon-H1-3B-Instruct)
on seven public red-team / cyber-security datasets with QLoRA.

Written after a notebook-based attempt kept failing on dtype plumbing. The rules here:

- **No monkeypatching.** No `trainer.compute_loss` override, no GradScaler hooks, no
  mid-training dtype coercion.
- **TRL is not a dependency.** The dataset is pre-rendered and pre-tokenised, so
  `transformers.Trainer` does the whole job.
- **Every stage is a command** with its own input, output and test.

## Hardware

Works on a **Colab T4** (fp16) and is faster on **Ampere or newer** (A100, L4,
RTX 30xx/40xx, H100) with bf16 and fused Mamba kernels:

| Requirement | Why |
|---|---|
| SM ≥ 8.0 (recommended) | `mamba-ssm` has no Turing kernels, so Falcon-H1 runs a PyTorch Mamba loop on a T4: **~1.68 s/sample**, ≈10.5 h for 22k rows. |
| bf16 on Ampere+ | `dtype: auto` selects bf16 where it is real; a T4 runs fp16 end-to-end instead. |

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[train]"                                   # runtime deps + package
pip install -r requirements-fast.txt --no-build-isolation   # Ampere+ only

make smoke      # 400 rows -> 5 steps. ~2 minutes. Run this before anything else.
make lint       # ruff + mypy (no GPU needed)
make test       # pytest; GPU-only tests skip themselves
make data       # full corpus -> data/processed/full (+ manifest.json)
make train      # adapter -> outputs/falcon-h1-3b-cyber
make eval       # per-sample loss, worst offenders, non-finite detection
make export     # merge -> f16 GGUF -> Q4_K_M
```

`HF_TOKEN` is read from the environment. Never commit it.

## Colab T4

`notebooks/run.ipynb` is the T4 entry point: open it in Colab with a T4 runtime and
`Run All` with the defaults (`RUN_TRAIN = False`). It probes the GPU, installs with
the `colab` extra (which keeps Colab's pre-installed torch instead of replacing it),
selects `configs/qlora-3b-t4.yaml` on pre-Ampere GPUs, and runs smoke → probe → subset
behind the `SMOKE` / `RUN_TRAIN` / `MAX_STEPS` flags. Do not install
`requirements-fast.txt` on a T4 (`mamba-ssm` has no Turing kernels); the PyTorch
Mamba fallback is slower (~1.68 s/sample) but numerically identical. The full ~22k
corpus is not a free-T4 target — the T4 profile trains a ~5–6k subset with
`save_steps: 100` + `resume: true`, so a disconnect costs at most 100 steps.

```bash
# equivalent CLI path (outside Colab)
CONFIG=configs/qlora-3b-t4.yaml make smoke-t4
CONFIG=configs/qlora-3b-t4.yaml make train-t4   # subset; set train.max_steps in YAML for a probe
```

## Use as a library

Every stage is a function; the CLI and the notebook are thin wrappers over these.

```python
from cyberft import load_config, prepare_data, train, evaluate, generate, export_gguf, train_dir

cfg = load_config("configs/qlora-3b.yaml")     # validated dataclass

splits  = prepare_data(cfg, smoke=True)        # cached DatasetDict
trainer = train(cfg, smoke=True)               # returns transformers.Trainer, saves adapter
adapter = train_dir(cfg, smoke=True)           # where the adapter/checkpoints live

report = evaluate(cfg, smoke=True)             # EvalReport, one loss per row
print(report.summary())
report.assert_finite()                         # raises NonFiniteLossError, never returns nan

answers = generate(cfg, ["Write a tcpdump command for DNS only"], adapter=adapter)
path = export_gguf(cfg, adapter=adapter)       # Path to the quantized GGUF
```

`notebooks/run.ipynb` is exactly this, cell by cell, with `inspect.getsourcefile()` links
back to each module.

The library never parses arguments, configures logging, or calls `sys.exit`. Failures are
typed exceptions (`ConfigError`, `NotBuiltError`, `DependencyError`, `NonFiniteLossError`)
so a notebook can catch them and a scheduler can map them to exit codes.

## CLI

```bash
python -m cyberft.cli info      # gpu, dtype, resolved config, corpus stats
python -m cyberft.cli data      # --smoke, --force
python -m cyberft.cli train     # --smoke
python -m cyberft.cli eval      # --smoke
python -m cyberft.cli export
```

No command requires a flag: `--config` defaults to `configs/qlora-3b.yaml`.
Step caps (`train.max_steps`), adapter paths and generation prompts live in the
YAML config and the library (`with_train`, `generate`), not in CLI flags.

## Layout

```
configs/qlora-3b.yaml        all hyperparameters, validated on load
notebooks/run.ipynb          thin driver over the library (no logic duplicated)
src/cyberft/errors.py        typed exceptions, no SystemExit in library code
src/cyberft/cli.py           argparse -> library -> exit codes, top-level imports
src/cyberft/config.py        typed config, rejects unknown keys
src/cyberft/loaders.py       LoaderBase + one class per dataset -> chat template -> Arrow + manifest
src/cyberft/model.py         4-bit load, LoRA, one dtype decision
src/cyberft/train.py         plain transformers Trainer, --smoke / resume
src/cyberft/evaluate.py      EvalReport, per-row loss
src/cyberft/export.py        merge -> GGUF with existence asserts
tests/                       config, loaders, cache, packaging
```

## Non-obvious decisions

**`llm_int8_skip_modules: ["out_proj"]`** — Mamba's `out_proj` weight is handed directly to
a fused kernel, which cannot consume a 4-bit packed tensor. Required by
[TII's fine-tuning guide](https://github.com/tiiuae/Falcon-H1/blob/main/docs/finetuning.md).

**LoRA targets exclude `out_proj` and `conv1d`** — those modules bypass the LoRA forward,
and depthwise `conv1d` has no channel mixing to adapt. `config.validate()` rejects both.

**`in_proj` is targeted** — it is an ordinary projection in the Mamba mixer and is included
in the config Axolotl ships for this model.

**Single precision end-to-end** — the base model is loaded in one dtype (fp16 by
default, bf16 on Ampere+ via `dtype: auto`) and `TrainingArguments` runs with
`bf16=False, fp16=False`. There is no AMP, no GradScaler and no autocast in the
train or eval paths; `amp_dtype()` only selects the 4-bit compute dtype for the
base-model load.

**Evaluation never masks failure** — `cyberft.evaluate` computes one loss per row and exits
non-zero if any is non-finite, printing the offending row's length and text. A mean of
`nan` is not an acceptable result.

**Dependencies are declared once** — the training runtime lives in the `train` extra, so
`pip install -e ".[train]"` cannot produce an importable-but-untrainable install, and a
test asserts every third-party import appears there.

**Data is built once and fingerprinted** — `data.sources` in the YAML is the single
description of where the corpus comes from (repo, subset, splits, optional single-file
Parquet, revision, row cap). `make data` writes Arrow + `manifest.json` with row counts,
token percentiles, a content hash over the token ids, the resolved commit SHA of every
source, and a configuration fingerprint. If anything that shapes the corpus changes,
`prepare_data` refuses to reuse the cache and tells you to pass `force=True`, instead of
silently training on the previous dataset. Set `revision:` on a source to pin it to a
branch, tag or commit.

## Operating notes

- **Throughput** (T4, PyTorch fallback, no `mamba-ssm`): 0.6 samples/s. On an Ampere GPU
  with the fused kernels expect roughly 4–5× faster.
- **Resume**: `train.resume: true` picks up the newest checkpoint automatically.
- **Monitoring**: set `train.report_to: ["tensorboard"]`; logs land in the output dir.
- **Memory**: 4-bit base + LoRA + gradient checkpointing fits 16 GB at `micro_batch: 1`,
  `max_len: 2048`. Raise `micro_batch` before `max_len`.
- **Export RAM**: the merge runs on CPU in bf16 and needs ~7 GB of host RAM.
- **CI**: `.github/workflows/ci.yml` runs ruff, mypy and the CPU tests; the GPU smoke test
  stays manual (`make smoke`).

## Troubleshooting

| Symptom | Cause | Action |
|---|---|---|
| `mat1 and mat2 shapes cannot be multiplied` in `mamba_split_conv1d_scan_combined` | Mamba `out_proj` was quantised and fed to the fused kernel | keep `out_proj` in `model.bnb_skip_modules` (TII fine-tuning guide) |
| `NonFiniteLossError` during training | loss went NaN at that step | stop; lower `lr`, check `max_len` |
| `NonFiniteLossError` during evaluation | one pathological row | `make eval` names the row; drop it in the loader class in `cyberft/loaders.py` |
| `NonFiniteLossError` on a T4 (fp16) | fp16 SSM arithmetic on a long row | lower `max_len` (try 768), then `lr` to 1e-4 |
| `No kernel image is available` from `mamba-ssm` | wrong CUDA arch | reinstall `requirements-fast.txt` on the target GPU |
| `ImportError: ... requires bitsandbytes` | runtime installed without the `train` extra | `pip install -e ".[train]"`, restart the kernel |

## Licence note

`reloading0101/threat-intelligence-dataset` is **CC-BY-NC-SA-4.0** (non-commercial);
`kimi-cyber-reasoning` is WTFPL; check every source before commercial use. The base model
ships under the Falcon-LLM licence. Use the result only for authorised testing, research
and education.
