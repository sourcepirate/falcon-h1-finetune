# cyberft

## Overview

Core Python package for QLoRA fine-tuning of Falcon-H1-3B-Instruct on seven
red-team / cyber-security datasets. Pipeline: YAML config -> per-dataset
loaders -> fingerprinted Arrow cache -> single-precision Trainer
(`bf16=False, fp16=False`) -> LoRA adapter -> per-row eval -> GGUF export.

## Interface

- `config.py`: `SourceConfig`, `ModelConfig` (`quant: 4bit|none` for QLoRA vs plain-LoRA base), `DataConfig`, `LoraConfig`, `TrainConfig` (`report_to` must not contain `"wandb"`), `ExportConfig` (`llama_cpp_rev` pin consumed by the clone step), `WandbConfig` (`enabled: auto|always|never`, env wins), `Config` (with `validate()`), `load_config(path)`, `warmup_steps(cfg, train_rows)`, `with_train(cfg, **changes)`, `with_data(cfg, **changes)`, `with_export(cfg, **changes)`, `with_wandb(cfg, **changes)`
- `args.py`: `build_parser()` — subcommands `data/train/eval/export/info`; `train` takes `--wandb/--no-wandb`
- `cli.py`: `run(args) -> int` (dispatches subcommands), `main(argv) -> int` (logging setup, `CyberftError` -> exit 1, `KeyboardInterrupt` -> 130)
- `errors.py`: `CyberftError`, `ConfigError`, `NotBuiltError`, `DependencyError`, `TrackingError`, `NonFiniteLossError`
- `loaders.py`: `LoaderBase`, `RedTeamLoader`, `KaliLoader`, `KimiLoader`, `CTILoader`, `ExploitDBLoader`, `LinuxCommandsLoader`, `DebianLoader`, `LOADERS`, `get_loader(name)`, `build_messages(cfg)`, `RenderSplitPipeline` (`run()`, `render_row()`, `tokenize_row()`), `render_and_split(cfg, tokenizer, dataset, eval_size)`, `build(cfg, tokenizer, smoke)`, `fingerprint(cfg, smoke)`, `content_hash(splits)`, `manifest(cfg, splits, smoke)`, `prepare_data(cfg, smoke, force, tokenizer)`, `data_manifest(cfg, smoke)`
- `model.py`: `amp_dtype(cfg)`, `build_tokenizer(cfg)` (Tiny fallback template `TINY_FALLBACK_CHAT_TEMPLATE` when the Hub ships none), `build_model(cfg)` (plain base when `model.quant == "none"`), `attach_adapter(cfg, adapter_dir)`
- `tracking.py`: `NullTracker`, `WandbTracker` (`log`, `summary`, `finish`, `enabled`), `resolve_enabled(conf, smoke, override, env)`, `metric_name(key)`, `clean_metrics(metrics)`, `final_summary(history, total_steps, global_step)`, `start(conf, smoke, override, process_index, run_config, env)`
- `train.py`: `FiniteLossCallback`, `MetricsCallback` (after `FiniteLossCallback`), `expected_total_steps(cfg, train_rows, smoke)`, `train_dir(cfg, smoke)`, `load_splits(cfg, smoke)`, `build_args(cfg, train_rows, smoke)`, `train(cfg, smoke, *, wandb)`
- `evaluate.py`: `RowLoss`, `EvalReport` (`worst()`, `summary()`, `assert_finite()`), `evaluate(cfg, adapter, split, smoke)`, `generate(cfg, prompts, adapter, smoke, max_new_tokens)`
- `export.py`: `merge(cfg, adapter_dir)`, `convert(cfg, merged)`, `quantize(cfg, f16)`, `export_gguf(cfg, adapter, skip_merge)`
- `__init__.py`: `__version__` (`0.5.0`), lazy `__getattr__` re-exports of Config/loaders/model/train/eval/export entry points

## Invariants

- `wandb.enabled` in `(auto, always, never)`; `wandb.mode` is None or in `(online, offline, disabled)`; `wandb.project/entity/run_name` non-empty when set; `wandb.tags` has no empty strings; `train.report_to` must not contain `"wandb"` (metrics go through `tracking.py`, not HF's callback).

- `Config.validate`: `model.id` non-empty; `model.dtype` in `(auto, bf16, fp16)`; `model.quant` in `(4bit, none)`; `model.attn` in `(sdpa, flash_attention_2)`; missing `out_proj`/`lm_head` in `bnb_skip_modules` warns (not fails) — hard-fail history assumed Falcon-H1-3B's vocab 65537, but Tiny-R-90M has vocab 32768 with tied embeddings.
- LoRA targets must be non-empty and must not intersect `{out_proj, conv1d}`; `lora.r` positive.
- Data sources non-empty, no duplicate names, each needs `name` + `repo` with non-empty `splits` and positive `cap`; `parquet` / `subset` mutually exclusive; `max_len >= 128`; `0 < eval_max_len <= max_len`; `eval_size > 0`.
- Train hyperparams: `lr > 0`, `epochs > 0`, `max_steps == -1` or `> 0`, `micro_batch` / `grad_accum > 0`, `warmup_ratio` in `[0, 1)`, `save` / `eval` / `logging_steps > 0`.
- Unknown dataclass keys and unknown top-level YAML sections raise `ConfigError`; YAML must be a mapping with a `model` section.
- `build_messages` requires configured sources to exactly cover `LOADERS` keys — unknown names and missing datasets both raise `ConfigError`.
- `LoaderBase` subclass must set `DATASET_NAME` matching `spec.name`, else `TypeError` / `ConfigError`.
- `LOADERS_VERSION` bump invalidates Arrow cache; `fingerprint` mismatch without `force=True` raises `NotBuiltError` (never silently reuses stale cache).
- Length filter drops (never truncates) rows exceeding `max_len`; eval rows drawn only from lengths `<= eval_max_len`.
- `build_tokenizer` raises `RuntimeError` if the model ships no chat template and is not a Tiny model (Tiny ids fall back to `TINY_FALLBACK_CHAT_TEMPLATE`); `amp_dtype(auto)` falls back to fp16 on T4.
- `quant == "none"` loads the base without `BitsAndBytesConfig` (plain LoRA path) in both `build_model` and `attach_adapter`.
- Single precision end-to-end: `build_args` sets `bf16=False, fp16=False` (no AMP/GradScaler).
- Checkpoint resume only when `cfg.train.resume` and not smoke.
- `EvalReport.assert_finite` raises `NonFiniteLossError` listing first 10 broken rows; `eval` CLI returns 1 when `report.broken` non-empty.
- `merge` raises `NotBuiltError` when `adapter_config.json` missing; `convert` raises `DependencyError` when `convert_hf_to_gguf.py` missing; `quantize` raises `DependencyError`/`RuntimeError` on missing binary/output; `export_gguf(skip_merge=True)` requires existing `merged_dir`.

## Side Effects

- Network: HuggingFace Hub fetch via `load_dataset` / `hf_hub_download` (`LoaderBase._load_ds`); `HfApi().dataset_info` per source for manifest revisions.
- Network: `train` opens a wandb run when enabled (Goggle `WANDB_*` env); init failure raises `TrackingError`, post-init failures only warn.
- Filesystem writes: `prepare_data` (`shutil.rmtree` + `save_to_disk` + `manifest.json` under `data.output_dir/{full,smoke}`); `train` writes adapter + tokenizer via `save_pretrained`; `merge` writes merged safetensors + tokenizer to `export.merged_dir` (CPU bf16 load, ~7GB RAM); `convert`/`quantize` run `llama.cpp` subprocesses into `gguf_dir`.
- Filesystem reads: Arrow cache hit via `load_from_disk`; checkpoint resume via `get_last_checkpoint`.
- Compute: GPU-heavy 4-bit QLoRA training via `transformers` Trainer; `evaluate` runs per-row forward passes under `torch.no_grad`.
- Stdout: split sizes + manifest JSON (`data`), eval summary (`eval`), GGUF path + size (`export`), gpu/dtype/config JSON/corpus manifest (`info`); `main` configures root `logging.basicConfig` INFO and maps `CyberftError` to stderr + exit 1.

## Verification

- Run: `python -m pytest`
