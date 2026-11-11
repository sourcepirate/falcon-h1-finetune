# Falcon-H1-Tiny-R-90M Repurpose + Docker + GHCR

* **Author:** Sathya (sourcepirate/falcon-h1-finetune)
* **Status:** Approved
* **Date:** 2026-11-10
* **Revision:** 2026-11-11 — GHCR decision incorporated (tags-only, public `ghcr.io/sourcepirate/falcon-h1-tiny`, owner-approved); two review rounds resolved (16 + 13 findings); Status Approved.

---

## Overview

This design moves the repo from Falcon-H1-3B-Instruct to Tiny (Tiny means `tiiuae/Falcon-H1-Tiny-R-90M`) while keeping the same seven cyber-security datasets. It adds one Tiny config, one CUDA Dockerfile that runs on the T4 smoke host and the RunPod RTX 3090 pod, and a CI job that builds the image and pushes it to GHCR on tags. The 3B files stay untouched as legacy.

## The Problem

The repo only trains a 3B model. Its settings were picked for 3B instruction tuning: rank 32, learning rate 2e-4, 4-bit base, 2048 token length, and required `lm_head`/`out_proj` skips. Applied unchanged to a 90M reasoning model, they risk overfitting, wasted capacity, and false validation failures. There is no container, so T4 test runs and 3090 RunPod runs drift. CI only runs lint and CPU tests. Export to GGUF (both f16 and Q4_K_M, mandatory for this work) is unverified for the Tiny architecture.

Terms: QLoRA (frozen 4-bit base plus small trainable side weights); GGUF (loadable quantized model file); GHCR (GitHub Container Registry); sm_75/sm_86 (CUDA compute capabilities of the T4 and the RTX 3090, both carried by one image); Mamba (non-attention sequence-mixing layers in Falcon-H1); Arrow cache (tokenized corpus on disk plus manifest).

## Goals

* Train Tiny on the same 7 datasets with the v1 probe recipe (r=8, alpha=16, dropout=0.05, 4-bit baseline with LoRA-only fallback) that passes `make smoke` on T4 and a full finite-loss run on the RunPod RTX 3090 pod.
* Ship one `configs/qlora-tiny.yaml` with corrected lengths, caps, paths, and skip/target lists verified against Tiny's real architecture.
* Ship one CUDA Dockerfile (sm_75 + sm_86) tested on T4 and runnable on a 3090 RunPod pod with pinned torch/CUDA/bitsandbytes and documented build/run commands.
* Extend `.github/workflows/ci.yml` to build on PRs and push to GHCR on main pushes and version tags, keeping ruff + mypy + pytest green.
* Keep eval finite-loss detection working and make `make export` produce loadable `model-f16.gguf` and `model-q4_k_m.gguf` for Tiny.

## Non-Goals

* No 3B retuning or joint 3B+Tiny test matrix; 3B files stay as legacy.
* No hyperparameter search, no multi-epoch tuning, no new datasets or loader rewrites beyond cap/length adjustments.
* No RunPod automation (manual pod + container run only).
* No GGUF quality benchmarking beyond finite-loss + loads + 3 sample generations (prompts recorded in the export check).
* No multi-arch manifest or image-size optimization beyond a single pinned CUDA base (single image carrying sm_75 + sm_86; over 12 GB measured by `docker images` fails this pass).
* No `notebooks/run.ipynb` changes; the notebook stays 3B-readable.

## Open Questions

* **Tiny architecture (resolved 2026-11-11, recorded in `configs/qlora-tiny.yaml` header)**: vocab 32768 with tied embeddings (3B 65537 rationale N/A), hybrid Mamba+attention (keep `out_proj` skip), reasoning delimiters shipped, NO chat template (Tiny fallback in `model.py`); layer names assumed from 3B family, re-verify on first smoke.
* **Training method**: small-rank LoRA vs 4-bit QLoRA vs full — needs a Tiny probe, not 3B habit.
* **CUDA base tag**: which tag matches `torch==2.11.0` and carries sm_75 + sm_86.
* **GHCR policy (decided 2026-11-11, extended 2026-11-12)**: public `ghcr.io/sourcepirate/falcon-h1-tiny`, tags `:sha` / `:tiny` / `:latest`, push on `main` pushes and version tags (`v*`), build-only on PRs.
* **llama.cpp support**: does `convert_hf_to_gguf.py` accept the Tiny hybrid arch; which pin fixes it if not.

## Risks / Assumptions / Dependencies

| Risk | Mitigation | Owner |
|---|---|---|
| Tiny arch facts unknown (vocab, layers, template, Mamba mix) | Hub read is build-order step 1; Hub-facts table (vocab, layers, template, delimiters, Mamba mix, block size) recorded in the config PR; no config/validator/model work lands without it | implementer |
| Adverse Hub facts (no template, layer mismatch, no Mamba mix) | Named branch per case: template absent takes the recorded fallback; layer mismatch re-derives skip/target lists before writing YAML; no Mamba mix drops the `out_proj` skip under the warn-only rule | implementer |
| v1 recipe wrong for 90M (rank, quant, lengths) | Probe defaults with fallback trigger and overfit/drop-rate gates; probe log records overlong-drop % and loss; caps cut only on trigger | implementer |
| CUDA base mismatch (sm_75 + sm_86) | Example tag decision recorded before Dockerfile lands; T4 smoke log proves both arches | implementer |
| GHCR policy (decided 2026-11-11, extended 2026-11-12: public `ghcr.io/sourcepirate`, push on `main` + tags) | CI item follows the decision; no code change beyond the `image` job | repo owner |
| `llama.cpp` rejects Tiny hybrid arch | Pin plus contingency (next pin or patch under `export.llama_cpp_dir`); export log shows both GGUF loads | implementer |
| Hub datasets or RunPod 3090 unavailable | Public Hub pulls assumed; data manifest proves corpus, pod log proves run; manual retry on preemption, no automation | implementer |

## Critical User Journeys (CUJs)

### CUJ 1: T4 smoke in container

* **Given**: clean checkout, Docker with GPU access on the T4 smoke host, `configs/qlora-tiny.yaml` present.
* **When**: user runs `docker build -t ghcr.io/sourcepirate/falcon-h1-tiny:dev .`, then `docker run --gpus all ghcr.io/sourcepirate/falcon-h1-tiny:dev CONFIG=configs/qlora-tiny.yaml make smoke` (smoke runs inside the container, workdir `/work`; the corpus cache is container-local, fine for a trial).
* **Then**: data builds (400 rows), 5 training steps finish with finite loss, no `NonFiniteLossError`.

### CUJ 2: 3090 full train on RunPod

* **Given**: GHCR image pulled on the RunPod RTX 3090 pod.
* **When**: on the pod, user runs `docker pull ghcr.io/sourcepirate/falcon-h1-tiny:tiny` then `docker run --gpus all <image> CONFIG=configs/qlora-tiny.yaml make data train eval` (train and eval run inside the container; outputs persist on pod disk).
* **Then**: training completes one epoch over the full corpus at starting caps (about 42,305 rows before dedupe/filter; exact count in the data manifest), `eval` reports zero non-finite rows.

### CUJ 3: Mandatory GGUF export

* **Given**: trained Tiny adapter on either GPU.
* **When**: user runs `make export` with the Tiny config.
* **Then**: merge produces safetensors, `convert_hf_to_gguf.py` produces `outputs/gguf-tiny/model-f16.gguf`, `llama-quantize` produces `outputs/gguf-tiny/model-q4_k_m.gguf`, and both load: `llama-cli -m outputs/gguf-tiny/model-f16.gguf -p "Write a tcpdump command for DNS only" --n-predict 64` exits 0 with non-empty output, same for the `model-q4_k_m.gguf` file.

### CUJ 4: CI build on pull request

* **Given**: open pull request.
* **When**: CI runs the `image` job with `push: false`.
* **Then**: image builds without pushing; `check` (ruff), `types` (mypy), and `resolvable` jobs pass.

### CUJ 5: CI publish on main push or version tag

* **Given**: push to `main` or version tag `v*` pushed.
* **When**: CI runs the `image` job with `push: true`.
* **Then**: GHCR holds `:sha`, `:tiny`, and `:latest` tags for the public `ghcr.io/sourcepirate/falcon-h1-tiny` image.

## Technical Plan (Architecture)

The current pipeline is YAML -> loaders (one class per dataset) -> Arrow cache + manifest -> Trainer (no AMP, single dtype) -> LoRA adapter -> eval per-row loss -> merge -> f16 GGUF -> Q4_K_M. The Tiny work keeps this shape and changes only the model-specific inputs, one config, and the packaging around it.

```text
configs/qlora-tiny.yaml
  -> loaders.build_messages (same 7 names, adjusted caps)
  -> render (Tiny chat template) -> dedupe -> length-filter -> split -> tokenize
  -> train (small-rank LoRA probe, FiniteLossCallback)
  -> eval (per-row finite check)
  -> export (merge -> f16 -> Q4_K_M, both mandatory)
  -> Dockerfile (one CUDA image, sm_75 + sm_86)
  -> CI (build on PR, push to GHCR on tags)
```

What changes and why:

* **Config (`configs/qlora-tiny.yaml`, create)**: single file for both GPUs. Holds Tiny model id, `dtype: auto`, and v1 probe defaults (`r: 8, alpha: 16, dropout: 0.05`). Quantization is decided by the probe: start 4-bit on as the carryover baseline and exit to LoRA-only on T4 OOM or fp16 loss instability. (Alternative A stays rejected as an unconditional carryover of r=32; it is retained only as this gated baseline.) `max_len`/`eval_max_len` and per-source caps are copied from `qlora-3b.yaml` as the start, and `bnb_skip_modules` / `target_modules` are corrected after reading Tiny's real layer names. Probe gate: smoke slice 400 rows, 5 steps, finite loss plus render overlong-drop at or under 5 percent of rendered rows in the smoke `length-filtered` log line (kimi rows counted separately). Overfit trigger for cap cuts: eval-loss rise or manual review after the first full run. One file avoids the current 3B two-file split (`qlora-3b.yaml`, `qlora-3b-t4.yaml`); a second Tiny file is added only if smoke OOMs at stated `max_len` on the T4 smoke host or overlong-drop stays over 5 percent after cap cuts, recorded in the config PR.
* **Validation (`src/cyberft/config.py`, modify)**: the hard requirements that `bnb_skip_modules` contain `out_proj` and `lm_head` cite 3B facts (Mamba kernel, vocab 65537). Change in this order: (a) demote a missing `lm_head`/`out_proj` skip to a warning for all models now; keep only `FORBIDDEN_LORA_TARGETS = {"out_proj", "conv1d"}` as a hard fail, re-verified once against Tiny layer names after the Hub read; (b) add model-id conditioning only if the probe proves warn-only unsafe. 3B behavior otherwise stays byte-identical.
* **Model load (`src/cyberft/model.py`, modify)**: Hub facts to record first are chat template present/absent, reasoning delimiters, special tokens, Mamba/attention mix, and vocab size. `build_tokenizer` keeps its raise-unless-template behavior until the read, then takes the named fallback if Tiny-R needs it; `_load_base` keeps `attn: sdpa` and 4-bit until the read locks Tiny values (`llm_int8_skip_modules`, `attn_implementation`). No new public functions; only corrected constants and template handling.
* **Corpus (`src/cyberft/loaders.py`, modify only if needed)**: keep the seven loader classes and the exact-7-set check in `build_messages`. Change only caps/lengths via YAML, unless the Tiny template renders `kimi-cyber-reasoning` rows into overlong sequences that get dropped — then adjust `max_chars` or row mapping in a small, tested edit.
* **Dockerfile (create)**: one CUDA base `nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04` with torch from `https://download.pytorch.org/whl/cu126` (torch 2.11.0 cu126 verified; confirm-or-repin on first build), Python 3.11, workdir `/work`, ordered layers (copy `pyproject.toml` + `src`, `pip install .[train]`, `requirements-fast.txt` fused-kernel layer with pure-torch fallback), `ENV CONFIG=configs/qlora-tiny.yaml`. Documents literal `docker build -t ghcr.io/sourcepirate/falcon-h1-tiny:dev .`, `docker run --gpus all` smoke, and RunPod pull-and-run commands. Expected size 8-12 GB; over 12 GB measured by `docker images` fails this pass.
* **CI (`.github/workflows/ci.yml`, modify)**: keep the CPU jobs untouched (`check` runs ruff plus pytest, `types` runs mypy, `resolvable` dry-runs the dependency resolve). Add a new job with `packages: write` that logs in to GHCR, builds the image on every PR without pushing, and pushes `:sha` / `:tiny` / `:latest` to public `ghcr.io/sourcepirate/falcon-h1-tiny` on `main` pushes and version tags (`v*`) both.
* **Docs (`README.md`, `Makefile`, modify lightly)**: README quickstart is the canonical Tiny entry (build/run/pull + `CONFIG=... make smoke/train/eval/export` + GGUF load check); mark 3B configs legacy. `Makefile` comments link there. No new automation.

Failure handling stays as today: `ConfigError` on bad YAML, `NotBuiltError` on stale cache, `NonFiniteLossError` on NaN (training stops, eval names the row). On a non-finite row: eval names it in the broken list; quarantine by dropping or re-examining it in the loader, then rerun eval. One bad row triggers investigation; the pass requires zero non-finite rows after quarantine or a documented cause. Export asserts each output exists. Tiny acceptance is smoke finite plus eval zero non-finite rows plus render overlong-drop at or under 5 percent; deeper quality checks are deferred for this pass.

## Alternatives Considered

### Alternative A: Carry over QLoRA 4-bit unchanged (r=32, NF4 base)

* **Pros**: No code change to `model.py`; same install as 3B; T4 memory headroom.
* **Cons**: At 90M, 4-bit saves about 300 MB but adds the fp16 Mamba instability already documented for T4; r=32 is proportionally large and invites overfit.
* **Reason for Rejection**: Rejected as an unconditional carryover (r=32 unchanged, which trades stability for little memory gain at 90M). Retained only as the gated probe baseline: start 4-bit on, exit to LoRA-only on T4 OOM or fp16 loss instability.

### Alternative B: Four configs (3B x 2 + Tiny x 2)

* **Pros**: Mirrors the current Ampere/T4 split exactly.
* **Cons**: Doubles config and Makefile surface; 90M fits on either card even without quantization, so the split may be unneeded.
* **Reason for Rejection**: Sprawl. One `qlora-tiny.yaml` is the default; a second Tiny file is added only if T4 memory forces it.

### Alternative C: Per-GPU Docker images

* **Pros**: Each image carries only its own kernels; smaller per-image size.
* **Cons**: Doubles build, test, and push work; drift between T4 and 3090 images.
* **Reason for Rejection**: One image carrying sm_75 + sm_86 is simpler and matches the single-config choice. Per-GPU images return only if kernel or size problems force them.

### Alternative D: Push image on every commit

* **Pros**: Every commit is immediately runnable on RunPod.
* **Cons**: 8-12 GB pushes on each commit burn CI minutes and registry storage.
* **Reason for Rejection**: Build-only on PRs, push `:sha` / `:tiny` / `:latest` on version tags only.

### Alternative E: f16-only GGUF export

* **Pros**: Avoids the unverified Q4_K_M converter path for the Tiny hybrid arch.
* **Cons**: Breaks the user's mandatory requirement; downstream runners expect quantized files.
* **Reason for Rejection**: User states GGUF is mandatory. Both f16 and Q4_K_M must pass; if the converter rejects Tiny, pin or patch `llama.cpp` under `export.llama_cpp_dir`.

### Alternative F: Delete 3B configs now

* **Pros**: Cleaner tree, no legacy confusion.
* **Cons**: Mixes deletion with addition; failures become hard to attribute.
* **Reason for Rejection**: Keep 3B files untouched as legacy in this pass; remove later.

---

## Non-Functional Requirements

* Rollout: tags `:sha` (full SHA for traceability), `:tiny`, and `:latest` (moving alias, never a rollback target). Retain every release `:sha` tag and every `:tiny` release tag; no auto-deploy, manual pod + container run only. Pushes happen on `main` pushes and version tags. Rollback is pulling the prior release `:sha` tag.
* Security: GHCR package `ghcr.io/sourcepirate/falcon-h1-tiny` is public, approved by repo owner (Sathya) 2026-11-11; CI uses `GITHUB_TOKEN` with `packages: write`; no secrets in the image; datasets remain public Hub pulls.
* Performance: T4 smoke is 400 rows and 5 steps (target under 30 minutes on T4, informational: slower only records actual and investigates caps before the 3090 run). 3090 full run is one epoch over the full corpus at starting caps (rows counted in the data manifest; wall-clock recorded in the run log, non-gating). Image build time recorded in the run log, non-gating; `docker images` size recorded; over 12 GB fails this pass (expected image 8-12 GB from CUDA base plus torch; on failure trim build outputs and re-measure, or record the base-alone size as accepted-unknown).
* Deferred with reason: RunPod preemption is manual retry; logs are Trainer logs plus the data manifest; CI minutes/storage, Hub checksums, and base-image digest pinning are recorded but not budgeted in this pass.

---

## Detailed Implementation Plan

### Build Order

1. Hub read: open `tiiuae/Falcon-H1-Tiny-R-90M` and record the Hub-facts table (columns: vocab size, layer names for attention and Mamba blocks, chat template text or Hub link, reasoning delimiters, Mamba/attention mix, quantization block size; done = all 6 filled plus adverse branches resolved). Artifact: a "Hub facts" section in the config PR description, mirrored as the comment header of `configs/qlora-tiny.yaml`.
2. Validator + model constants (+ `export` pin-field schema if a new field is needed).
3. Tiny YAML + fixture test.
4. Makefile + README.
5. Dockerfile + CI `image` job.
6. T4 smoke, 3090 full train + eval, mandatory export.

### Target Files to Create/Modify

#### 1. `configs/qlora-tiny.yaml` (Create)

* **Type / Scope**: Training config, single file for T4 and 3090.
* **Proposed Changes**: Set `model.id` to `tiiuae/Falcon-H1-Tiny-R-90M`, `dtype: auto`, `attn: sdpa`, and new `model.quant: "4bit" | "none"` (default `"4bit"`; unknown values rejected by `ConfigError` like other enums). `"none"` selects the LoRA-only path: `_load_base` skips `BitsAndBytesConfig` and `build_model` skips `prepare_model_for_kbit_training`. v1 probe defaults: `r: 8, alpha: 16, dropout: 0.05`, 4-bit on (fallback to `"none"` on T4 smoke CUDA OOM or fp16 loss NaN/spike over the 5 probe steps); `max_len`/`eval_max_len` and per-source caps copied from `qlora-3b.yaml` as the start. Set `bnb_skip_modules` and `lora.target_modules` after the Hub read (candidates: keep `out_proj` skip, re-check `lm_head` need; full target names `q_proj, k_proj, v_proj, o_proj, in_proj, gate_proj, up_proj, down_proj`). Sources (exact-7-set): redteam (20000), kali (343), kimi (997), cti (6000), exploitdb (8000), linuxcmd (6000), debian (965). Map outputs to keys: `train.output_dir: outputs/falcon-h1-tiny-cyber`, `export.merged_dir: outputs/falcon-h1-tiny-merged`, `export.gguf_dir: outputs/gguf-tiny`.

#### 2. `Dockerfile` (Create)

* **Type / Scope**: Container build, one image for sm_75 (T4) + sm_86 (3090).
* **Proposed Changes**: Pin the literal base `nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04` with torch from index `https://download.pytorch.org/whl/cu126` (torch 2.11.0 ships cu126 and cu128 wheels, verified on the wheel index; cu126 is the starting point, confirm-or-repin on first build). torch cu126 wheels bundle the CUDA runtime; the base provides the toolkit plus driver ABI (host driver must support CUDA 12.6). Python 3.11, workdir `/work`, ordered layers (copy `pyproject.toml` + `src`, `pip install .[train]`, `requirements-fast.txt` fused-kernel layer — install in the image; if the fused-kernel build fails for an arch, fall back to the pure-torch Mamba path as documented for T4 and record it in the build log); set `ENV CONFIG=configs/qlora-tiny.yaml`; expose the literal `docker build -t ghcr.io/sourcepirate/falcon-h1-tiny:dev .`, `docker run --gpus all`, and RunPod pull-and-run commands in comments. Arch proof: T4 smoke log plus 3090 pod `python -c "import torch; print(torch.cuda.get_arch_list())"` covering sm_75 + sm_86.

#### 3. `.dockerignore` (Create)

* **Type / Scope**: Build hygiene.
* **Proposed Changes**: Exclude `outputs/`, `data/processed/`, `.git`, `__pycache__`, `vendor/` build outputs to keep the 8-12 GB image from growing further. `llama.cpp` is a host-side export dependency (cloned at export time per item 12), never `COPY`d into the image, so excluding `vendor/` is safe; if the image ever must contain the converter, state the `COPY` exception here first.

#### 4. `src/cyberft/config.py` (Modify)

* **Type / Scope**: Validation rules.
* **Proposed Changes**: Demote missing `lm_head`/`out_proj` to a warning for all models now; keep `FORBIDDEN_LORA_TARGETS` (`out_proj`, `conv1d`) as the only hard fail, re-verified once against Tiny layer names after the Hub read. Add model-id conditioning only if the probe proves warn-only unsafe; 3B behavior otherwise byte-identical. Rationale: current validator cites 3B-only facts and would reject a correct Tiny setting.

#### 5. `src/cyberft/model.py` (Modify)

* **Type / Scope**: Model load and tokenizer.
* **Proposed Changes**: Record Hub facts first (template present/absent, reasoning delimiters, special tokens, Mamba/attention mix, vocab size). `build_tokenizer` keeps raise-unless-template until the read, then takes the named fallback if Tiny-R needs it; verify `_load_base` `llm_int8_skip_modules` and `attn_implementation` and lock Tiny values (`attn: sdpa` default, 4-bit on/off). No new public functions; constants/branches touched are listed here.

#### 6. `src/cyberft/loaders.py` (Modify only if probe forces it)

* **Type / Scope**: Corpus mapping, conditional.
* **Proposed Changes**: Keep the seven loader classes and exact-7-set check unchanged; adjust only if the trigger fires: over 5 percent of `kimi` rows dropped at `max_len` in smoke (then tune `max_chars` or row mapping). Any such edit bumps `LOADERS_VERSION` so the Arrow cache invalidates. Otherwise this item is YAML-only with no code change. Caps and lengths otherwise come from YAML, not code.

#### 7. `Makefile` (Modify)

* **Type / Scope**: Developer commands.
* **Proposed Changes**: Keep generic verbs driven by `CONFIG=`; keep `CONFIG`/`T4CONFIG` and the existing `smoke-t4`/`train-t4` targets untouched for 3B legacy; add no `smoke-tiny` wrappers. Tiny runs as `CONFIG=configs/qlora-tiny.yaml make smoke|train|eval|export`.

#### 8. `.github/workflows/ci.yml` (Modify)

* **Type / Scope**: CI build and publish.
* **Proposed Changes**: Keep `check` (ruff + pytest), `types` (mypy), `resolvable` jobs untouched. Add `image` job with `permissions: packages: write`: docker/login-action to GHCR, docker/setup-buildx-action, docker/build-push-action with `push: ${{ github.event_name != 'pull_request' }}`, `tags: ghcr.io/sourcepirate/falcon-h1-tiny:sha-${{ github.sha }}\n  ghcr.io/sourcepirate/falcon-h1-tiny:tiny\n  ghcr.io/sourcepirate/falcon-h1-tiny:latest` (the workflow `if:` fires on PRs, `main` pushes, and `v*` tags; PRs build with `push: false`, `main` and tags push), plus `cache-from: type=gha` / `cache-to: type=gha,mode=max`.

#### 9. `README.md` (Modify)

* **Type / Scope**: User docs.
* **Proposed Changes**: Mark `qlora-3b.yaml` files legacy; add Tiny quickstart (`CONFIG=configs/qlora-tiny.yaml make smoke/train/eval/export`), Docker build/run and RunPod pull commands, expected T4 vs 3090 behavior, and GGUF load check commands. Plain language, no new tooling.

#### 10. `notebooks/run.ipynb` (No change)

* **Type / Scope**: T4 entry point, explicitly no change.
* **Proposed Changes**: None. The notebook stays 3B-readable (Non-Goals); Tiny is served via CLI + Docker quickstart only.

#### 11. `tests/test_config.py` (Modify) + new Tiny fixture

* **Type / Scope**: Unit tests.
* **Proposed Changes**: Extend existing validator tests to cover the relaxed rule: update `test_missing_bnb_skip_modules_is_rejected` and `test_missing_lm_head_skip_is_rejected` to assert a `log.warning` naming the missing skip (not a raise); keep the `FORBIDDEN_LORA_TARGETS` rejection tests as hard fails. Tiny skip policy: exact `bnb_skip_modules` list recorded after the Hub read, default keep both skips until then. Add a test that `configs/qlora-tiny.yaml` loads, has 7 sources, and `eval_max_len <= max_len`; add a test that `model.quant: "bogus"` raises `ConfigError`.

#### 12. `vendor/llama.cpp` pin (Verify, no code)

* **Type / Scope**: Export dependency.
* **Proposed Changes**: Add `export.llama_cpp_rev: str | None = None` (non-empty when set; no other validation). The pin is consumed by the clone step (`git clone https://github.com/ggerganov/llama.cpp vendor/llama.cpp && git checkout <rev>` in the Dockerfile docs and CI), not by `export.py`, which keeps reading only `llama_cpp_dir`/`gguf_dir`/`quant`. TODO(implementer, before-export): fill the SHA under test; filenames derive from `quant.lower()`, so the plan uses `model-f16.gguf` and `model-q4_k_m.gguf` verbatim — copy-pasting `Q4_K_M` as a filename will 404. Give the literal merge, `convert_hf_to_gguf.py`, and `llama-quantize` commands plus both literal load commands from the Verification Plan. Contingency: next pin to try, or patch location under `export.llama_cpp_dir`, if conversion rejects the hybrid arch.

### Verification Plan

* **Build**: `docker build -t ghcr.io/sourcepirate/falcon-h1-tiny:dev .` passes; `docker run --gpus all` on T4 reaches `make info`.
* **Unit Tests**: `pytest` passes (includes new Tiny config test); existing CPU tests still skip GPU paths.
* **Static Analysis & Linting**: `ruff check .` and `mypy` pass with validator changes.
* **Smoke (T4)**: `CONFIG=configs/qlora-tiny.yaml make smoke` — 400 rows, 5 steps, finite loss, no `NonFiniteLossError`; same inside the container.
* **Full train (3090 RunPod)**: full `data` + `train` + `eval` finishes one epoch, `eval` reports zero non-finite rows.
* **Export (mandatory)**: `CONFIG=configs/qlora-tiny.yaml make export` produces `outputs/gguf-tiny/model-f16.gguf` and `outputs/gguf-tiny/model-q4_k_m.gguf`; `llama-cli -m outputs/gguf-tiny/model-f16.gguf -p "Write a tcpdump command for DNS only" --n-predict 64` and the same with `model-q4_k_m.gguf` each exit 0 with non-empty output. The other two sample generations use "Write the Linux shell command to list listening ports" and "Explain the difference between TCP SYN scan and connect scan for authorized testing".
* **CI**: PR shows the `image` job green with `push: false` plus `check`/`types`/`resolvable` green; tag `v*` push shows `image` with `push: true` and `docker manifest inspect ghcr.io/sourcepirate/falcon-h1-tiny:sha-<sha>` plus the `:tiny` and `:latest` tags visible.
* **`make info` gate**: `docker run --gpus all <image> CONFIG=configs/qlora-tiny.yaml make info` prints gpu, dtype, resolved config JSON, and corpus rows; missing `data/` cache must report `corpus: not built`, not crash.

---

*Drafted with EGM*
