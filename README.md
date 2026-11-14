# Falcon-H1-3B Cyber QLoRA

[![CI](https://github.com/sourcepirate/falcon-h1-finetune/actions/workflows/ci.yml/badge.svg)](https://github.com/sourcepirate/falcon-h1-finetune/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![Model](https://img.shields.io/badge/model-Falcon--H1--3B-yellow)](https://huggingface.co/tiiuae/Falcon-H1-3B-Instruct)

Fine-tune [Falcon-H1-3B-Instruct](https://huggingface.co/tiiuae/Falcon-H1-3B-Instruct)
on seven public red-team / cyber-security datasets with QLoRA — one precision
end-to-end, no AMP, no dtype plumbing.

```bash
pip install -e ".[train]"
make smoke   # 400 rows -> 5 steps. Run this before anything else.
make train   # full run -> outputs/falcon-h1-3b-cyber
make eval    # per-sample loss, worst offenders, non-finite detection
make export  # merge -> f16 GGUF -> Q4_K_M
```

## How it works

```mermaid
flowchart LR
    YAML[configs/*.yaml] --> LOAD[loaders: one class per dataset]
    LOAD --> CACHE[Arrow cache + manifest.json]
    CACHE --> TRAIN[Trainer, bf16=False, fp16=False]
    TRAIN --> ADAPTER[LoRA adapter]
    ADAPTER --> EVAL[eval: per-row loss report]
    ADAPTER --> EXPORT[merge -> GGUF Q4_K_M]
```

Seven loader classes (`RedTeam`, `Kali`, `Kimi`, `CTI`, `ExploitDB`,
`LinuxCommands`, `Debian`) share one `LoaderBase`: fetch → cap → map rows to
chat messages → render with the model's chat template → dedupe → length-filter →
split → tokenise. The corpus is fingerprinted, so a config change invalidates
the cache instead of silently training on stale rows.

## Datasets

| Loader | Source | Rows (cap) |
|---|---|---|
| `RedTeam` | [WNT3D/Ultimate-Offensive-Red-Team](https://huggingface.co/datasets/WNT3D/Ultimate-Offensive-Red-Team) | 20,000 |
| `Kali` | [suryanshp1/kali-linux-pentesting-data](https://huggingface.co/datasets/suryanshp1/kali-linux-pentesting-data) | 343 |
| `Kimi` | [echel0nn1881/kimi-cyber-reasoning](https://huggingface.co/datasets/echel0nn1881/kimi-cyber-reasoning) | 997 |
| `CTI` | [reloading0101/threat-intelligence-dataset](https://huggingface.co/datasets/reloading0101/threat-intelligence-dataset) | 6,000 |
| `ExploitDB` | [Waiper/ExploitDB_DataSet](https://huggingface.co/datasets/Waiper/ExploitDB_DataSet) | 8,000 |
| `LinuxCommands` | [Romit2004/LinuxCommands](https://huggingface.co/datasets/Romit2004/LinuxCommands) | 6,000 |
| `Debian` | [2etatg/Debian_Hacking_Networking](https://huggingface.co/datasets/2etatg/Debian_Hacking_Networking) | 965 |

## Configs

| File | For | Precision | `max_len` |
|---|---|---|---|
| `configs/qlora-tiny.yaml` | T4 smoke + RTX 3090 full train (Tiny-R-90M) | auto (fp16 T4 / bf16 Ampere) | 1024 |
| `configs/qlora-3b.yaml` | Ampere+ (A100, L4, RTX 30xx/40xx), legacy | bf16 | 2048 |
| `configs/qlora-3b-t4.yaml` | Colab T4, legacy | fp16 | 1024 |

LoRA: `r=32, alpha=16, dropout=0.05`, targets `q/k/v/o/in/gate/up/down_proj`.
Training: 4-bit NF4 base, `paged_adamw_8bit`, `lr=2e-4`, 1 epoch, cosine schedule.
`notebooks/run.ipynb` is the T4 entry point (`Run All` with `RUN_TRAIN = False` is safe).

## Tiny-R-90M quickstart

The 3B configs stay readable as legacy. Production runs use the `cyberft` CLI
directly with `-c` (no `make` needed):

```bash
C=configs/qlora-tiny.yaml
python -m cyberft.cli -c $C info    # verify GPU, dtype, resolved config
python -m cyberft.cli -c $C data    # build corpus (400 rows + --smoke for the 5-step T4 gate)
python -m cyberft.cli -c $C train   # full 1-epoch run (add --smoke for the T4 gate)
python -m cyberft.cli -c $C eval    # per-row loss gate; exit 1 on any non-finite row
python -m cyberft.cli -c $C export  # f16 + Q4_K_M, both must load
```

Container (one image, sm_75 + sm_86; over 12 GB fails this pass):

```bash
docker build -t ghcr.io/sourcepirate/falcon-h1-tiny:dev .
docker run --gpus all ghcr.io/sourcepirate/falcon-h1-tiny:dev python -m cyberft.cli -c configs/qlora-tiny.yaml train --smoke
# RunPod 3090: pull ghcr.io/sourcepirate/falcon-h1-tiny:tiny, same CLI commands
```

GGUF load check:

```bash
llama-cli -m outputs/gguf-tiny/model-f16.gguf -p "Write a tcpdump command for DNS only" --n-predict 64
llama-cli -m outputs/gguf-tiny/model-q4_k_m.gguf -p "Write a tcpdump command for DNS only" --n-predict 64
```

## CLI

No command requires a flag (`--config`, or `-c`, defaults to `configs/qlora-3b.yaml`):

```bash
python -m cyberft.cli info      # gpu, dtype, resolved config, corpus stats
python -m cyberft.cli data      # --smoke, --force
python -m cyberft.cli train     # --smoke
python -m cyberft.cli eval      # --smoke
python -m cyberft.cli export
```

## Library

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

Failures are typed exceptions (`ConfigError`, `NotBuiltError`, `DependencyError`,
`NonFiniteLossError`) — the library never parses arguments or calls `sys.exit`.

## Operating notes

- **Throughput** (T4, PyTorch Mamba fallback, no `mamba-ssm`): ~0.6 samples/s.
  Ampere with fused kernels is roughly 4–5× faster.
- **Resume**: `train.resume: true` picks up the newest checkpoint automatically.
- **Memory**: 4-bit base + LoRA + gradient checkpointing fits 16 GB at
  `micro_batch: 1`, `max_len: 2048`. Raise `micro_batch` before `max_len`.
- **Export RAM**: the CPU bf16 merge needs ~7 GB of host RAM.
- **CI**: ruff, mypy and the CPU tests run on every push; the `image` job builds on PRs (no push) and pushes `:sha`/`:tiny`/`:latest` to GHCR on `main` pushes and `v*` tags. The GPU smoke test
  stays manual (`python -m cyberft.cli -c configs/qlora-tiny.yaml train --smoke` in the container).

## Troubleshooting

| Symptom | Cause | Action |
|---|---|---|
| `mat1 and mat2 shapes cannot be multiplied` in `mamba_split_conv1d_scan_combined` | Mamba `out_proj` was quantised and fed to the fused kernel | keep `out_proj` in `model.bnb_skip_modules` |
| `NonFiniteLossError` during training | loss went NaN at that step | stop; lower `lr`, check `max_len` |
| `NonFiniteLossError` during evaluation | one pathological row | `make eval` names the row; drop it in the loader class in `cyberft/loaders.py` |
| `NonFiniteLossError` on a T4 (fp16) | fp16 SSM arithmetic on a long row | lower `max_len` (try 768), then `lr` to 1e-4 |
| `No kernel image is available` from `mamba-ssm` | wrong CUDA arch | reinstall `requirements-fast.txt` on the target GPU |
| `ImportError: ... requires bitsandbytes` | runtime installed without the `train` extra | `pip install -e ".[train]"`, restart the kernel |

## Licence note

`reloading0101/threat-intelligence-dataset` is **CC-BY-NC-SA-4.0** (non-commercial);
check every source before commercial use. The base model ships under the Falcon-LLM
licence. Use the result only for authorised testing, research and education.
