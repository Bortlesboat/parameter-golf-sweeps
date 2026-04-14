# Parameter Golf Sweeps

Run small, repeatable `openai/parameter-golf` ablation sweeps from a simple JSON file.

This repo is for the common "try 3-10 training variants and compare the footer metrics" workflow without dragging in a larger experiment platform.

## What it does

- runs each sweep entry sequentially from one config file
- writes one log file per run
- parses the official training-log footer lines when they exist
- writes `summary.csv` and `summary.md`
- ranks runs by best `val_bpb`

## Example

```bash
python run_sweeps.py --config examples/dev-sp1024-sweep.json
```

That produces an output folder with:

- per-run log files
- `summary.csv`
- `summary.md`

## Config shape

```json
{
  "workdir": "/workspace/parameter-golf",
  "log_dir": "sweep_runs",
  "command": ["torchrun", "--standalone", "--nproc_per_node=1", "train_gpt.py"],
  "base_env": {
    "DATA_PATH": "/workspace/parameter-golf/data/datasets/fineweb10B_sp1024",
    "TOKENIZER_PATH": "/workspace/parameter-golf/data/tokenizers/fineweb_1024_bpe.model",
    "VOCAB_SIZE": "1024",
    "MAX_WALLCLOCK_SECONDS": "1800",
    "VAL_LOSS_EVERY": "200",
    "TRAIN_LOG_EVERY": "50"
  },
  "runs": [
    {
      "name": "seq1024_batch8192",
      "env": {
        "RUN_ID": "seq1024_batch8192",
        "TRAIN_BATCH_TOKENS": "8192",
        "SEQ_LEN": "1024"
      }
    }
  ]
}
```

## Notes

- The runner assumes the current official log footer format for `val_loss`, `val_bpb`, and total bytes.
- If a run exits non-zero or never prints the footer lines, the summary still includes it and leaves those metric columns blank.
- For bootstrapping a RunPod pod or checking a finished record folder, use the companion repos:
  - `https://github.com/Bortlesboat/parameter-golf-runpod-starter`
  - `https://github.com/Bortlesboat/parameter-golf-size-checker`
