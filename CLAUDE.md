# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt          # runtime
pip install -r requirements.txt -r requirements_dev.txt  # with dev tools (flake8, ipdb, rope)

# Lint
flake8                                   # max line length: 120

# Build datasets
python scripts/synthetic_datasets.py    # synthetic datasets
python scripts/xor_dataset.py           # XOR dataset
python scripts/download_cumida_datasets.py  # CuMiDa gene expression datasets

# Run the full pipeline (select → score → stability) with a preset
python src/main.py all --presets test_algorithms --datasets_path datasets --results-path results -vv

# Run only one stage
python src/main.py select  --presets test_algorithms --datasets_path datasets --results-path results --selection-filename my-run -vv
python src/main.py scoring --datasets_path datasets   --results-path results  --selection-filename my-run.csv --scoring-filename my-scoring -vv
python src/main.py stability --results-path results   --selection-filename my-run.csv --stability-filename my-stability --workers 2 -vv
```

All commands must be run from the repo root with `src/` implicitly on the Python path (the process sets it via `sys.path`; just `cd` to root first).

## Architecture

### Pipeline stages

`src/main.py` is the entry point. It runs up to four sequential stages driven by `--mode`:

1. **select** — fits feature selectors in parallel (`multiprocessing.Pool`) and appends results to a CSV.
2. **scoring** — loads the selection CSV, runs classifiers on the selected feature subsets, writes accuracy/F-measure/AUC scores.
3. **stability** — computes Jaccard, Hamming, Dice, Ochiai, POF, Kuncheva, Canberra, Spearman, Pearson across bootstrap/percent90 resampled runs.
4. **times** — aggregates execution times from the selection CSV.

### Presets

Presets (`src/util/presets/*.json`) declare which datasets and algorithms to run, with what parameters and how many repetitions. `task_creation_helper.py` converts a preset into a flat list of `Task` objects. `run_once: true` means the config block runs exactly once regardless of `-n`; `run_once: false` blocks repeat `-n` times. Each `params` entry is a positional-arg list passed to the selector constructor.

### Adding a new feature selector

1. Create `src/feature_selectors/my_selector.py` subclassing `BaseSelector` (or a base like `BaseEmbeddedFeatureSelector`).
2. Set `result_type = ResultType.WEIGHTS | RANK | SUBSET`.
3. Implement `fit(self, X, y)`: set `self._weights` / `self._rank` / `self._selected`, `self._support_mask`, and `self._fitted = True`.
4. Export from `src/feature_selectors/__init__.py`.
5. Register in `src/util/task_creation_helper.py` → `feature_selectors` dict with the string key used in preset JSON.

`BaseSelector.get_weights()` applies `minmax_scale` automatically; return raw weights from `fit()`.

### Result serialization

`TaskRunner.run()` serializes selector output to JSON and appends a row to the selection CSV via `ResultsWritter`. The row schema is defined in `src/results/model.py` (`Result` dataclass). Downstream stages load this CSV via `ResultsLoader` and dispatch on `result_type` to interpret `values`.

### Datasets

Datasets live under `datasets/` (gitignored). `DataLoader` normalizes features on load. `SharedDatasets` caches loaded datasets in shared memory across worker processes. Dataset name strings in preset JSON must match the keys in the `datasets_paths` dict in `main.py`.

### Stability sampling

Each `Task` carries a `sampling` field: `'none'` (full dataset), `'bootstrap'` (resample with replacement), or `'percent90'` (90 % subsample without replacement). The same preset config generates tasks for all three sampling modes; stability is then computed by comparing runs with the same `sampling` value.
