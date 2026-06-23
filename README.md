# Analysis and Comparison of Feature Selection Methods Towards Performance and Stability

> This project aims to evaluate and compare Feature Selection algorithms. 

The amount of gathered data is increasing at unprecedented rates for machine learning applications
such as natural language processing, computer vision, and bioinformatics. This increase implies a
higher number of samples and features; thus, some problems regarding highly dimensioned data
arise. The curse of dimensionality, small samples, noisy or redundant features, and biased data
are among them. Feature selection is fundamental to dealing with such problems. It reduces the
data dimensionality by selecting the most relevant and less redundant features. Thus, reducing the
computational cost, improving accuracy, and enhancing the data’s interpretability to machine learning
models and domain experts. However, there are several selectors options from which to choose.
This work compares some of the most representative algorithms from different feature selection
groups regarding a broad range of measures, several datasets, and different strategies from diverse
perspectives. We employ metrics to appraise selection accuracy, selection redundancy, prediction
performance, algorithmic stability, selection reliability, and computational time of several feature
selection algorithms. The results highlight the strengths and weaknesses of these algorithms and can
guide their application.

## Deep Feature Selection Extension

This repository extends the original framework with deep learning-based feature
selectors, following the future-work direction stated in the paper. The goal is
to compare these methods against the classical algorithms using the same
evaluation pipeline (accuracy, stability, redundancy, execution time).

### Methods

All five selectors are in `src/feature_selectors/` and inherit from
`BaseDeepSelector` (`src/feature_selectors/base_models/deep_selector.py`).
They all produce `ResultType.WEIGHTS` and are fully compatible with the
existing scoring and stability pipeline.

| Selector | Key | Supervised | Importance source |
|---|---|---|---|
| `AutoencoderFeatureSelector` | `Autoencoder` | No | L1 norm of first encoder layer weights |
| `VAEFeatureSelector` | `VAE` | No | L1 norm of first encoder layer weights |
| `SensitivityFeatureSelector` | `Sensitivity` | Yes | Mean absolute gradient of predicted score w.r.t. input |
| `IntegratedGradientsFeatureSelector` | `IntegratedGradients` | Yes | Integrated Gradients (Sundararajan et al., 2017) |
| `ChannelAttentionFeatureSelector` | `ChannelAttention` | Yes | Learned softmax attention weights over features |

**Supervised vs unsupervised:** `Autoencoder` and `VAE` do not use class
labels; they learn which features are structurally important for reconstruction.
The other three methods use labels to identify features that are discriminative
for classification.

**Sensitivity vs Integrated Gradients:** both compute gradients of the
predicted class score w.r.t. inputs. IG additionally integrates along a path
from a zero baseline, which avoids under-attribution in saturated network
regions and satisfies the completeness axiom.

**Channel Attention:** learns one scalar weight per feature via a softmax over
all features. This is per-feature (channel) attention, not the sample-specific
self-attention used in transformers.

### Presets

| Preset | Datasets | Purpose |
|---|---|---|
| `compare_deep` | Prostate GSE6919 | original single-dataset baseline |
| `compare_deep_multi` | Prostate GSE6919 + Breast GSE70947 | multi-dataset comparison (v3) |
| `compare_deep_breast` | Breast GSE70947 | isolated breast experiments |
| `seed_ablation` | Prostate GSE6919 | seed variance analysis |

### Pipeline extensions

- **MLP downstream classifier** — `MLPClassifier(128→64, relu, early_stopping)` added alongside SVM/RF/DT/NB/ZeroR
- **Statistical tests** — `src/evaluation/statistical_tests.py`: Friedman + Nemenyi + Wilcoxon pairwise; exposed as `main.py stats` mode
- **Dataset filter** — `--dataset <name>` flag on scoring/stability/times/stats modes to scope analysis to one dataset
- **Resume support** — interrupted `select` runs resume from where they stopped without reprocessing completed rows
- **Stability std** — stability CSV now includes `*_std` columns (jaccard, kuncheva, spearman, pearson) alongside means

### Reproducing the comparison (v3)

```bash
# 1. Download datasets
make download-cumida

# 2. Smoke test (~5 min)
make run-test

# 3. Full comparison — Prostate + Breast, all 9 selectors
#    Runs select → scoring → stability → times sequentially.
#    --workers 1 avoids GPU memory contention between parallel deep processes.
make run-v3

# 4. Statistical tests (Friedman, Nemenyi, Wilcoxon)
make run-v3-stats

# 5. Analysis figures (filtered to Prostate GSE6919)
make analyze-v3
```

Results are written to `results_v3/`. Each stage produces a CSV:
- `compare-selection.csv` — feature rankings and execution times _(gitignored, large)_
- `compare-scoring.csv` / `compare-scoring-complete.csv` — Macro F1 per method and k _(complete version gitignored)_
- `compare-stability-bootstrap.csv` / `compare-stability-90perecent.csv` — Jaccard and other stability metrics
- `compare-stats-friedman/nemenyi/wilcoxon.csv` — statistical test results
- `results_v3/figures/` — plots generated by `make analyze-v3`

### Results summary (Prostate GSE6919, `sampling=none`)

| Method | Peak F1 | k | Jaccard@200 | T (s) |
|---|---|---|---|---|
| SVM-RFE | **0.974** | 50 | 0.153 | 778 |
| Lasso | 0.908 | 50 | **0.659** | 0.7 |
| Int. Gradients | 0.883 | 200 | 0.029 | 11 |
| Sensitivity | 0.878 | 200 | 0.025 | 10 |
| Random Forest | 0.807 | 200 | 0.034 | 0.3 |
| KW Filter | 0.802 | 50 | 0.251 | 8 |
| Channel Attention | 0.755† | 100 | 0.031 | 13 |
| VAE | 0.752 | 100 | 0.008 | 9 |
| Autoencoder | 0.703 | 200 | 0.019 | 34 |

† High variance across seeds (individual peak 0.878, mean 0.755).

Nemenyi post-hoc identified 5 significant pairs (p < 0.05): SVM-RFE vs {AE, VAE, ChanAttn} and Lasso vs {AE, VAE}.

### Documentation

Article and presentation (LaTeX) are in `docs/`.

## How to cite

If you use our code, methods, or results in your research, please consider citing the main publication:

- Matheus Cezimbra Barbieri, Bruno Iochins Grisci, Marcio Dorn. _Analysis and Comparison of Feature Selection Methods Towards Performance and Stability_, **Expert Systems with Applications**, 123667, March **2024**, DOI: [https://doi.org/10.1016/j.eswa.2024.123667](https://doi.org/10.1016/j.eswa.2024.123667)

Bibtex entry:
```
@article{barbieri2024analysis,
  title={Analysis and comparison of feature selection methods towards performance and stability},
  author={Barbieri, Matheus Cezimbra and Grisci, Bruno Iochins and Dorn, M{\'a}rcio},
  journal={Expert Systems with Applications},
  pages={123667},
  year={2024},
  publisher={Elsevier}
}

```

## Contact information

- [Matheus Cezimbra Barbieri](https://orcid.org/0000-0002-5389-7064)

    - mcbarbieri@inf.ufrgs.br

- [Bruno I. Grisci](https://orcid.org/0000-0003-4083-5881) - Associate Professor ([Institute of Informatics](https://www.inf.ufrgs.br/site/en) - [UFRGS](http://www.ufrgs.br/english/home))

    - bigrisci@inf.ufrgs.br

- [Dr. Marcio Dorn](https://orcid.org/0000-0001-8534-3480) - Associate Professor ([Institute of Informatics](https://www.inf.ufrgs.br/site/en) - [UFRGS](http://www.ufrgs.br/english/home))

    - mdorn@inf.ufrgs.br

- http://sbcb.inf.ufrgs.br/
