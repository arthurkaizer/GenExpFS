"""
Exploratory analysis of CuMiDa gene-expression datasets.

Outputs
-------
results_v2/cumida_summary.csv          — per-dataset overview table
results_v2/figures/cumida_*.png        — one figure per analysis type
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

# ── CLI ───────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="CuMiDa EDA")
parser.add_argument("--datasets-path", default="datasets/cumida")
parser.add_argument("--results-path",  default="results_v2")
args = parser.parse_args()

DATASETS_DIR = Path(args.datasets_path)
RESULTS_DIR  = Path(args.results_path)
FIGURES_DIR  = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

PALETTE = "tab10"
plt.rcParams.update({"figure.dpi": 130, "font.size": 10})

# ── Load ──────────────────────────────────────────────────────────────────────

csv_paths = sorted(DATASETS_DIR.glob("*.csv"))
if not csv_paths:
    sys.exit(f"No CSV files found in {DATASETS_DIR}")

datasets: dict[str, dict] = {}
for path in csv_paths:
    df = pd.read_csv(path, index_col="samples")
    X  = df.drop(columns=["type"])
    y  = df["type"]
    datasets[path.stem] = {"X": X, "y": y}

names = list(datasets.keys())
print(f"Loaded {len(names)} datasets: {names}\n")

# ── 1. Summary table ──────────────────────────────────────────────────────────

rows = []
for name, d in datasets.items():
    X, y = d["X"], d["y"]
    vc   = y.value_counts()
    rows.append({
        "dataset":      name,
        "n_samples":    len(X),
        "n_features":   X.shape[1],
        "n_classes":    y.nunique(),
        "classes":      " | ".join(vc.index.tolist()),
        "class_counts": " | ".join(vc.astype(str).tolist()),
        "missing":      int(X.isna().sum().sum()),
        "mean_var":     round(float(X.var().mean()), 4),
    })

summary = pd.DataFrame(rows)
out_csv = RESULTS_DIR / "cumida_summary.csv"
summary.to_csv(out_csv, index=False)
print(summary.to_string(index=False))
print(f"\n[saved] {out_csv}\n")

# ── 2. Class distribution ─────────────────────────────────────────────────────

n = len(datasets)
fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), constrained_layout=True)
if n == 1:
    axes = [axes]

for ax, (name, d) in zip(axes, datasets.items()):
    vc = d["y"].value_counts()
    colors = sns.color_palette(PALETTE, len(vc))
    bars = ax.bar(vc.index, vc.values, color=colors, edgecolor="white", linewidth=0.6)
    for bar, val in zip(bars, vc.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
                str(val), ha="center", va="bottom", fontsize=8)
    short = name.split("_")[0]
    ax.set_title(short, fontsize=11, fontweight="bold")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=20)
    ax.spines[["top", "right"]].set_visible(False)

fig.suptitle("Class Distribution — CuMiDa Datasets", fontsize=13, fontweight="bold")
out = FIGURES_DIR / "cumida_class_distribution.png"
fig.savefig(out, bbox_inches="tight")
plt.close(fig)
print(f"[saved] {out}")

# ── 3. PCA (2-D) ──────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4), constrained_layout=True)
if n == 1:
    axes = [axes]

for ax, (name, d) in zip(axes, datasets.items()):
    X, y = d["X"].values, d["y"]
    pca  = PCA(n_components=2, random_state=42)
    Z    = pca.fit_transform(X)
    ev   = pca.explained_variance_ratio_ * 100

    le      = LabelEncoder().fit(y)
    labels  = le.transform(y)
    classes = le.classes_
    colors  = sns.color_palette(PALETTE, len(classes))

    for i, cls in enumerate(classes):
        mask = labels == i
        ax.scatter(Z[mask, 0], Z[mask, 1], s=20, alpha=0.7,
                   color=colors[i], label=cls, edgecolors="none")

    ax.set_xlabel(f"PC1 ({ev[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({ev[1]:.1f}%)")
    ax.set_title(name.split("_")[0], fontsize=11, fontweight="bold")
    ax.legend(fontsize=7, markerscale=1.4, framealpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)

fig.suptitle("PCA 2-D — CuMiDa Datasets", fontsize=13, fontweight="bold")
out = FIGURES_DIR / "cumida_pca.png"
fig.savefig(out, bbox_inches="tight")
plt.close(fig)
print(f"[saved] {out}")

# ── 4. Feature variance distribution ─────────────────────────────────────────

fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), constrained_layout=True)
if n == 1:
    axes = [axes]

for ax, (name, d) in zip(axes, datasets.items()):
    variances = d["X"].var().values
    ax.hist(variances, bins=60, color="#4C72B0", edgecolor="none", alpha=0.85)
    ax.axvline(np.median(variances), color="crimson", lw=1.2, linestyle="--",
               label=f"median={np.median(variances):.2f}")
    ax.set_title(name.split("_")[0], fontsize=11, fontweight="bold")
    ax.set_xlabel("Variance")
    ax.set_ylabel("# Features")
    ax.legend(fontsize=8, framealpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)

fig.suptitle("Feature Variance Distribution — CuMiDa Datasets", fontsize=13, fontweight="bold")
out = FIGURES_DIR / "cumida_feature_variance.png"
fig.savefig(out, bbox_inches="tight")
plt.close(fig)
print(f"[saved] {out}")

# ── 5. Top-20 variance features — boxplot per class ──────────────────────────

for name, d in datasets.items():
    X, y = d["X"], d["y"]
    top20 = X.var().nlargest(20).index.tolist()

    classes = sorted(y.unique())
    colors  = sns.color_palette(PALETTE, len(classes))

    fig = plt.figure(figsize=(18, 5), constrained_layout=True)
    fig.suptitle(f"Top-20 Variance Features — {name}", fontsize=12, fontweight="bold")

    for fi, feat in enumerate(top20):
        ax = fig.add_subplot(2, 10, fi + 1)
        data_per_class = [X.loc[y == cls, feat].values for cls in classes]
        bp = ax.boxplot(data_per_class, patch_artist=True, widths=0.55,
                        medianprops={"color": "black", "lw": 1.2},
                        whiskerprops={"lw": 0.8}, capprops={"lw": 0.8},
                        flierprops={"marker": ".", "markersize": 2, "alpha": 0.4})
        for patch, col in zip(bp["boxes"], colors):
            patch.set_facecolor(col)
            patch.set_alpha(0.75)
        short_feat = feat[:12] + "…" if len(feat) > 13 else feat
        ax.set_title(short_feat, fontsize=5.5)
        ax.set_xticks(range(1, len(classes) + 1))
        ax.set_xticklabels([c[:6] for c in classes], fontsize=5, rotation=30)
        ax.tick_params(axis="y", labelsize=5)
        ax.spines[["top", "right"]].set_visible(False)

    out = FIGURES_DIR / f"cumida_top20_boxplot_{name}.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out}")

# ── 6. Correlation heatmap — top-30 variance features ────────────────────────

for name, d in datasets.items():
    X = d["X"]
    top30 = X.var().nlargest(30).index.tolist()
    corr  = X[top30].corr()

    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, ax=ax, cmap="RdBu_r", center=0,
                vmin=-1, vmax=1, linewidths=0.3, linecolor="white",
                xticklabels=False, yticklabels=False,
                cbar_kws={"shrink": 0.7, "label": "Pearson r"})
    ax.set_title(f"Feature Correlation (top-30 variance) — {name}", fontsize=11, fontweight="bold")

    out = FIGURES_DIR / f"cumida_correlation_{name}.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out}")

# ── 7. Mean expression per class — heatmap (top-30) ──────────────────────────

for name, d in datasets.items():
    X, y = d["X"], d["y"]
    top30 = X.var().nlargest(30).index.tolist()

    means = X[top30].groupby(y).mean().T   # features × classes
    means_z = (means - means.mean(axis=1).values[:, None]) / (means.std(axis=1).values[:, None] + 1e-9)

    fig, ax = plt.subplots(figsize=(max(4, len(means_z.columns) * 1.5), 9), constrained_layout=True)
    sns.heatmap(means_z, ax=ax, cmap="RdBu_r", center=0,
                linewidths=0.4, linecolor="white",
                xticklabels=True, yticklabels=True,
                cbar_kws={"shrink": 0.6, "label": "z-score"})
    ax.set_title(f"Mean Expression per Class (z-score, top-30) — {name}", fontsize=10, fontweight="bold")
    ax.set_xlabel("Class")
    ax.set_ylabel("Feature")
    ax.tick_params(axis="x", rotation=20, labelsize=8)
    ax.tick_params(axis="y", labelsize=6)

    out = FIGURES_DIR / f"cumida_mean_expr_{name}.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out}")

print("\nDone.")
