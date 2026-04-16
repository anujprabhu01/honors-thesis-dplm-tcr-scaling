#!/usr/bin/env python
"""Generate all scaling law figures for the TCR DPLM thesis.

Outputs (in results/figures/):
  scaling_law_2m.png   - L(N) at 2M dataset: pretrain + SFT power-law fits
  scaling_law_nd.png   - L(N) at all 4 data sizes (pretrain + SFT panels)
  ld_curves.png        - L(D) at all 5 model sizes (pretrain + SFT panels)
  eval_vs_n.png        - 4-panel eval metrics vs model size
  eval_vs_d.png        - 4-panel eval metrics vs dataset size

Data sources:
  wandb_export_2026-04-15T21_44_50.992-07_00.csv  (Name/val/nll_loss/params for all 40 runs)
  results/scored/tcr_{size}_{dataset}.csv          (eval metrics from ashour)

Usage:
    python scripts/scaling/plot_scaling.py
"""

from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).parents[2]
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
SCORED  = RESULTS / "scored"

# ── Size / dataset constants ──────────────────────────────────────────────────
SIZES       = ["0.1m", "1m", "5m", "15m", "50m"]
SIZE_LABELS = {"0.1m": "0.1M", "1m": "1M", "5m": "5M", "15m": "15M", "50m": "50M"}
SIZE_COLORS = {
    "0.1m": "#e41a1c",
    "1m":   "#ff7f00",
    "5m":   "#4daf4a",
    "15m":  "#377eb8",
    "50m":  "#984ea3",
}
# Fallback param counts (millions) when WandB export lacks model/params/total
SIZE_PARAMS = {"0.1m": 0.110, "1m": 0.946, "5m": 5.077, "15m": 14.752, "50m": 50.326}

DATASETS       = ["500k", "2m", "8m", "32m"]
DATASET_LABELS = {"500k": "500K", "2m": "2M", "8m": "8M", "32m": "32M"}
DATASET_SEQS   = {"500k": 0.5, "2m": 2.0, "8m": 8.0, "32m": 32.0}   # millions
DATASET_COLORS = {
    "500k": "#e41a1c",
    "2m":   "#ff7f00",
    "8m":   "#4daf4a",
    "32m":  "#377eb8",
}

EVAL_METRICS = [
    ("bap_cnn",                "BAP CNN Score",   "higher = better"),
    ("bap_lstm",               "BAP LSTM Score",  "higher = better"),
    ("tcr_match",              "TCRMatch Score",  "higher = better"),
    ("tcrbert_mlm_ll_masking", "TCR-BERT PLL",    "higher = better"),
]

plt.rcParams.update({"font.size": 11, "axes.titlesize": 13, "axes.labelsize": 11})


# ── Parsing ───────────────────────────────────────────────────────────────────
def parse_run_name(name: str):
    """Return (size_str, dataset_str, is_sft) or None.

    Handled patterns:
      tcr_50m          → ("50m", "2m",   False)  Phase-1 style (2m implied)
      tcr_sft_50m      → ("50m", "2m",   True)
      tcr_50m_32m      → ("50m", "32m",  False)
      tcr_sft_0.1m_8m  → ("0.1m", "8m", True)
      tcr_50m_8m_lr_high → ("50m", "8m", False)  LR sanity-check run
    """
    name = str(name).strip()
    name = re.sub(r"_lr_(high|low)$", "", name)   # strip LR sanity suffixes

    is_sft = name.startswith("tcr_sft_")
    stem = name[len("tcr_sft_"):] if is_sft else name[len("tcr_"):]

    for size in SIZES:
        if stem == size:
            return size, "2m", is_sft          # Phase-1 style → 2M implied
        for dataset in DATASETS:
            if stem == f"{size}_{dataset}":
                return size, dataset, is_sft
    return None


# ── Math ──────────────────────────────────────────────────────────────────────
def power_law(x, a, alpha, c):
    return a * np.power(x, -alpha) + c


def fit_power_law(x, y):
    try:
        popt, _ = curve_fit(
            power_law, np.array(x, dtype=float), np.array(y, dtype=float),
            p0=[1.0, 0.3, float(np.min(y)) * 0.9],
            bounds=([0, 0.01, -np.inf], [np.inf, 2.0, np.inf]),
            maxfev=20000,
        )
        return popt
    except Exception:
        return None


# ── Data loading ──────────────────────────────────────────────────────────────
def load_wandb_summary() -> pd.DataFrame:
    """Load all compact wandb_export_*.csv files.

    Returns DataFrame(size_str, dataset, is_sft, n_params, val_nll_loss).

    Searches both the project root and the 'wandb exports/' subdirectory.
    Files processed in chronological (filename-sorted) order; later files
    overwrite earlier ones for the same (size, dataset, is_sft) key, so the
    most-recent export wins. Min-aggregation exports ("val/nll_loss (Min)")
    are preferred over last-step exports ("val/nll_loss") when both exist.
    Skips step-by-step exports (no "Name" column).
    """
    # Accepted val/nll_loss column names, in preference order (Min first)
    VAL_COL_CANDIDATES = ("val/nll_loss (Min)", "val/nll_loss")

    # Collect CSVs from root and 'wandb exports/' subdirectory, sort by filename
    csv_paths = sorted(
        list(ROOT.glob("wandb_export_*.csv"))
        + list((ROOT / "wandb exports").glob("wandb_export_*.csv")),
        key=lambda p: p.name,
    )

    records: dict = {}
    for csv_path in csv_paths:
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        if "Name" not in df.columns:
            continue
        val_col = next((c for c in VAL_COL_CANDIDATES if c in df.columns), None)
        if val_col is None:
            continue
        for _, row in df.iterrows():
            parsed = parse_run_name(str(row["Name"]))
            if parsed is None:
                continue
            size_str, dataset, is_sft = parsed
            try:
                val_loss = float(row[val_col])
                if np.isnan(val_loss):
                    continue
                n_params = (float(row["model/params/total"]) / 1e6
                            if "model/params/total" in row.index else SIZE_PARAMS[size_str])
            except (ValueError, TypeError):
                continue
            key = (size_str, dataset, is_sft)
            records[key] = dict(
                size_str=size_str, dataset=dataset, is_sft=is_sft,
                n_params=n_params, val_nll_loss=val_loss,
            )
    return pd.DataFrame(list(records.values()))


def load_eval_scores() -> pd.DataFrame:
    """Load results/scored/tcr_{size}_{dataset}.csv → mean metrics per pair.

    Returns DataFrame(size_str, dataset, n_params, bap_cnn, bap_lstm,
                       tcr_match, tcrbert_mlm_ll_masking).
    """
    records = []
    for fpath in sorted(SCORED.glob("tcr_*.csv")):
        stem = fpath.stem[4:]     # strip leading "tcr_"
        parsed = None
        for size in SIZES:
            for dataset in DATASETS:
                if stem == f"{size}_{dataset}":
                    parsed = (size, dataset)
                    break
            if parsed:
                break
        if parsed is None:
            continue
        size_str, dataset = parsed
        try:
            df = pd.read_csv(fpath)
        except Exception:
            continue
        record: dict = {"size_str": size_str, "dataset": dataset,
                        "n_params": SIZE_PARAMS[size_str]}
        for col, _, _ in EVAL_METRICS:
            if col in df.columns:
                record[col] = df[col].mean()
        records.append(record)
    return pd.DataFrame(records)


# ── Figure helpers ────────────────────────────────────────────────────────────
def _save(fig: plt.Figure, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ── Figure 1: Classic L(N) at 2M ─────────────────────────────────────────────
def plot_scaling_law_2m(wdb: pd.DataFrame, outpath: Path):
    """L(N) at 2M dataset — pretrain + SFT side by side with power-law fits."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    fig.suptitle("Scaling Law: Validation Loss vs Model Size (2M Dataset)",
                 fontsize=14, fontweight="bold")

    for ax, is_sft, title, color in [
        (axes[0], False, "Pretraining", "#377eb8"),
        (axes[1], True,  "SFT",         "#e41a1c"),
    ]:
        sub = (wdb[(wdb["dataset"] == "2m") & (wdb["is_sft"] == is_sft)]
               .sort_values("n_params"))
        if sub.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, color="gray")
            ax.set_title(title)
            continue

        sizes  = sub["n_params"].values
        losses = sub["val_nll_loss"].values
        ax.scatter(sizes, losses, color=color, s=80, zorder=5)
        for _, row in sub.iterrows():
            ax.annotate(SIZE_LABELS[row["size_str"]], (row["n_params"], row["val_nll_loss"]),
                        textcoords="offset points", xytext=(6, 4), fontsize=9, color="gray")

        popt = fit_power_law(sizes, losses)
        if popt is not None:
            a, alpha, c = popt
            xs = np.logspace(np.log10(sizes.min() * 0.7), np.log10(sizes.max() * 1.5), 300)
            ax.plot(xs, power_law(xs, *popt), "--", color=color, alpha=0.7, linewidth=1.5,
                    label=rf"$L(N)={a:.3f}\cdot N^{{-{alpha:.3f}}}+{c:.3f}$")
            ax.legend(fontsize=9)
            print(f"{title} 2M: a={a:.4f}, α={alpha:.4f}, c={c:.4f}")

        ax.set_xscale("log")
        ax.set_xlabel("Model Parameters")
        ax.set_title(f"{title}: L(N) at 2M")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Val NLL Loss")
    plt.tight_layout()
    _save(fig, outpath)


# ── Figure 4: L(N) at all datasets ───────────────────────────────────────────
def plot_scaling_nd(wdb: pd.DataFrame, outpath: Path):
    """L(N) curves at each data size — pretrain + SFT panels."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Val Loss vs Model Size at Different Data Sizes",
                 fontsize=14, fontweight="bold")

    for ax, is_sft, title in [
        (axes[0], False, "Pretrain"),
        (axes[1], True,  "SFT"),
    ]:
        for dataset in DATASETS:
            sub = (wdb[(wdb["dataset"] == dataset) & (wdb["is_sft"] == is_sft)]
                   .sort_values("n_params"))
            if sub.empty:
                continue
            sizes  = sub["n_params"].values
            losses = sub["val_nll_loss"].values
            c = DATASET_COLORS[dataset]
            ax.scatter(sizes, losses, color=c, s=50, zorder=5)
            if len(sizes) >= 3:
                popt = fit_power_law(sizes, losses)
                if popt is not None:
                    a, alpha, _ = popt
                    xs = np.logspace(np.log10(sizes.min() * 0.7),
                                     np.log10(sizes.max() * 1.5), 300)
                    ax.plot(xs, power_law(xs, *popt), "--", color=c, alpha=0.7,
                            linewidth=1.5,
                            label=rf"D={DATASET_LABELS[dataset]} ($\alpha$={alpha:.3f})")
                    print(f"L(N) {title} D={DATASET_LABELS[dataset]}: "
                          f"a={a:.4f}, α={alpha:.4f}")
                    continue
            # Fallback: plain line without fit
            ax.plot(sizes, losses, "o--", color=c, alpha=0.7, linewidth=1.5,
                    label=DATASET_LABELS[dataset])

        ax.set_xscale("log")
        ax.set_xlabel("Model Parameters (M)")
        ax.set_ylabel("Best Val Loss")
        ax.set_title(f"{title}: L(N) at fixed D")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        if not ax.get_lines() and not ax.collections:
            ax.text(0.5, 0.5, "No data yet", ha="center", va="center",
                    transform=ax.transAxes, color="gray")

    plt.tight_layout()
    _save(fig, outpath)


# ── Figure 5: L(D) curves ─────────────────────────────────────────────────────
def plot_ld_curves(wdb: pd.DataFrame, outpath: Path):
    """L(D) curves — one curve per model size, pretrain + SFT panels."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Val Loss vs Data Size at Different Model Sizes",
                 fontsize=14, fontweight="bold")

    for ax, is_sft, title in [
        (axes[0], False, "Pretrain"),
        (axes[1], True,  "SFT"),
    ]:
        for size in SIZES:
            sub = (wdb[(wdb["size_str"] == size) & (wdb["is_sft"] == is_sft)]
                   .copy())
            sub["d_seqs"] = sub["dataset"].map(DATASET_SEQS)
            sub = sub.sort_values("d_seqs")
            if sub.empty:
                continue
            d_arr = sub["d_seqs"].values
            l_arr = sub["val_nll_loss"].values
            c = SIZE_COLORS[size]
            ax.scatter(d_arr, l_arr, color=c, s=50, zorder=5)
            if len(d_arr) >= 3:
                popt = fit_power_law(d_arr, l_arr)
                if popt is not None:
                    b, beta, _ = popt
                    xs = np.logspace(np.log10(d_arr.min() * 0.7),
                                     np.log10(d_arr.max() * 1.5), 300)
                    ax.plot(xs, power_law(xs, *popt), "--", color=c, alpha=0.7,
                            linewidth=1.5,
                            label=rf"{SIZE_LABELS[size]} ($\beta$={beta:.3f})")
                    print(f"L(D) {title} N={SIZE_LABELS[size]}: "
                          f"b={b:.4f}, β={beta:.4f}")
                    continue
            ax.plot(d_arr, l_arr, "o--", color=c, alpha=0.7, linewidth=1.5,
                    label=SIZE_LABELS[size])

        ax.set_xscale("log")
        ax.set_xlabel("Dataset Size (M sequences)")
        ax.set_ylabel("Best Val Loss")
        ax.set_title(f"{title}: L(D) at fixed N")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        if not ax.get_lines() and not ax.collections:
            ax.text(0.5, 0.5, "No data yet", ha="center", va="center",
                    transform=ax.transAxes, color="gray")

    plt.tight_layout()
    _save(fig, outpath)


# ── Figure 6: Eval metrics vs N ───────────────────────────────────────────────
def plot_eval_vs_n(eval_df: pd.DataFrame, outpath: Path):
    """4-panel eval metrics vs model size — one line per dataset."""
    if eval_df.empty:
        print("No eval data — skipping eval_vs_n")
        return
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Evaluation Metrics vs Model Size", fontsize=15, fontweight="bold")

    for ax, (col, label, note) in zip(axes.flat, EVAL_METRICS):
        if col not in eval_df.columns:
            ax.text(0.5, 0.5, f"No {col}", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(label)
            continue
        for dataset in DATASETS:
            sub = eval_df[eval_df["dataset"] == dataset].sort_values("n_params")
            if sub.empty or col not in sub.columns:
                continue
            ax.plot(sub["n_params"], sub[col], "o--",
                    color=DATASET_COLORS[dataset], linewidth=1.5, markersize=6,
                    label=DATASET_LABELS[dataset])
        ax.set_xscale("log")
        ax.set_xlabel("Model Parameters (M)")
        ax.set_ylabel(f"Mean {label}")
        ax.set_title(f"{label}\n({note})")
        ax.legend(title="Dataset size", fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save(fig, outpath)


# ── Figure 7: Eval metrics vs D ───────────────────────────────────────────────
def plot_eval_vs_d(eval_df: pd.DataFrame, outpath: Path):
    """4-panel eval metrics vs dataset size — one line per model size."""
    if eval_df.empty:
        print("No eval data — skipping eval_vs_d")
        return
    eval_df = eval_df.copy()
    eval_df["d_seqs"] = eval_df["dataset"].map(DATASET_SEQS)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Evaluation Metrics vs Dataset Size", fontsize=15, fontweight="bold")

    for ax, (col, label, note) in zip(axes.flat, EVAL_METRICS):
        if col not in eval_df.columns:
            ax.text(0.5, 0.5, f"No {col}", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(label)
            continue
        for size in SIZES:
            sub = eval_df[eval_df["size_str"] == size].sort_values("d_seqs")
            if sub.empty or col not in sub.columns:
                continue
            ax.plot(sub["d_seqs"], sub[col], "o--",
                    color=SIZE_COLORS[size], linewidth=1.5, markersize=6,
                    label=SIZE_LABELS[size])
        ax.set_xscale("log")
        ax.set_xlabel("Dataset Size (M sequences)")
        ax.set_ylabel(f"Mean {label}")
        ax.set_title(f"{label}\n({note})")
        ax.legend(title="Model size", fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save(fig, outpath)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    FIGURES.mkdir(parents=True, exist_ok=True)

    # ── WandB summary (all 40 Phase-2 runs) ───────────────────────────────────
    wdb = load_wandb_summary()
    print(f"\nLoaded {len(wdb)} WandB runs from compact exports")
    if not wdb.empty:
        summary = wdb.groupby(["is_sft", "dataset"])["size_str"].apply(list)
        print(summary.to_string())

    # ── Eval scores from scored CSVs ──────────────────────────────────────────
    eval_df = load_eval_scores()
    print(f"\nLoaded eval scores for {len(eval_df)} (size, dataset) pairs")
    if not eval_df.empty:
        print(eval_df[["size_str", "dataset"]].to_string(index=False))

    print()

    # ── Generate figures ──────────────────────────────────────────────────────
    if not wdb.empty:
        plot_scaling_law_2m(wdb, FIGURES / "scaling_law_2m.png")
        plot_scaling_nd(wdb,     FIGURES / "scaling_law_nd.png")
        plot_ld_curves(wdb,      FIGURES / "ld_curves.png")
    else:
        print("No compact WandB data found — skipping scaling law figures")

    if not eval_df.empty:
        plot_eval_vs_n(eval_df, FIGURES / "eval_vs_n.png")
        plot_eval_vs_d(eval_df, FIGURES / "eval_vs_d.png")
    else:
        print("No eval scores found — skipping eval metric figures")


if __name__ == "__main__":
    main()
