#!/usr/bin/env python3
"""Comprehensive analysis of Mamba-MoE experiments across feature extractions and configurations.

Parses logs/ for all summary JSONs, jsonl epoch logs, and runner.log failures.
Produces statistical tables and publication-ready graphs in analysis/graphs/.
"""
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

# ── Configuration ────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent   # analysis/
LOG_DIR = BASE_DIR / "logs"
GRAPH_DIR = BASE_DIR / "graphs"
GRAPH_DIR.mkdir(exist_ok=True)

TEXT_BACKBONES = {
    "TFE-A": "RoBERTa-Base", "TFE-B": "RoBERTa-Large",
    "TFE-C": "DeBERTa-v3",   "TFE-D": "BERT-Large",
    "TFE-E": "Electra-Large",
}
AUDIO_BACKBONES = {"AFE-B": "WavLM-Large", "AFE-C": "AFE-C-1024"}
VIDEO_BACKBONES = {"VFE-A": "ResNet50", "VFE-B": "VFE-B-1792"}
MODEL_LABELS = {
    "variant1": "MoE Mamba Experts",
    "variant2": "Modality Specialists",
    "variant3": "Hierarchical MoE",
    "variant4": "Context Conditioned",
}
MODEL_ORDER = ["variant1", "variant2", "variant3", "variant4"]

sns.set_theme(style="whitegrid", context="talk", palette="Set2")
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.1,
    "font.size": 11,
})


# ── Data loading ─────────────────────────────────────────────────────────────
def parse_run_id(filename):
    """Parse run_id into its components.
    Example: variant1_TFE-C_cfg1_AFE-B_cfg1_VFE-A_cfg1_d512_e8
    """
    stem = filename.replace("_summary.json", "").replace(".jsonl", "")
    parts = stem.split("_")
    model = parts[0]
    # text_cfg: parts[1] + "_" + parts[2]
    text_cfg = f"{parts[1]}_{parts[2]}"
    # audio_cfg: parts[3] + "_" + parts[4]
    audio_cfg = f"{parts[3]}_{parts[4]}"
    # video_cfg: parts[5] + "_" + parts[6]
    video_cfg = f"{parts[5]}_{parts[6]}"
    # d_model: parts[7] (e.g. "d512" -> 512)
    d_model = int(parts[7][1:])
    # num_experts: parts[8] (e.g. "e8" -> 8)
    num_experts = int(parts[8][1:])
    text_backbone = text_cfg.split("_")[0]
    audio_backbone = audio_cfg.split("_")[0]
    video_backbone = video_cfg.split("_")[0]
    text_cfg_only = text_cfg.split("_")[1]
    audio_cfg_only = audio_cfg.split("_")[1]
    video_cfg_only = video_cfg.split("_")[1]
    return {
        "model": model, "text_cfg": text_cfg, "audio_cfg": audio_cfg,
        "video_cfg": video_cfg, "d_model": d_model, "num_experts": num_experts,
        "text_backbone": text_backbone, "audio_backbone": audio_backbone,
        "video_backbone": video_backbone, "text_cfg_only": text_cfg_only,
        "audio_cfg_only": audio_cfg_only, "video_cfg_only": video_cfg_only,
        "run_id": stem,
    }


def load_summaries():
    """Load all summary JSONs into a DataFrame."""
    rows = []
    for f in sorted(LOG_DIR.glob("*_summary.json")):
        with open(f) as fh:
            data = json.load(fh)
        meta = parse_run_id(f.name)
        data.update(meta)
        data["text_backbone_label"] = TEXT_BACKBONES.get(data["text_backbone"], data["text_backbone"])
        data["audio_backbone_label"] = AUDIO_BACKBONES.get(data["audio_backbone"], data["audio_backbone"])
        data["video_backbone_label"] = VIDEO_BACKBONES.get(data["video_backbone"], data["video_backbone"])
        data["model_label"] = MODEL_LABELS.get(data["model"], data["model"])
        rows.append(data)
    df = pd.DataFrame(rows)
    if not df.empty:
        df["model"] = pd.Categorical(df["model"], categories=MODEL_ORDER, ordered=True)
    return df


def load_epoch_logs():
    """Load all jsonl files. Detect and separate multiple runs within a single file."""
    records = []
    for f in sorted(LOG_DIR.glob("*.jsonl")):
        meta = parse_run_id(f.name)
        runs = []
        current_run = []
        for line in f.read_text().strip().split("\n"):
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("epoch") == 1 and current_run:
                runs.append(current_run)
                current_run = []
            current_run.append(d)
        if current_run:
            runs.append(current_run)
        for d in runs[-1]:  # use last run only
            rec = {**meta, **d}
            records.append(rec)
    return pd.DataFrame(records)


def load_failures():
    """Extract FAILED lines from runner.log."""
    log_path = LOG_DIR / "runner.log"
    if not log_path.exists():
        return []
    text = log_path.read_text(errors="replace")
    failures = []
    for line in text.split("\n"):
        if line.startswith("FAILED:"):
            failures.append(line[len("FAILED: "):].strip())
    return failures


# ── Statistics ───────────────────────────────────────────────────────────────
def compute_stats(df):
    """Compute descriptive statistics by various groupings."""
    stats_list = []

    def _add(title, group_cols, group_names):
        for cols, name in zip(group_cols, group_names):
            grp = df.groupby(cols).agg(
                count=("test_f1", "count"),
                test_acc_mean=("test_acc", "mean"),
                test_acc_std=("test_acc", "std"),
                test_f1_mean=("test_f1", "mean"),
                test_f1_std=("test_f1", "std"),
                best_dev_f1_mean=("best_dev_f1", "mean"),
                best_dev_f1_std=("best_dev_f1", "std"),
                best_epoch_mean=("best_epoch", "mean"),
                best_epoch_std=("best_epoch", "std"),
                elapsed_mean=("elapsed_sec", "mean"),
                elapsed_std=("elapsed_sec", "std"),
                params_mean=("tfe_dim", "first"),  # just carrying dims
            ).round(4)
            grp["group"] = name
            grp = grp.reset_index()
            stats_list.append(grp)

    group_specs = [
        (["model", "model_label"], ["by_model"]),
        (["text_backbone", "text_backbone_label"], ["by_text_backbone"]),
        (["audio_backbone", "audio_backbone_label"], ["by_audio_backbone"]),
        (["video_backbone", "video_backbone_label"], ["by_video_backbone"]),
        (["d_model"], ["by_d_model"]),
        (["num_experts"], ["by_num_experts"]),
        (["model", "d_model"], ["by_model_dmodel"]),
        (["model", "num_experts"], ["by_model_experts"]),
        (["text_backbone", "model"], ["by_text_model"]),
        (["text_cfg"], ["by_text_config"]),
        (["model", "text_backbone", "d_model", "num_experts"], ["by_model_text_dmodel_experts"]),
    ]
    _add("", [cols for cols, _ in group_specs], [names[0] for _, names in group_specs])

    return pd.concat(stats_list, ignore_index=True, sort=False)


def compute_text_backbone_cross_modality(df):
    """Analyze the two cross-modality configs: cross-audio and cross-video."""
    cross = df[(df["text_cfg"].str.startswith("TFE-C_cfg1"))]
    return cross.groupby(["audio_backbone", "video_backbone", "model"]).agg(
        count=("test_f1", "count"),
        test_f1_mean=("test_f1", "mean"),
        test_acc_mean=("test_acc", "mean"),
    ).round(4).reset_index()


# ── Tables ───────────────────────────────────────────────────────────────────
def write_text_report(df, stats, failures):
    """Write a comprehensive text report."""
    report_path = BASE_DIR / "report.txt"
    lines = []
    lines.append("=" * 80)
    lines.append("MAMBA-MoE EXPERIMENT ANALYSIS REPORT")
    lines.append("=" * 80)
    lines.append(f"\nTotal experiments: {len(df)}")
    lines.append(f"Failed experiments: {len(failures)}")
    lines.append(f"Total models: {df['model'].nunique()}")
    lines.append(f"Total text backbones: {df['text_backbone'].nunique()}")
    lines.append(f"Total audio backbones: {df['audio_backbone'].nunique()}")
    lines.append(f"Total video backbones: {df['video_backbone'].nunique()}")
    lines.append(f"d_model values: {sorted(df['d_model'].unique())}")
    lines.append(f"num_experts values: {sorted(df['num_experts'].unique())}")

    lines.append("\n" + "-" * 80)
    lines.append("OVERALL STATISTICS")
    lines.append("-" * 80)
    lines.append(f"  test_acc  mean={df['test_acc'].mean():.4f}  std={df['test_acc'].std():.4f}  min={df['test_acc'].min():.4f}  max={df['test_acc'].max():.4f}")
    lines.append(f"  test_f1   mean={df['test_f1'].mean():.4f}  std={df['test_f1'].std():.4f}  min={df['test_f1'].min():.4f}  max={df['test_f1'].max():.4f}")
    lines.append(f"  best_dev_f1 mean={df['best_dev_f1'].mean():.4f}  std={df['best_dev_f1'].std():.4f}")
    lines.append(f"  best_epoch mean={df['best_epoch'].mean():.1f}  std={df['best_epoch'].std():.1f}")
    lines.append(f"  elapsed mean={df['elapsed_sec'].mean():.1f}s  std={df['elapsed_sec'].std():.1f}s")

    # By model
    lines.append("\n" + "-" * 80)
    lines.append("BY MODEL (aggregated across all configs)")
    lines.append("-" * 80)
    for m in MODEL_ORDER:
        sub = df[df["model"] == m]
        if len(sub) == 0:
            continue
        lines.append(f"\n  {m} ({MODEL_LABELS.get(m, m)})  n={len(sub)}")
        lines.append(f"    test_f1   mean={sub['test_f1'].mean():.4f}  std={sub['test_f1'].std():.4f}  max={sub['test_f1'].max():.4f}")
        lines.append(f"    test_acc  mean={sub['test_acc'].mean():.4f}  std={sub['test_acc'].std():.4f}  max={sub['test_acc'].max():.4f}")
        lines.append(f"    best_dev_f1 mean={sub['best_dev_f1'].mean():.4f}  std={sub['best_dev_f1'].std():.4f}")

    # By text backbone
    lines.append("\n" + "-" * 80)
    lines.append("BY TEXT BACKBONE (aggregated across all configs)")
    lines.append("-" * 80)
    for tb in sorted(TEXT_BACKBONES):
        sub = df[df["text_backbone"] == tb]
        if len(sub) == 0:
            continue
        lines.append(f"\n  {tb} ({TEXT_BACKBONES[tb]})  n={len(sub)}")
        lines.append(f"    test_f1   mean={sub['test_f1'].mean():.4f}  std={sub['test_f1'].std():.4f}  max={sub['test_f1'].max():.4f}")

    # By d_model
    lines.append("\n" + "-" * 80)
    lines.append("BY D_MODEL")
    lines.append("-" * 80)
    for dm in sorted(df["d_model"].unique()):
        sub = df[df["d_model"] == dm]
        lines.append(f"  d_model={dm}: test_f1 mean={sub['test_f1'].mean():.4f}  std={sub['test_f1'].std():.4f}  n={len(sub)}")

    # By num_experts
    lines.append("\n" + "-" * 80)
    lines.append("BY NUM_EXPERTS")
    lines.append("-" * 80)
    for ne in sorted(df["num_experts"].unique()):
        sub = df[df["num_experts"] == ne]
        lines.append(f"  experts={ne}: test_f1 mean={sub['test_f1'].mean():.4f}  std={sub['test_f1'].std():.4f}  n={len(sub)}")

    # Top 10 best runs
    lines.append("\n" + "-" * 80)
    lines.append("TOP 10 BEST RUNS (by test_f1)")
    lines.append("-" * 80)
    best = df.nlargest(10, "test_f1")[["model", "text_cfg", "d_model", "num_experts", "test_f1", "test_acc", "best_dev_f1", "best_epoch"]]
    for _, r in best.iterrows():
        lines.append(f"  {r['model']:12s}  {r['text_cfg']:14s}  d={r['d_model']}  e={r['num_experts']}  F1={r['test_f1']:.4f}  ACC={r['test_acc']:.4f}  devF1={r['best_dev_f1']:.4f}  epoch={r['best_epoch']}")

    # Top 10 worst runs
    lines.append("\n" + "-" * 80)
    lines.append("BOTTOM 10 WORST RUNS (by test_f1)")
    lines.append("-" * 80)
    worst = df.nsmallest(10, "test_f1")[["model", "text_cfg", "d_model", "num_experts", "test_f1", "test_acc", "best_dev_f1", "best_epoch"]]
    for _, r in worst.iterrows():
        lines.append(f"  {r['model']:12s}  {r['text_cfg']:14s}  d={r['d_model']}  e={r['num_experts']}  F1={r['test_f1']:.4f}  ACC={r['test_acc']:.4f}  devF1={r['best_dev_f1']:.4f}  epoch={r['best_epoch']}")

    # Failures
    if failures:
        lines.append("\n" + "-" * 80)
        lines.append("FAILED EXPERIMENTS")
        lines.append("-" * 80)
        for f in failures:
            lines.append(f"  {f}")

    # Cross-modality analysis
    lines.append("\n" + "-" * 80)
    lines.append("CROSS-MODALITY ANALYSIS (TFE-C_cfg1 with different A/V backbones)")
    lines.append("-" * 80)
    cross_df = compute_text_backbone_cross_modality(df)
    for _, r in cross_df.iterrows():
        lines.append(f"  {r['audio_backbone']} + {r['video_backbone']} | {r['model']:12s}  F1={r['test_f1_mean']:.4f}  n={r['count']}")

    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[report] {report_path}")

    # Also dump full summary CSVs
    csv_path = BASE_DIR / "all_summaries.csv"
    df.to_csv(csv_path, index=False)
    print(f"[csv]    {csv_path}")

    csv_stats = BASE_DIR / "statistics_by_group.csv"
    stats.to_csv(csv_stats, index=False)
    print(f"[csv]    {csv_stats}")


# ── Graph 1: Model comparison ────────────────────────────────────────────────
def plot_model_comparison(df):
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    order = MODEL_ORDER
    for ax, metric, ylabel in zip(
        axes,
        ["test_f1", "test_acc", "best_dev_f1"],
        ["Test F1", "Test Accuracy", "Best Dev F1"],
    ):
        sns.boxplot(data=df, x="model", y=metric, order=order, ax=ax, hue="model",
                    palette="Set2", legend=False, linewidth=1.2)
        sns.stripplot(data=df, x="model", y=metric, order=order, ax=ax,
                      color="black", alpha=0.3, size=4, jitter=True)
        ax.set_xlabel("")
        ax.set_ylabel(ylabel)
        ax.set_xticklabels([MODEL_LABELS.get(m.get_text(), m.get_text()) for m in ax.get_xticklabels()],
                           rotation=20, ha="right")
    fig.suptitle("Model Performance Comparison", fontweight="bold")
    fig.tight_layout()
    path = GRAPH_DIR / "01_model_comparison.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"[graph]  {path}")


# ── Graph 2: Text backbone comparison ────────────────────────────────────────
def plot_text_backbone_comparison(df):
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    order = sorted(df["text_backbone"].unique())
    for ax, metric, ylabel in zip(
        axes,
        ["test_f1", "test_acc", "best_dev_f1"],
        ["Test F1", "Test Accuracy", "Best Dev F1"],
    ):
        sns.boxplot(data=df, x="text_backbone", y=metric, order=order, ax=ax,
                    palette="Set2", linewidth=1.2)
        sns.stripplot(data=df, x="text_backbone", y=metric, order=order, ax=ax,
                      color="black", alpha=0.3, size=4, jitter=True)
        labels = [f"{b}\n({TEXT_BACKBONES.get(b,'')})" for b in order]
        ax.set_xticklabels(labels, rotation=0, ha="center")
        ax.set_xlabel("")
        ax.set_ylabel(ylabel)
    fig.suptitle("Text Feature Extractor Comparison", fontweight="bold")
    fig.tight_layout()
    path = GRAPH_DIR / "02_text_backbone_comparison.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"[graph]  {path}")


# ── Graph 3: d_model vs num_experts heatmap ─────────────────────────────────
def plot_dmodel_experts_heatmap(df):
    pivot = df.groupby(["d_model", "num_experts"]).agg(
        test_f1=("test_f1", "mean"),
        test_acc=("test_acc", "mean"),
        count=("test_f1", "count"),
    ).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, metric, title in zip(
        axes,
        ["test_f1", "test_acc"],
        ["Mean Test F1", "Mean Test Accuracy"],
    ):
        tbl = pivot.pivot_table(index="d_model", columns="num_experts", values=metric)
        sns.heatmap(tbl, annot=True, fmt=".4f", cmap="YlOrRd", ax=ax,
                    cbar_kws={"label": title}, linewidths=1, linecolor="white")
        ax.set_title(title)
        ax.set_ylabel("d_model")
        ax.set_xlabel("num_experts")
    fig.suptitle("Hyperparameter Grid: d_model × num_experts", fontweight="bold")
    fig.tight_layout()
    path = GRAPH_DIR / "03_dmodel_experts_heatmap.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"[graph]  {path}")


# ── Graph 4: Model × d_model interaction ─────────────────────────────────────
def plot_model_dmodel_interaction(df):
    g = sns.catplot(
        data=df, x="d_model", y="test_f1", hue="model", kind="point",
        palette="Set2", height=6, aspect=1.5, ci="sd", markers=["o", "s", "D", "^"],
        capsize=0.1, errwidth=1.5,
    )
    g.set_axis_labels("d_model", "Mean Test F1")
    g.fig.suptitle("Model × d_model Interaction", fontweight="bold")
    g.fig.tight_layout()
    path = GRAPH_DIR / "04_model_dmodel_interaction.png"
    g.savefig(path)
    plt.close(g.fig)
    print(f"[graph]  {path}")


# ── Graph 5: Model × num_experts interaction ─────────────────────────────────
def plot_model_experts_interaction(df):
    g = sns.catplot(
        data=df, x="num_experts", y="test_f1", hue="model", kind="point",
        palette="Set2", height=6, aspect=1.5, ci="sd", markers=["o", "s", "D", "^"],
        capsize=0.1, errwidth=1.5,
    )
    g.set_axis_labels("num_experts", "Mean Test F1")
    g.fig.suptitle("Model × num_experts Interaction", fontweight="bold")
    g.fig.tight_layout()
    path = GRAPH_DIR / "05_model_experts_interaction.png"
    g.savefig(path)
    plt.close(g.fig)
    print(f"[graph]  {path}")


# ── Graph 6: Text backbone × Model heatmap ──────────────────────────────────
def plot_textbackbone_model_heatmap(df):
    pivot = df.groupby(["text_backbone", "model"]).agg(
        test_f1=("test_f1", "mean"),
        count=("test_f1", "count"),
    ).reset_index()
    tbl = pivot.pivot_table(index="text_backbone", columns="model", values="test_f1")
    tbl = tbl.reindex(sorted(tbl.index), axis=0)
    tbl = tbl.reindex(MODEL_ORDER, axis=1)

    fig, ax = plt.subplots(figsize=(14, 8))
    annot = tbl.map(lambda x: f"{x:.4f}").values if not tbl.empty else None
    sns.heatmap(tbl, annot=annot, fmt="", cmap="YlOrRd", ax=ax,
                cbar_kws={"label": "Mean Test F1"}, linewidths=1, linecolor="white")
    ax.set_title("Mean Test F1: Text Backbone × Model", fontweight="bold")
    idx_labels = [f"{b}\n({TEXT_BACKBONES.get(b,'')})" for b in tbl.index]
    col_labels = [MODEL_LABELS.get(c, c) for c in tbl.columns]
    ax.set_xticklabels(col_labels, rotation=20, ha="right")
    ax.set_yticklabels(idx_labels, rotation=0)
    fig.tight_layout()
    path = GRAPH_DIR / "06_text_backbone_model_heatmap.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"[graph]  {path}")


# ── Graph 7: Learning curves per model ───────────────────────────────────────
def plot_learning_curves(summary_df, epoch_df):
    if epoch_df.empty:
        print("[skip]   No epoch log data for learning curves")
        return

    # Create model-level aggregated curves
    # Group by model and epoch, compute mean dev_f1 and train_loss
    epoch_df_copy = epoch_df.copy()
    if "dev_loss" not in epoch_df_copy.columns:
        epoch_df_copy["dev_loss"] = np.nan

    # Aggregate across all runs for each model
    model_curves = epoch_df_copy.groupby(["model", "epoch"]).agg(
        dev_f1=("dev_f1", "mean"),
        dev_f1_std=("dev_f1", "std"),
        dev_acc=("dev_acc", "mean"),
        train_loss=("train_loss", "mean"),
    ).reset_index()

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Dev F1 curves
    ax = axes[0, 0]
    for m in MODEL_ORDER:
        sub = model_curves[model_curves["model"] == m]
        ax.plot(sub["epoch"], sub["dev_f1"], label=MODEL_LABELS.get(m, m), linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Dev F1")
    ax.set_title("Average Dev F1 over Training")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Dev Accuracy curves
    ax = axes[0, 1]
    for m in MODEL_ORDER:
        sub = model_curves[model_curves["model"] == m]
        ax.plot(sub["epoch"], sub["dev_acc"], label=MODEL_LABELS.get(m, m), linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Dev Accuracy")
    ax.set_title("Average Dev Accuracy over Training")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Train Loss curves
    ax = axes[1, 0]
    for m in MODEL_ORDER:
        sub = model_curves[model_curves["model"] == m]
        ax.plot(sub["epoch"], sub["train_loss"], label=MODEL_LABELS.get(m, m), linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train Loss")
    ax.set_title("Average Train Loss over Training")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # d_model effect on learning curves
    ax = axes[1, 1]
    dm_curves = epoch_df_copy.groupby(["d_model", "epoch"]).agg(
        dev_f1=("dev_f1", "mean"),
    ).reset_index()
    for dm in sorted(dm_curves["d_model"].unique()):
        sub = dm_curves[dm_curves["d_model"] == dm]
        ax.plot(sub["epoch"], sub["dev_f1"], label=f"d_model={dm}", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Dev F1")
    ax.set_title("Dev F1 by d_model (across all models)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Training Dynamics (Learning Curves)", fontweight="bold")
    fig.tight_layout()
    path = GRAPH_DIR / "07_learning_curves.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"[graph]  {path}")


# ── Graph 8: Best epoch distribution ─────────────────────────────────────────
def plot_best_epoch_distribution(df):
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Histogram overall
    ax = axes[0]
    ax.hist(df["best_epoch"], bins=20, edgecolor="white", alpha=0.8)
    ax.set_xlabel("Best Epoch")
    ax.set_ylabel("Count")
    ax.set_title("Best Epoch Distribution (all runs)")

    # By model
    ax = axes[1]
    for m in MODEL_ORDER:
        sub = df[df["model"] == m]
        if len(sub) == 0:
            continue
        ax.hist(sub["best_epoch"], bins=15, alpha=0.5, label=MODEL_LABELS.get(m, m), edgecolor="white")
    ax.set_xlabel("Best Epoch")
    ax.set_ylabel("Count")
    ax.set_title("Best Epoch by Model")
    ax.legend(fontsize=8)

    # By d_model
    ax = axes[2]
    for dm in sorted(df["d_model"].unique()):
        sub = df[df["d_model"] == dm]
        ax.hist(sub["best_epoch"], bins=15, alpha=0.5, label=f"d_model={dm}", edgecolor="white")
    ax.set_xlabel("Best Epoch")
    ax.set_ylabel("Count")
    ax.set_title("Best Epoch by d_model")
    ax.legend(fontsize=8)

    fig.suptitle("Best Epoch (Early Stopping) Analysis", fontweight="bold")
    fig.tight_layout()
    path = GRAPH_DIR / "08_best_epoch_distribution.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"[graph]  {path}")


# ── Graph 9: Training time analysis ──────────────────────────────────────────
def plot_training_time(df):
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    ax = axes[0]
    sns.boxplot(data=df, x="model", y="elapsed_sec", order=MODEL_ORDER, ax=ax,
                palette="Set2", linewidth=1.2)
    ax.set_xticklabels([MODEL_LABELS.get(m.get_text(), m.get_text()) for m in ax.get_xticklabels()],
                       rotation=20, ha="right")
    ax.set_ylabel("Elapsed (seconds)")
    ax.set_title("Training Time by Model")

    ax = axes[1]
    sns.boxplot(data=df, x="d_model", y="elapsed_sec", ax=ax, palette="Reds", linewidth=1.2)
    ax.set_xlabel("d_model")
    ax.set_ylabel("Elapsed (seconds)")
    ax.set_title("Training Time by d_model")

    ax = axes[2]
    sns.boxplot(data=df, x="num_experts", y="elapsed_sec", ax=ax, palette="Blues", linewidth=1.2)
    ax.set_xlabel("num_experts")
    ax.set_ylabel("Elapsed (seconds)")
    ax.set_title("Training Time by num_experts")

    fig.suptitle("Training Time Analysis", fontweight="bold")
    fig.tight_layout()
    path = GRAPH_DIR / "09_training_time.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"[graph]  {path}")


# ── Graph 10: Scatter test_f1 vs best_dev_f1 ────────────────────────────────
def plot_test_vs_dev_scatter(df):
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = sns.color_palette("Set2", n_colors=4)
    for i, m in enumerate(MODEL_ORDER):
        sub = df[df["model"] == m]
        ax.scatter(sub["best_dev_f1"], sub["test_f1"], c=[colors[i]],
                   label=MODEL_LABELS.get(m, m), alpha=0.6, s=60, edgecolors="black", linewidth=0.3)
    # Diagonal line
    lims = [min(df["best_dev_f1"].min(), df["test_f1"].min()) - 0.02,
            max(df["best_dev_f1"].max(), df["test_f1"].max()) + 0.02]
    ax.plot(lims, lims, "--", color="gray", alpha=0.5, label="y = x")
    ax.set_xlabel("Best Dev F1")
    ax.set_ylabel("Test F1")
    ax.set_title("Test F1 vs Best Dev F1 (generalization gap)", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = GRAPH_DIR / "10_test_vs_dev_scatter.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"[graph]  {path}")


# ── Graph 11: d_model distribution impact ────────────────────────────────────
def plot_dmodel_distribution(df):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, metric, ylabel in zip(
        axes,
        ["test_f1", "elapsed_sec"],
        ["Test F1", "Elapsed (seconds)"],
    ):
        sns.violinplot(data=df, x="d_model", y=metric, ax=ax, palette="Set2",
                       inner="quartile", linewidth=1.2)
        ax.set_xlabel("d_model")
        ax.set_ylabel(ylabel)
    fig.suptitle("Impact of d_model on Performance and Training Time", fontweight="bold")
    fig.tight_layout()
    path = GRAPH_DIR / "11_dmodel_violin.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"[graph]  {path}")


# ── Graph 12: Multi-faceted comparison ─────────────────────────────────────────
def plot_multifaceted(df):
    """One big figure showing performance breakdown by model, text backbone, d_model, experts."""
    fig, axes = plt.subplots(2, 3, figsize=(22, 14))

    # (0,0) model
    ax = axes[0, 0]
    sns.barplot(data=df, x="model", y="test_f1", order=MODEL_ORDER, ax=ax,
                palette="Set2", ci="sd", capsize=0.1)
    ax.set_xlabel("")
    ax.set_ylabel("Test F1")
    ax.set_title("By Model")
    ax.set_xticklabels([MODEL_LABELS.get(m.get_text(), m.get_text()) for m in ax.get_xticklabels()],
                       rotation=20, ha="right")

    # (0,1) text backbone
    ax = axes[0, 1]
    torder = sorted(df["text_backbone"].unique())
    sns.barplot(data=df, x="text_backbone", y="test_f1", order=torder, ax=ax,
                palette="Set2", ci="sd", capsize=0.1)
    ax.set_xlabel("")
    ax.set_ylabel("Test F1")
    ax.set_title("By Text Backbone")
    ax.set_xticklabels([f"{b}\n({TEXT_BACKBONES.get(b,'')})" for b in torder], rotation=0, ha="center")

    # (0,2) d_model
    ax = axes[0, 2]
    sns.barplot(data=df, x="d_model", y="test_f1", ax=ax, palette="Reds", ci="sd", capsize=0.1)
    ax.set_xlabel("d_model")
    ax.set_ylabel("Test F1")
    ax.set_title("By d_model")

    # (1,0) num_experts
    ax = axes[1, 0]
    sns.barplot(data=df, x="num_experts", y="test_f1", ax=ax, palette="Blues", ci="sd", capsize=0.1)
    ax.set_xlabel("num_experts")
    ax.set_ylabel("Test F1")
    ax.set_title("By num_experts")

    # (1,1) text backbone × model
    ax = axes[1, 1]
    sns.pointplot(data=df, x="text_backbone", y="test_f1", hue="model",
                  order=torder, hue_order=MODEL_ORDER, ax=ax, palette="Set2",
                  ci="sd", markers=["o", "s", "D", "^"], capsize=0.1)
    ax.set_xlabel("Text Backbone")
    ax.set_ylabel("Test F1")
    ax.set_title("Text Backbone × Model")
    ax.legend(fontsize=7, title="")
    ax.set_xticklabels([TEXT_BACKBONES.get(b.get_text(), b.get_text()) for b in ax.get_xticklabels()],
                       rotation=15, ha="right")

    # (1,2) d_model × num_experts interaction
    ax = axes[1, 2]
    sns.pointplot(data=df, x="d_model", y="test_f1", hue="num_experts",
                  ax=ax, palette="Blues", ci="sd", markers=["o", "s"],
                  capsize=0.1)
    ax.set_xlabel("d_model")
    ax.set_ylabel("Test F1")
    ax.set_title("d_model × num_experts")
    ax.legend(title="experts", fontsize=8)

    fig.suptitle("Multi-Faceted Performance Analysis", fontweight="bold", fontsize=16)
    fig.tight_layout()
    path = GRAPH_DIR / "12_multifaceted.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"[graph]  {path}")


# ── Graph 13: Audio/Video cross-modality ─────────────────────────────────────
def plot_cross_modality(df):
    # Compare AFE-B vs AFE-C and VFE-A vs VFE-B for TFE-C runs
    cross = df[df["text_cfg"].str.startswith("TFE-C_cfg1")]
    audio_base = cross[cross["audio_backbone"] == "AFE-B"]
    audio_cross = cross[cross["audio_backbone"] == "AFE-C"]
    video_base = cross[cross["video_backbone"] == "VFE-A"]
    video_cross = cross[cross["video_backbone"] == "VFE-B"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Audio comparison
    ax = axes[0]
    audio_data = pd.concat([
        audio_base.assign(Modality="AFE-B (WavLM-Large)"),
        audio_cross.assign(Modality="AFE-C (Cross)"),
    ])
    sns.boxplot(data=audio_data, x="model", y="test_f1", hue="Modality",
                order=MODEL_ORDER, ax=ax, palette="Set2", linewidth=1.2)
    ax.set_xlabel("")
    ax.set_ylabel("Test F1")
    ax.set_title("Audio Backbone: AFE-B vs AFE-C\n(with TFE-C_cfg1)")
    ax.set_xticklabels([MODEL_LABELS.get(m.get_text(), m.get_text()) for m in ax.get_xticklabels()],
                       rotation=20, ha="right")
    ax.legend(fontsize=8)

    # Video comparison
    ax = axes[1]
    video_data = pd.concat([
        video_base.assign(Modality="VFE-A (ResNet50)"),
        video_cross.assign(Modality="VFE-B (Cross)"),
    ])
    sns.boxplot(data=video_data, x="model", y="test_f1", hue="Modality",
                order=MODEL_ORDER, ax=ax, palette="Set2", linewidth=1.2)
    ax.set_xlabel("")
    ax.set_ylabel("Test F1")
    ax.set_title("Video Backbone: VFE-A vs VFE-B\n(with TFE-C_cfg1)")
    ax.set_xticklabels([MODEL_LABELS.get(m.get_text(), m.get_text()) for m in ax.get_xticklabels()],
                       rotation=20, ha="right")
    ax.legend(fontsize=8)

    fig.suptitle("Cross-Modality Feature Extractor Impact", fontweight="bold")
    fig.tight_layout()
    path = GRAPH_DIR / "13_cross_modality.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"[graph]  {path}")


# ── Graph 14: Text config (pooling) comparison for DeBERTa ───────────────────
def plot_text_config_variants(df):
    # Filter to DeBERTa with different configs
    deberta = df[df["text_backbone"] == "TFE-C"]
    if len(deberta) == 0:
        print("[skip]   No DeBERTa config variant data")
        return

    fig, ax = plt.subplots(figsize=(14, 7))
    sns.pointplot(data=deberta, x="text_cfg_only", y="test_f1", hue="model",
                  hue_order=MODEL_ORDER, ax=ax, palette="Set2",
                  ci="sd", markers=["o", "s", "D", "^"], capsize=0.1)
    ax.set_xlabel("Text Config (Pooling)")
    ax.set_ylabel("Test F1")
    ax.set_title("DeBERTa-v3 Pooling Configuration Variants (cfg1–cfg5)", fontweight="bold")
    ax.legend(title="", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = GRAPH_DIR / "14_deberta_pooling_variants.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"[graph]  {path}")


# ── Graph 15: Config group summary heatmap ───────────────────────────────────
def plot_config_heatmap(df):
    """Heatmap of all 12 config groups × 4 models average test_f1."""
    # Map TFE backbone to readable label
    df_copy = df.copy()
    df_copy["text_label"] = df_copy["text_backbone"].map(TEXT_BACKBONES)

    # Create a compact label for each config group
    config_map = {
        "TFE-C_cfg1 + AFE-B_cfg1 + VFE-A_cfg1": "DeBERTa cfg1",
        "TFE-C_cfg3 + AFE-B_cfg3 + VFE-A_cfg3": "DeBERTa cfg3",
        "TFE-C_cfg4 + AFE-B_cfg1 + VFE-A_cfg1": "DeBERTa cfg4",
        "TFE-C_cfg5 + AFE-B_cfg1 + VFE-A_cfg1": "DeBERTa cfg5",
        "TFE-B_cfg1 + AFE-B_cfg1 + VFE-A_cfg1": "RoBERTa-L cfg1",
        "TFE-B_cfg3 + AFE-B_cfg3 + VFE-A_cfg3": "RoBERTa-L cfg3",
        "TFE-D_cfg1 + AFE-B_cfg1 + VFE-A_cfg1": "BERT-L cfg1",
        "TFE-E_cfg1 + AFE-B_cfg1 + VFE-A_cfg1": "Electra-L cfg1",
        "TFE-A_cfg1 + AFE-B_cfg1 + VFE-A_cfg1": "RoBERTa-B cfg1",
        "TFE-A_cfg3 + AFE-B_cfg3 + VFE-A_cfg3": "RoBERTa-B cfg3",
        "TFE-C_cfg1 + AFE-C_cfg1 + VFE-A_cfg1": "DeBERTa ×AFE-C",
        "TFE-C_cfg1 + AFE-B_cfg1 + VFE-B_cfg1": "DeBERTa ×VFE-B",
    }

    df_copy["config_label"] = df_copy.apply(
        lambda r: config_map.get(f"{r['text_cfg']} + {r['audio_cfg']} + {r['video_cfg']}", "Other"),
        axis=1,
    )

    pivot = df_copy.groupby(["config_label", "model"]).agg(
        test_f1=("test_f1", "mean"),
    ).reset_index()
    tbl = pivot.pivot_table(index="config_label", columns="model", values="test_f1")
    tbl = tbl.reindex(MODEL_ORDER, axis=1)

    fig, ax = plt.subplots(figsize=(14, 10))
    annot = tbl.map(lambda x: f"{x:.4f}").values if not tbl.empty else None
    sns.heatmap(tbl, annot=annot, fmt="", cmap="YlOrRd", ax=ax,
                cbar_kws={"label": "Mean Test F1"}, linewidths=1, linecolor="white",
                vmin=df["test_f1"].min(), vmax=df["test_f1"].max())
    col_labels = [MODEL_LABELS.get(c, c) for c in tbl.columns]
    ax.set_xticklabels(col_labels, rotation=20, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.set_title("Mean Test F1 by Configuration Group × Model", fontweight="bold")
    fig.tight_layout()
    path = GRAPH_DIR / "15_config_heatmap.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"[graph]  {path}")


# ── Graph 16: Per-config full detail bar chart ───────────────────────────────
def plot_config_detail_bars(df):
    """Horizontal bar chart of every run sorted by test_f1."""
    df_sorted = df.sort_values("test_f1", ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(16, max(12, len(df_sorted) * 0.35)))
    colors = [sns.color_palette("Set2")[MODEL_ORDER.index(m)] for m in df_sorted["model"]]
    bars = ax.barh(range(len(df_sorted)), df_sorted["test_f1"], color=colors, edgecolor="white", linewidth=0.5, alpha=0.9)

    # Create labels
    labels = []
    for _, r in df_sorted.iterrows():
        trun = r["text_backbone"].replace("TFE-", "") + "_" + r["text_cfg_only"]
        labels.append(f"{r['model'][-1]}|{trun}|d{r['d_model']}|e{r['num_experts']}")
    ax.set_yticks(range(len(df_sorted)))
    ax.set_yticklabels(labels, fontsize=7, fontfamily="monospace")
    ax.set_xlabel("Test F1")
    ax.set_title(f"All {len(df_sorted)} Runs Sorted by Test F1", fontweight="bold")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=sns.color_palette("Set2")[i], label=MODEL_LABELS[m])
                       for i, m in enumerate(MODEL_ORDER)]
    ax.legend(handles=legend_elements, fontsize=9, loc="lower right")

    ax.axvline(df["test_f1"].mean(), color="red", linestyle="--", alpha=0.5, linewidth=1,
               label=f"Mean: {df['test_f1'].mean():.4f}")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    path = GRAPH_DIR / "16_all_runs_ranked.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"[graph]  {path}")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("MAMBA-MoE ANALYSIS")
    print("=" * 60)

    # Load data
    print("\n[load]  Loading summary data...")
    df = load_summaries()
    print(f"        {len(df)} experiments found")

    print("[load]  Loading epoch logs...")
    epoch_df = load_epoch_logs()
    print(f"        {len(epoch_df)} epoch log entries")

    print("[load]  Loading failures...")
    failures = load_failures()
    print(f"        {len(failures)} failures found")

    if df.empty:
        print("[fatal] No summary JSONs found. Exiting.")
        sys.exit(1)

    # Compute statistics
    print("[stat]  Computing statistics...")
    stats = compute_stats(df)

    # Write text report
    print("[text]  Writing report and CSVs...")
    write_text_report(df, stats, failures)

    # Generate all graphs
    print("\n[graph] Generating graphs...")
    plot_model_comparison(df)
    plot_text_backbone_comparison(df)
    plot_dmodel_experts_heatmap(df)
    plot_model_dmodel_interaction(df)
    plot_model_experts_interaction(df)
    plot_textbackbone_model_heatmap(df)
    plot_learning_curves(df, epoch_df)
    plot_best_epoch_distribution(df)
    plot_training_time(df)
    plot_test_vs_dev_scatter(df)
    plot_dmodel_distribution(df)
    plot_multifaceted(df)
    plot_cross_modality(df)
    plot_text_config_variants(df)
    plot_config_heatmap(df)
    plot_config_detail_bars(df)

    print(f"\n{'=' * 60}")
    print(f"DONE. All results in analysis/")
    print(f"  report.txt  — comprehensive text report")
    print(f"  all_summaries.csv  — full experiment data")
    print(f"  statistics_by_group.csv  — per-group statistics")
    print(f"  graphs/  — {len(list(GRAPH_DIR.glob('*.png')))} PNG figures")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
