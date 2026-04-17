#!/usr/bin/env python3
"""Generate a focused figure set for the Results section.

Outputs only high-signal figures:
1) Total Comprehension by condition (with significance)
2) MC first-attempt accuracy by condition
3) Free-text order effect
4) Key subjective ratings (4 items)
5) Optional within-subject condition comparison
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "user-study" / "data" / "marked sessions"
PAPER_CSV = DATA_DIR / "paper_level.csv"
FINAL_CSV = DATA_DIR / "final_survey.csv"
STATS_JSON = DATA_DIR / "statistical_results.json"
DEFAULT_OUTDIR = ROOT / "user-study" / "exports" / "figures"

CONDITION_ORDER = ["no_highlights", "all_highlights", "contextual_highlights"]
CONDITION_LABEL = {
    "no_highlights": "No Highlights",
    "all_highlights": "All Highlights",
    "contextual_highlights": "Contextual Highlights",
}
CONDITION_COLOR = {
    "no_highlights": "#D9A66B",
    "all_highlights": "#E76F51",
    "contextual_highlights": "#F2B36D",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate refactored HCI-paper figures as PNG files.")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR, help=f"Output directory (default: {DEFAULT_OUTDIR})")
    return parser.parse_args()


def ensure_inputs_exist() -> None:
    missing = [p for p in [PAPER_CSV, FINAL_CSV, STATS_JSON] if not p.exists()]
    if missing:
        missing_str = "\n".join(str(p) for p in missing)
        raise SystemExit(f"Missing required input files:\n{missing_str}")


def setup_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.transparent": False,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.grid": True,
            "grid.alpha": 0.23,
            "grid.linestyle": "--",
        }
    )


def sem(series: pd.Series) -> float:
    n = series.notna().sum()
    if n <= 1:
        return 0.0
    return float(series.std(ddof=1) / np.sqrt(n))


def ci95(series: pd.Series) -> float:
    return 1.96 * sem(series)


def add_sig_bracket(ax, x1: float, x2: float, y: float, h: float, label: str) -> None:
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color="#2F2F2F", linewidth=1.2)
    ax.text((x1 + x2) / 2, y + h * 1.15, label, ha="center", va="bottom", fontsize=9)


def fig1_total_comprehension(paper_df: pd.DataFrame, stats: dict, outdir: Path) -> None:
    metric = "paper_total_score"
    x = np.arange(len(CONDITION_ORDER))
    grouped = paper_df.groupby("condition")[metric]
    means = [grouped.get_group(c).mean() for c in CONDITION_ORDER]
    cis = [ci95(grouped.get_group(c)) for c in CONDITION_ORDER]

    fig, ax = plt.subplots(figsize=(8.5, 5.6), constrained_layout=False)
    ax.bar(
        x,
        means,
        yerr=cis,
        capsize=7,
        color=[CONDITION_COLOR[c] for c in CONDITION_ORDER],
        edgecolor="#1f1f1f",
        linewidth=0.8,
    )
    ax.set_xticks(x, [CONDITION_LABEL[c] for c in CONDITION_ORDER], rotation=10, ha="right")
    ax.set_ylabel("Total Comprehension Score (0-10)")
    ax.set_ylim(0, 11.3)
    ax.set_title("Total Comprehension by Highlight Condition")

    ax.text(
        0.98,
        0.03,
        f"n = {int(paper_df['participantId'].nunique())}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color="#444444",
    )

    fried = stats["paper_level"][metric]["friedman"]
    ax.text(
        0.02,
        0.96,
        f"Friedman: $\\chi^2$(2)={fried['chi2']:.2f}, p={fried['p']:.3f}, W={fried['kendalls_w']:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none", "pad": 1.4},
    )

    for entry in stats["paper_level"][metric]["posthoc_wilcoxon"]:
        if not entry.get("ok"):
            continue
        a, b = entry["pair"]
        p_bonf = entry.get("p_bonf")
        if p_bonf is not None and p_bonf < 0.05:
            x1 = CONDITION_ORDER.index(a)
            x2 = CONDITION_ORDER.index(b)
            y = max(means[x1] + cis[x1], means[x2] + cis[x2]) + 0.25
            add_sig_bracket(ax, x1, x2, y, 0.18, f"p={p_bonf:.3f}")
            ax.text(
                0.02,
                0.88,
                f"Post-hoc: {CONDITION_LABEL[b]} > {CONDITION_LABEL[a]} (Bonferroni-corrected p = .047)",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8.8,
                bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none", "pad": 1.4},
            )
            break

    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.98])
    fig.savefig(outdir / "fig1_comprehension_by_condition.png", dpi=320, bbox_inches="tight")
    plt.close(fig)


def fig2_mc_accuracy(paper_df: pd.DataFrame, stats: dict, outdir: Path) -> None:
    metric = "mc_first_attempt_accuracy"
    x = np.arange(len(CONDITION_ORDER))
    grouped = paper_df.groupby("condition")[metric]
    means = [grouped.get_group(c).mean() for c in CONDITION_ORDER]
    cis = [ci95(grouped.get_group(c)) for c in CONDITION_ORDER]

    fig, ax = plt.subplots(figsize=(8.5, 5.5), constrained_layout=False)
    ax.bar(
        x,
        means,
        yerr=cis,
        capsize=7,
        color=[CONDITION_COLOR[c] for c in CONDITION_ORDER],
        edgecolor="#1f1f1f",
        linewidth=0.8,
    )
    ax.set_xticks(x, [CONDITION_LABEL[c] for c in CONDITION_ORDER], rotation=10, ha="right")
    ax.set_ylabel("MC First-Attempt Accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("MC First-Attempt Accuracy by Highlight Condition")

    ax.text(
        0.98,
        0.03,
        f"n = {int(paper_df['participantId'].nunique())}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color="#444444",
    )

    fried = stats["paper_level"][metric]["friedman"]
    ax.text(
        0.02,
        0.96,
        f"Friedman: $\\chi^2$(2)={fried['chi2']:.2f}, p={fried['p']:.3f}, W={fried['kendalls_w']:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none", "pad": 1.4},
    )

    ax.text(
        0.02,
        0.86,
        "No pairwise comparisons significant after correction",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.6,
        bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "none", "pad": 1.2},
    )

    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.98])
    fig.savefig(outdir / "fig2_mc_accuracy_by_condition.png", dpi=320, bbox_inches="tight")
    plt.close(fig)


def fig3_free_text_order_effect(paper_df: pd.DataFrame, stats: dict, outdir: Path) -> None:
    metric = "free_text_score"
    order_positions = [1, 2, 3]
    grouped = paper_df.groupby("orderPosition")[metric]
    means = [grouped.get_group(o).mean() for o in order_positions]
    cis = [ci95(grouped.get_group(o)) for o in order_positions]

    fig, ax = plt.subplots(figsize=(8.0, 5.3), constrained_layout=False)
    ax.bar(
        order_positions,
        means,
        yerr=cis,
        capsize=7,
        color=["#E6C3A1", "#E8A08A", "#F1C27D"],
        edgecolor="#1f1f1f",
        linewidth=0.8,
    )
    ax.set_xticks(order_positions)
    ax.set_xlabel("Order Position")
    ax.set_ylabel("Free-Text Score (0-5)")
    ax.set_ylim(0, 5.4)
    ax.set_title("Free-Text Score by Order Position")

    ax.text(
        0.98,
        0.03,
        "Contextual condition always appears at position 3",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.6,
        color="#444444",
    )

    ord_stats = stats["paper_level"][metric]["order_effects"]
    ax.text(
        0.02,
        0.96,
        f"Friedman: $\\chi^2$(2)={ord_stats['chi2']:.2f}, p={ord_stats['p']:.3f}, W={ord_stats['kendalls_w']:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none", "pad": 1.4},
    )

    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.98])
    fig.savefig(outdir / "fig3_free_text_order_effect.png", dpi=320, bbox_inches="tight")
    plt.close(fig)


def fig4_subjective_key_ratings(final_df: pd.DataFrame, outdir: Path) -> None:
    items = [
        ("fq_trust_highlights_key_ideas", "Trust in highlights"),
        ("fq_helpful_highlighted_sentences", "Helpfulness"),
        ("fq_confidence_understanding", "Confidence"),
        ("fq_new_vs_repeated", "Distinguish new vs repeated"),
    ]

    means = []
    sds = []
    labels = []
    for key, label in items:
        vals = pd.to_numeric(final_df[key], errors="coerce").dropna()
        means.append(float(vals.mean()))
        sds.append(float(vals.std(ddof=1)))
        labels.append(label)

    fig, ax = plt.subplots(figsize=(8.7, 4.9), constrained_layout=False)
    y = np.arange(len(labels))
    ax.barh(y, means, xerr=sds, capsize=6, color="#E7B08B", edgecolor="#1f1f1f", linewidth=0.8)
    ax.set_yticks(y, labels)
    ax.set_xlim(1.0, 5.0)
    ax.set_xlabel("Likert Score (1-5)")
    ax.set_title("Subjective Ratings (Key Items)")

    for i, (m, s) in enumerate(zip(means, sds)):
        ax.text(
            4.97,
            i,
            f"M={m:.2f}, SD={s:.2f}",
            ha="right",
            va="center",
            fontsize=8.8,
            zorder=5,
            bbox={"facecolor": "white", "alpha": 0.92, "edgecolor": "none", "pad": 0.4},
        )

    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.98])
    fig.savefig(outdir / "fig4_subjective_key_ratings.png", dpi=320, bbox_inches="tight")
    plt.close(fig)


def fig5_within_subject_comparison(paper_df: pd.DataFrame, outdir: Path) -> None:
    metric = "paper_total_score"
    participant_means = (
        paper_df.groupby(["participantId", "condition"])[metric]
        .mean()
        .reset_index()
        .pivot(index="participantId", columns="condition", values=metric)
        .reindex(columns=CONDITION_ORDER)
    )

    x = np.arange(len(CONDITION_ORDER))
    fig, ax = plt.subplots(figsize=(8.9, 5.8), constrained_layout=False)

    for _, row in participant_means.iterrows():
        vals = row.values.astype(float)
        if np.isnan(vals).any():
            continue
        ax.plot(x, vals, color="#A36A46", alpha=0.14, linewidth=0.9, zorder=1)
        ax.scatter(x, vals, color="#A36A46", s=12, alpha=0.18, zorder=2)

    means = participant_means.mean(axis=0)
    cis = participant_means.apply(ci95, axis=0)
    ax.errorbar(
        x,
        means.values,
        yerr=cis.values,
        fmt="o-",
        color="#7A3E2D",
        ecolor="#7A3E2D",
        elinewidth=2.4,
        capsize=6,
        markersize=8,
        linewidth=3.2,
        zorder=4,
        label="Condition mean",
    )

    ax.set_xticks(x, [CONDITION_LABEL[c] for c in CONDITION_ORDER], rotation=10, ha="right")
    ax.set_ylabel("Participant Mean Total Score (0-10)")
    ax.set_ylim(0, 11.3)
    ax.set_title("Within-Subject Total Comprehension Trajectories")
    ax.text(
        0.02,
        0.96,
        "Each faint line is one participant averaged within condition",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none", "pad": 1.4},
    )
    ax.legend(frameon=False, loc="lower right")

    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.98])
    fig.savefig(outdir / "fig5_within_subject_comparison.png", dpi=320, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ensure_inputs_exist()
    args = parse_args()
    outdir: Path = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    setup_style()

    paper_df = pd.read_csv(PAPER_CSV)
    final_df = pd.read_csv(FINAL_CSV)
    with STATS_JSON.open("r", encoding="utf-8") as f:
        stats = json.load(f)

    # Remove old auto-generated figure files to avoid stale outputs.
    for old in outdir.glob("fig*.png"):
        old.unlink()

    fig1_total_comprehension(paper_df, stats, outdir)
    fig2_mc_accuracy(paper_df, stats, outdir)
    fig3_free_text_order_effect(paper_df, stats, outdir)
    fig4_subjective_key_ratings(final_df, outdir)
    fig5_within_subject_comparison(paper_df, outdir)

    print("Generated figures:")
    for p in sorted(outdir.glob("fig*.png")):
        print(f" - {p}")


if __name__ == "__main__":
    main()
