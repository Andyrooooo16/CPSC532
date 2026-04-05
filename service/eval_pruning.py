"""
eval_pruning.py — Evaluate how ranking quality changes as the user accumulates
previously-read paper highlights (the "pruning" effect).

Compares several ranking strategies:
  - global_top_k : original method — read sentences compete in the PageRank graph
  - novel_max    : PageRank on target only, penalize by max sim to read (lambda=0.5)
  - novel_mean   : PageRank on target only, penalize by mean sim to read (lambda=0.5)
  - novel_thresh : hard exclude sentences with max sim >= 0.7 to any read sentence

Usage:
    python service/eval_pruning.py
    python service/eval_pruning.py --split test --trials 20 --max-papers 50
"""
import sys
import random
import argparse
import numpy as np
from collections import defaultdict

sys.path.insert(0, ".")

from classifier import classify
from embedder import embed
from ranker import rank, rank_novel


READ_COUNTS = [0, 1, 2, 3, 5, 10]

METHODS = {
    "global_top_k":  "Global top-k (original)",
    "novel_max":     "Novel: max sim penalty (λ=0.5)",
    "novel_mean":    "Novel: mean sim penalty (λ=0.5)",
    "novel_thresh":  "Novel: hard threshold (≥0.7)",
    "random":        "Random baseline",
}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def score(gold: list[str], highlighted: list[str]) -> dict:
    if not gold:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "ndcg": 0.0, "n_highlighted": 0}
    if not highlighted:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "ndcg": 0.0, "n_highlighted": 0}
    gold_set = {s.strip().lower() for s in gold}
    hits = sum(1 for h in highlighted if h.strip().lower() in gold_set)
    p  = hits / len(highlighted)
    r  = hits / len(gold)
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(gold_set), len(highlighted))))
    dcg  = sum(1.0 / np.log2(i + 2)
               for i, h in enumerate(highlighted)
               if h.strip().lower() in gold_set)
    ndcg = dcg / idcg if idcg > 0 else 0.0
    return {"precision": p, "recall": r, "f1": f1, "ndcg": ndcg,
            "n_highlighted": len(highlighted)}


def _get_highlighted(method, target, read_sents, read_embs, k_frac):
    sents = target["sentences"]
    labels = target["labels"]
    confs  = target["confs"]
    embs   = target["embeddings"]
    empty  = np.empty((0, embs.shape[1]), dtype=np.float32)

    if method == "global_top_k":
        results = rank(
            sents, labels, confs, embs,
            read_sentences=read_sents,
            read_embeddings=read_embs if len(read_sents) > 0 else empty,
            top_k_fraction=k_frac,
            global_top_k=True,
        )
    elif method == "novel_max":
        results = rank_novel(
            sents, labels, confs, embs,
            read_sentences=read_sents,
            read_embeddings=read_embs if len(read_sents) > 0 else empty,
            top_k_fraction=k_frac,
            novelty_lambda=0.5,
            sim_mode="max",
        )
    elif method == "novel_mean":
        results = rank_novel(
            sents, labels, confs, embs,
            read_sentences=read_sents,
            read_embeddings=read_embs if len(read_sents) > 0 else empty,
            top_k_fraction=k_frac,
            novelty_lambda=0.5,
            sim_mode="mean",
        )
    elif method == "novel_thresh":
        results = rank_novel(
            sents, labels, confs, embs,
            read_sentences=read_sents,
            read_embeddings=read_embs if len(read_sents) > 0 else empty,
            top_k_fraction=k_frac,
            novelty_lambda=0.5,
            sim_mode="max",
            threshold=0.7,
        )
        return [r["sentence"] for r in results]
    elif method == "random":
        k = max(1, int(len(sents) * k_frac))
        return random.sample(sents, min(k, len(sents)))
    return [r["sentence"] for r in results]


# ---------------------------------------------------------------------------
# Per-paper evaluation
# ---------------------------------------------------------------------------

def evaluate_paper(target_idx, all_papers, k_frac, n_trials):
    target   = all_papers[target_idx]
    all_gold = target["all_gold"]
    if not target["sentences"] or len(target["sentences"]) < 2 or not all_gold:
        return None

    other_indices = [i for i in range(len(all_papers))
                     if i != target_idx and all_papers[i] is not None]

    # method -> read_count -> list of metric dicts
    results = {m: {} for m in METHODS}

    for read_count in READ_COUNTS:
        if read_count > len(other_indices):
            continue

        method_trial_scores = {m: defaultdict(list) for m in METHODS}
        n_trials_actual = 1 if read_count == 0 else n_trials

        for _ in range(n_trials_actual):
            if read_count == 0:
                read_sents = []
                read_embs  = np.empty((0, target["embeddings"].shape[1]), dtype=np.float32)
            else:
                sampled = random.sample(other_indices, read_count)
                read_sents = [s for i in sampled for s in all_papers[i]["highlighted"]]
                emb_list   = [all_papers[i]["highlighted_embeddings"] for i in sampled
                              if len(all_papers[i]["highlighted"]) > 0]
                read_embs  = (np.vstack(emb_list) if emb_list
                              else np.empty((0, target["embeddings"].shape[1]), dtype=np.float32))

            for method in METHODS:
                highlighted = _get_highlighted(method, target, read_sents, read_embs, k_frac)
                s = score(all_gold, highlighted)
                for metric, val in s.items():
                    method_trial_scores[method][metric].append(val)

        for method in METHODS:
            results[method][read_count] = {
                m: np.mean(v) for m, v in method_trial_scores[method].items()
            }

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="validation",
                        choices=["train", "validation", "test"])
    parser.add_argument("--max-papers", type=int, default=None)
    parser.add_argument("--k", type=float, default=0.20)
    parser.add_argument("--trials", type=int, default=20)
    args = parser.parse_args()

    print("Loading ACLSum dataset (extractive config)...")
    from datasets import load_dataset
    ds = load_dataset("sobamchan/aclsum", "extractive", split=args.split)
    if args.max_papers:
        ds = ds.select(range(min(args.max_papers, len(ds))))

    print(f"\nPre-processing {len(ds)} papers...\n")
    all_papers = []
    for i, paper in enumerate(ds, start=1):
        sents    = paper["source_sentences"]
        all_gold = [s for key in ("challenge_sentences", "approach_sentences",
                                  "outcome_sentences") for s in paper[key]]
        if not sents or len(sents) < 2 or not all_gold:
            all_papers.append(None)
            continue

        print(f"  [{i}/{len(ds)}] classifying + embedding ({len(sents)} sentences)...")
        classifications = classify(sents)
        labels = [c[0] for c in classifications]
        confs  = [c[1] for c in classifications]
        embs   = embed(sents)

        empty = np.empty((0, embs.shape[1]), dtype=np.float32)
        ranked_baseline = rank(sents, labels, confs, embs,
                               read_sentences=[], read_embeddings=empty,
                               top_k_fraction=args.k, global_top_k=False)
        highlighted      = [r["sentence"] for r in ranked_baseline]
        highlighted_idxs = [j for j, s in enumerate(sents) if s in set(highlighted)]
        highlighted_embs = embs[highlighted_idxs] if highlighted_idxs else empty

        all_papers.append({
            "sentences":              sents,
            "labels":                 labels,
            "confs":                  confs,
            "embeddings":             embs,
            "all_gold":               all_gold,
            "highlighted":            highlighted,
            "highlighted_embeddings": highlighted_embs,
        })

    valid_indices = [i for i, p in enumerate(all_papers) if p is not None]
    print(f"\nEvaluating {len(valid_indices)} papers ({args.trials} trials per read count)...\n")

    # method -> read_count -> metric -> [values across papers]
    agg = {m: defaultdict(lambda: defaultdict(list)) for m in METHODS}

    for idx, paper_idx in enumerate(valid_indices, start=1):
        print(f"  [{idx}/{len(valid_indices)}] target paper {paper_idx}...")
        paper_results = evaluate_paper(paper_idx, all_papers, args.k, args.trials)
        if paper_results is None:
            continue
        for method in METHODS:
            for read_count, metrics in paper_results[method].items():
                for metric, val in metrics.items():
                    agg[method][read_count][metric].append(val)

    # Print table per method
    metrics = ["precision", "recall", "f1", "ndcg", "n_highlighted"]
    print()
    print("=" * 75)
    print(f"Pruning evaluation — ACLSum '{args.split}' | k={args.k:.0%} | trials={args.trials}")
    for method, label in METHODS.items():
        print(f"\n{label}")
        print(f"  {'Read':>6}  " + "  ".join(f"{m:>12}" for m in metrics))
        print("  " + "-" * 60)
        for rc in READ_COUNTS:
            if rc not in agg[method]:
                continue
            vals = "  ".join(f"{np.mean(agg[method][rc][m]):>12.3f}" for m in metrics)
            print(f"  {rc:>6}  {vals}")

    # Plot — one figure per metric, all methods on same axes
    import matplotlib.pyplot as plt
    plot_metrics  = ["precision", "recall", "f1", "ndcg", "n_highlighted"]
    metric_labels = {"precision": "Precision", "recall": "Recall", "f1": "F1",
                     "ndcg": "nDCG", "n_highlighted": "# Highlights"}
    method_colors = {
        "global_top_k": "#1f77b4",
        "novel_max":    "#ff7f0e",
        "novel_mean":   "#2ca02c",
        "novel_thresh": "#d62728",
        "random":       "#9467bd",
    }
    method_styles = {
        "global_top_k": "-",
        "novel_max":    "-",
        "novel_mean":   "--",
        "novel_thresh": "-.",
        "random":       ":",
    }

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    x = [rc for rc in READ_COUNTS if rc in agg["global_top_k"]]
    for ax, metric in zip(axes, plot_metrics):
        for method, label in METHODS.items():
            y   = [np.mean(agg[method][rc][metric]) for rc in x if rc in agg[method]]
            err = [np.std(agg[method][rc][metric]) / np.sqrt(len(agg[method][rc][metric]))
                   for rc in x if rc in agg[method]]
            ax.errorbar(x, y, yerr=err, marker="o", linewidth=2, capsize=4,
                        color=method_colors[method], linestyle=method_styles[method],
                        label=label)
        ax.set_xlabel("Number of previously-read papers")
        ax.set_ylabel(metric_labels[metric])
        ax.set_title(metric_labels[metric])
        ax.set_xticks(x)
        if metric != "n_highlighted":
            ax.set_ylim(0, 1)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    axes[-1].set_visible(False)  # 6th subplot unused
    fig.suptitle(
        f"Pruning strategies — ACLSum '{args.split}' | k={args.k:.0%} | {args.trials} trials",
        fontsize=12)
    plt.tight_layout()
    fname = f"pruning_eval_{args.split}.png"
    plt.savefig(fname, dpi=150)
    plt.show()
    print(f"\nPlot saved to {fname}")


if __name__ == "__main__":
    main()
