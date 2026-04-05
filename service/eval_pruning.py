"""
eval_pruning.py — Evaluate how ranking quality changes as the user accumulates
previously-read paper highlights (the "pruning" effect).

Compares several ranking strategies:
  - global_top_k  : original method — read sentences compete in the PageRank graph
  - novel_0.3/0.5/0.7 : re-rank by novelty (norm PR, lambda=0.3/0.5/0.7)
  - thresh_0.7/0.9    : hard cosine-similarity threshold against read highlights
  - random        : random baseline

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

GRID_LAMBDAS = [round(i * 0.2, 1) for i in range(6)]   # 0.0, 0.2, 0.4, 0.6, 0.8, 1.0
GRID_TAUS    = [round(i * 0.2, 1) for i in range(6)]   # 0.0, 0.2, 0.4, 0.6, 0.8, 1.0
RANDOM_DROPS = [10, 20, 30, 40, 50, 60, 70, 80, 90]

METHODS = {"global_top_k": "Global top-k (original)", "random": "Random (no drop)"}
for pct in RANDOM_DROPS:
    METHODS[f"random_drop_{pct}"] = f"Random drop {pct}%"
for lam in GRID_LAMBDAS:
    for tau in GRID_TAUS:
        key = f"novel_thresh_L{lam}_T{tau}"
        METHODS[key] = f"λ={lam} τ={tau}"


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


def _max_cosine_sim_to_read(target_embs: np.ndarray, read_embs: np.ndarray) -> np.ndarray:
    """For each target sentence, return max cosine similarity to any read sentence."""
    if read_embs.shape[0] == 0:
        return np.zeros(target_embs.shape[0])
    t_norms = np.linalg.norm(target_embs, axis=1, keepdims=True)
    t_norms = np.where(t_norms == 0, 1e-9, t_norms)
    r_norms = np.linalg.norm(read_embs, axis=1, keepdims=True)
    r_norms = np.where(r_norms == 0, 1e-9, r_norms)
    sim = (target_embs / t_norms) @ (read_embs / r_norms).T  # (n_target, n_read)
    return sim.max(axis=1)


def _get_highlighted(method, target, read_sents, read_embs, k_frac):
    sents  = target["sentences"]
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
        return [r["sentence"] for r in results]

    elif method == "random":
        k = max(1, int(len(sents) * k_frac))
        return random.sample(sents, min(k, len(sents)))

    elif method.startswith("random_drop_"):
        drop = int(method.split("_")[-1]) / 100.0
        k = int(round(max(1, int(len(sents) * k_frac)) * (1 - drop)))
        return random.sample(sents, min(k, len(sents))) if k > 0 else []

    elif method.startswith("novel_thresh_"):
        # Parse "novel_thresh_L{lam}_T{tau}"
        _, _, lam_part, tau_part = method.split("_")
        lam = float(lam_part[1:])
        tau = float(tau_part[1:])
        # Step 1: re-rank by novelty
        results = rank_novel(
            sents, labels, confs, embs,
            read_sentences=read_sents,
            read_embeddings=read_embs if len(read_sents) > 0 else empty,
            top_k_fraction=k_frac,
            novelty_lambda=lam,
        )
        # Step 2: cosine-threshold prune (tau=1.0 means no pruning)
        if len(read_sents) > 0 and tau < 1.0:
            sent_to_idx = {s: i for i, s in enumerate(sents)}
            max_sim = _max_cosine_sim_to_read(embs, read_embs)
            return [r["sentence"] for r in results
                    if max_sim[sent_to_idx[r["sentence"]]] < tau]
        return [r["sentence"] for r in results]

    else:
        raise ValueError(f"Unknown method: {method}")


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

    results = {m: {} for m in METHODS}
    random_drop_methods = [m for m in METHODS if m.startswith("random_drop_")]
    context_methods     = [m for m in METHODS if not m.startswith("random_drop_")]

    # Compute random_drop methods once (context-free), reuse for all read counts
    expected_k = max(1, int(len(target["sentences"]) * k_frac))
    random_drop_results = {}
    for method in random_drop_methods:
        trial_scores = defaultdict(list)
        for _ in range(n_trials):
            highlighted = _get_highlighted(method, target, [], np.empty((0, target["embeddings"].shape[1]), dtype=np.float32), k_frac)
            s = score(all_gold, highlighted)
            s["drop_pct"] = (expected_k - len(highlighted)) / expected_k
            for metric, val in s.items():
                trial_scores[metric].append(val)
        random_drop_results[method] = {m: np.mean(v) for m, v in trial_scores.items()}

    for read_count in READ_COUNTS:
        if read_count > len(other_indices):
            continue

        method_trial_scores = {m: defaultdict(list) for m in context_methods}
        n_trials_actual = 1 if read_count == 0 else n_trials

        for _ in range(n_trials_actual):
            if read_count == 0:
                read_sents = []
                read_embs  = np.empty((0, target["embeddings"].shape[1]), dtype=np.float32)
            else:
                sampled    = random.sample(other_indices, read_count)
                read_sents = [s for i in sampled for s in all_papers[i]["highlighted"]]
                emb_list   = [all_papers[i]["highlighted_embeddings"] for i in sampled
                              if len(all_papers[i]["highlighted"]) > 0]
                read_embs  = (np.vstack(emb_list) if emb_list
                              else np.empty((0, target["embeddings"].shape[1]), dtype=np.float32))

            for method in context_methods:
                highlighted = _get_highlighted(method, target, read_sents, read_embs, k_frac)
                s = score(all_gold, highlighted)
                s["drop_pct"] = (expected_k - len(highlighted)) / expected_k
                for metric, val in s.items():
                    method_trial_scores[method][metric].append(val)

        for method in context_methods:
            results[method][read_count] = {
                m: np.mean(v) for m, v in method_trial_scores[method].items()
            }
        # Copy random_drop results (same for all read counts)
        for method in random_drop_methods:
            results[method][read_count] = random_drop_results[method]

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_k(all_papers, valid_indices, k_frac, n_trials):
    """Run full evaluation for a single k value. Returns agg dict."""
    agg = {m: defaultdict(lambda: defaultdict(list)) for m in METHODS}
    for idx, paper_idx in enumerate(valid_indices, start=1):
        print(f"  [{idx}/{len(valid_indices)}] target paper {paper_idx}  (k={k_frac:.0%})...")
        paper_results = evaluate_paper(paper_idx, all_papers, k_frac, n_trials)
        if paper_results is None:
            continue
        for method in METHODS:
            for read_count, metrics in paper_results[method].items():
                for metric, val in metrics.items():
                    agg[method][read_count][metric].append(val)
    return agg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="validation",
                        choices=["train", "validation", "test"])
    parser.add_argument("--max-papers", type=int, default=None)
    parser.add_argument("--k-values", type=float, nargs="+",
                        default=[round(i * 0.1, 1) for i in range(1, 11)])
    parser.add_argument("--trials", type=int, default=5)
    args = parser.parse_args()

    print("Loading ACLSum dataset (extractive config)...")
    from datasets import load_dataset
    ds = load_dataset("sobamchan/aclsum", "extractive", split=args.split)
    if args.max_papers:
        ds = ds.select(range(min(args.max_papers, len(ds))))

    # Pre-process once at max k so highlighted_embeddings covers the widest set
    max_k = max(args.k_values)
    print(f"\nPre-processing {len(ds)} papers (baseline k={max_k:.0%})...\n")
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
                               top_k_fraction=max_k, global_top_k=False)
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
    print(f"\n{len(valid_indices)} valid papers, {args.trials} trials per read count.\n")

    # Run evaluation for each k
    all_agg = {}
    for k_frac in args.k_values:
        print(f"\n{'='*60}\nRunning k={k_frac:.0%}\n{'='*60}")
        all_agg[k_frac] = run_k(all_papers, valid_indices, k_frac, args.trials)

    # -----------------------------------------------------------------------
    # vs-random interpolation helper
    # -----------------------------------------------------------------------
    COMPARE_METRICS  = ["precision", "f1", "ndcg"]
    random_drop_pcts = sorted([int(m.split("_")[-1]) / 100.0
                                for m in METHODS if m.startswith("random_drop_")])

    def interp_random(agg, drop, metric, rc):
        if drop <= 0:
            return np.mean(agg["random"][rc][metric]) if rc in agg["random"] else 0.0
        lo = max((d for d in random_drop_pcts if d <= drop), default=random_drop_pcts[0])
        hi = min((d for d in random_drop_pcts if d >= drop), default=random_drop_pcts[-1])
        lo_key = f"random_drop_{int(lo*100)}"
        hi_key = f"random_drop_{int(hi*100)}"
        v_lo = np.mean(agg[lo_key][rc][metric]) if rc in agg[lo_key] else 0.0
        v_hi = np.mean(agg[hi_key][rc][metric]) if rc in agg[hi_key] else 0.0
        if hi == lo:
            return v_lo
        return v_lo + (drop - lo) / (hi - lo) * (v_hi - v_lo)

    # -----------------------------------------------------------------------
    # Print tables
    # -----------------------------------------------------------------------
    table_metrics = ["precision", "recall", "f1", "ndcg", "drop_pct", "n_highlighted"]
    print()
    print("=" * 85)
    print(f"Pruning evaluation — ACLSum '{args.split}' | trials={args.trials}")
    for k_frac in args.k_values:
        agg = all_agg[k_frac]
        print(f"\n{'─'*85}")
        print(f"k = {k_frac:.0%}")
        for method, label in METHODS.items():
            print(f"\n  {label}")
            print(f"    {'Read':>6}  " + "  ".join(f"{m:>12}" for m in table_metrics))
            print("    " + "-" * 70)
            for rc in READ_COUNTS:
                if rc not in agg[method]:
                    continue
                vals = "  ".join(f"{np.mean(agg[method][rc][m]):>12.3f}" for m in table_metrics)
                print(f"    {rc:>6}  {vals}")

    # -----------------------------------------------------------------------
    # vs-random comparison
    # -----------------------------------------------------------------------
    grid_methods = [m for m in METHODS if m.startswith("novel_thresh_")]
    print(f"\n\n{'='*85}")
    print("vs-Random comparison (Δ = method − random at same drop%)")
    for k_frac in args.k_values:
        agg = all_agg[k_frac]
        print(f"\n{'─'*85}")
        print(f"k = {k_frac:.0%}")
        for method in grid_methods:
            label = METHODS[method]
            print(f"\n  {label}")
            print(f"    {'Read':>6}  {'drop%':>7}  " +
                  "  ".join(f"{'Δ'+m:>10}" for m in COMPARE_METRICS) +
                  "  " + "  ".join(f"{'rand_'+m:>12}" for m in COMPARE_METRICS))
            print("    " + "-" * 75)
            for rc in READ_COUNTS:
                if rc not in agg[method]:
                    continue
                drop = np.mean(agg[method][rc]["drop_pct"])
                deltas    = [np.mean(agg[method][rc][m]) - interp_random(agg, drop, m, rc)
                             for m in COMPARE_METRICS]
                rand_vals = [interp_random(agg, drop, m, rc) for m in COMPARE_METRICS]
                print(f"    {rc:>6}  {drop:>7.1%}  " +
                      "  ".join(f"{d:>+10.3f}" for d in deltas) + "  " +
                      "  ".join(f"{v:>12.3f}" for v in rand_vals))

    # -----------------------------------------------------------------------
    # Save results to JSON for later visualization
    # -----------------------------------------------------------------------
    import json
    out = {
        "meta": {
            "split":   args.split,
            "trials":  args.trials,
            "k_values": args.k_values,
            "read_counts": READ_COUNTS,
            "methods": METHODS,
            "grid_lambdas": GRID_LAMBDAS,
            "grid_taus":    GRID_TAUS,
        },
        "results": {}
    }
    for k_frac in args.k_values:
        agg = all_agg[k_frac]
        k_key = f"{k_frac:.2f}"
        out["results"][k_key] = {}
        for method in METHODS:
            out["results"][k_key][method] = {}
            for rc in READ_COUNTS:
                if rc not in agg[method]:
                    continue
                out["results"][k_key][method][str(rc)] = {
                    m: float(np.mean(agg[method][rc][m]))
                    for m in table_metrics
                }
                # Also store vs-random deltas for grid methods
                if method in grid_methods:
                    drop = out["results"][k_key][method][str(rc)]["drop_pct"]
                    for m in COMPARE_METRICS:
                        rand_val = interp_random(agg, drop, m, rc)
                        out["results"][k_key][method][str(rc)][f"delta_{m}"] = \
                            out["results"][k_key][method][str(rc)][m] - rand_val

    fname = f"pruning_results_{args.split}.json"
    with open(fname, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {fname}")


if __name__ == "__main__":
    main()
