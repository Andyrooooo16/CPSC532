"""
eval_aclsum.py — Evaluate highlight coverage on the ACLSum extractive dataset.

ACLSum provides pre-split source_sentences for each paper, with gold subsets
for three topics: challenge, approach, and outcome. These map naturally to our
classifier facets:
    challenge  → BACKGROUND, OBJECTIVE
    approach   → METHODS
    outcome    → RESULTS, CONCLUSIONS

For each paper we run three selection methods and evaluate:
  1. Overall coverage  — what % of gold sentences appear in the highlights
  2. Facet-aligned     — same, but only counting sentences our classifier
                         placed in the matching facet group

Usage:
    python service/eval_aclsum.py
    python service/eval_aclsum.py --max-papers 20 --split test
"""
import sys
import random
import argparse
import numpy as np

sys.path.insert(0, ".")

from classifier import classify
from embedder import embed
from ranker import rank, _cosine_similarity_matrix, _pagerank

# Which classifier labels map to each ACLSum topic
TOPIC_FACETS = {
    "challenge": {"BACKGROUND", "OBJECTIVE"},
    "approach":  {"METHODS"},
    "outcome":   {"RESULTS", "CONCLUSIONS"},
}


# ---------------------------------------------------------------------------
# Selection methods — three NONE-handling conditions
# ---------------------------------------------------------------------------

def _pagerank_scores_all(embeddings):
    """Run PageRank on all sentences (including NONE). Returns score array."""
    sim = _cosine_similarity_matrix(embeddings)
    return _pagerank(sim)


def _sorted_none_before_topk(sentences, labels, embeddings):
    """Condition 1: NONE participates in PageRank graph, but is removed
    *before* the top-k cutoff. Top-k is computed over non-NONE sentences only.
    Returns non-NONE sentences sorted by score descending (caller applies top-k)."""
    scores = _pagerank_scores_all(embeddings)
    pairs = [(sc, s) for sc, s, l in zip(scores, sentences, labels) if l != "NONE"]
    pairs.sort(reverse=True)
    return [s for _, s in pairs]


def _sorted_none_after_topk(sentences, labels, embeddings):
    """Condition 2: NONE participates in PageRank graph AND in top-k selection.
    After taking the top-k% of *all* sentences, NONE are removed.
    Returns (all_sorted_sentences, all_sorted_labels) so the caller can apply
    the all-inclusive top-k cutoff then filter."""
    scores = _pagerank_scores_all(embeddings)
    triples = sorted(zip(scores, sentences, labels), reverse=True)
    all_sents  = [s for _, s, _ in triples]
    all_labels = [l for _, _, l in triples]
    return all_sents, all_labels


def _sorted_none_excluded(sentences, labels, confs, embeddings):
    """Condition 3: NONE sentences are removed *before* the PageRank graph is built.
    Top-k is computed over non-NONE sentences only.
    Returns non-NONE sentences sorted by score descending (caller applies top-k)."""
    empty = np.empty((0, embeddings.shape[1]), dtype=np.float32)
    results = rank(
        sentences, labels, confs, embeddings,
        read_sentences=[], read_embeddings=empty,
        top_k_fraction=1.0, global_top_k=False,
    )
    # rank() already excludes NONE before building the graph
    return [r["sentence"] for r in results]


# Maps classifier label → ACLSum topic group (used when --topic-groups is set)
LABEL_TO_TOPIC = {
    "BACKGROUND":  "challenge",
    "OBJECTIVE":   "challenge",
    "METHODS":     "approach",
    "RESULTS":     "outcome",
    "CONCLUSIONS": "outcome",
}


def _remap_labels(labels, use_topic_groups):
    """Optionally collapse 5 classifier labels into 3 ACLSum topic groups."""
    if not use_topic_groups:
        return labels
    return [LABEL_TO_TOPIC.get(l, l) for l in labels]


def _facet_conf_groups(sentences, labels, confs):
    """Return {label: [(conf, sent), ...]} sorted by confidence descending."""
    from collections import defaultdict
    groups = defaultdict(list)
    for sent, label, conf in zip(sentences, labels, confs):
        groups[label].append((conf, sent))
    for items in groups.values():
        items.sort(reverse=True)
    return groups


def _facet_tr_groups(sentences, labels, embeddings):
    """Return {label: [(score, sent), ...]} sorted by intra-facet PageRank descending."""
    from collections import defaultdict
    label_indices = defaultdict(list)
    for i, label in enumerate(labels):
        label_indices[label].append(i)

    groups = defaultdict(list)
    for label, indices in label_indices.items():
        facet_embs = embeddings[indices]
        if len(indices) == 1:
            groups[label].append((1.0, sentences[indices[0]]))
        else:
            scores = _pagerank(_cosine_similarity_matrix(facet_embs))
            for score, idx in sorted(zip(scores, indices), reverse=True):
                groups[label].append((score, sentences[idx]))
    return groups


def _select_k(sorted_sents, k_frac):
    k = max(1, int(len(sorted_sents) * k_frac))
    return sorted_sents[:k]


def _select_k_filter_after(all_sents, all_labels, n_total, k_frac):
    """Condition 2 selection: take top k% of n_total (all-inclusive), then drop NONE."""
    k = max(1, int(n_total * k_frac))
    return [s for s, l in zip(all_sents[:k], all_labels[:k]) if l != "NONE"]


def _flatten_groups(groups: dict) -> list[str]:
    """Merge non-NONE scored group items into a single list ranked by score descending.
    NONE sentences are excluded — they are never expected to be gold highlights."""
    all_items = [
        (score, sent)
        for label, items in groups.items()
        if label != "NONE"
        for score, sent in items
    ]
    all_items.sort(reverse=True)
    return [sent for _, sent in all_items]


K_BINS = 50  # number of evenly-spaced k-fraction points for curves

FACET_TR = "Facet+TR"


def curve_at_k(ranked: list[str], gold: list[str]) -> dict[str, list[float]]:
    """Compute P, R, F1, nDCG at every k=1..N, then bin into K_BINS fractions.

    Returns dict with keys 'k_frac', 'precision', 'recall', 'f1', 'ndcg',
    each a list of length K_BINS.
    """
    N = len(ranked)
    if N == 0 or not gold:
        empty = [0.0] * K_BINS
        return {"k_frac": list(np.linspace(1/N, 1.0, K_BINS)) if N else [],
                "precision": empty, "recall": empty, "f1": empty, "ndcg": empty}

    gold_set = {s.strip().lower() for s in gold}
    n_gold   = len(gold_set)

    # Pre-compute cumulative hits and DCG at every k
    hits_at  = [0] * (N + 1)
    dcg_at   = [0.0] * (N + 1)
    idcg     = sum(1.0 / np.log2(i + 2) for i in range(min(n_gold, N)))

    for i, sent in enumerate(ranked):
        hit = int(sent.strip().lower() in gold_set)
        hits_at[i + 1] = hits_at[i] + hit
        dcg_at[i + 1]  = dcg_at[i] + (hit / np.log2(i + 2))

    # Evaluate at K_BINS evenly-spaced cutoffs
    bin_fracs = np.linspace(1 / N, 1.0, K_BINS)
    result = {"k_frac": [], "precision": [], "recall": [], "f1": [], "ndcg": []}
    for frac in bin_fracs:
        k    = max(1, int(round(frac * N)))
        hits = hits_at[k]
        p    = hits / k
        r    = hits / n_gold if n_gold else 0.0
        f1   = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        ng   = dcg_at[k] / idcg if idcg > 0 else 0.0
        result["k_frac"].append(frac)
        result["precision"].append(p)
        result["recall"].append(r)
        result["f1"].append(f1)
        result["ndcg"].append(ng)

    return result


def curve_at_k_filter_after(all_sents, all_labels, gold):
    """Curve for condition 2: at each k%, take top-k of all sentences then drop NONE.
    The x-axis is still fraction of *all* sentences; y-axis reflects the post-filter set."""
    N = len(all_sents)
    if N == 0 or not gold:
        empty = [0.0] * K_BINS
        return {"k_frac": list(np.linspace(1/N, 1.0, K_BINS)) if N else [],
                "precision": empty, "recall": empty, "f1": empty, "ndcg": empty}

    gold_set = {s.strip().lower() for s in gold}
    n_gold   = len(gold_set)
    bin_fracs = np.linspace(1 / N, 1.0, K_BINS)

    # Pre-compute cumulative hits (over non-NONE only) at every position
    is_hit = [int(s.strip().lower() in gold_set and l != "NONE")
              for s, l in zip(all_sents, all_labels)]
    is_none = [int(l == "NONE") for l in all_labels]

    cum_hits = [0] * (N + 1)
    cum_nonnone = [0] * (N + 1)
    for i in range(N):
        cum_hits[i + 1]    = cum_hits[i] + is_hit[i]
        cum_nonnone[i + 1] = cum_nonnone[i] + (1 - is_none[i])

    # nDCG ideal: all gold at top
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(n_gold, N)))

    result = {"k_frac": [], "precision": [], "recall": [], "f1": [], "ndcg": []}
    for frac in bin_fracs:
        k        = max(1, int(round(frac * N)))
        hits     = cum_hits[k]
        nn       = cum_nonnone[k]          # non-NONE sentences in top-k
        p        = hits / nn if nn else 0.0
        r        = hits / n_gold if n_gold else 0.0
        f1       = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        # nDCG: discount by position in the *all-inclusive* ranking
        dcg = sum(is_hit[i] / np.log2(i + 2) for i in range(k))
        ng  = dcg / idcg if idcg > 0 else 0.0
        result["k_frac"].append(frac)
        result["precision"].append(p)
        result["recall"].append(r)
        result["f1"].append(f1)
        result["ndcg"].append(ng)

    return result


def select_from_groups(groups, k_frac):
    """Select top k_frac% from each non-NONE group."""
    selected = []
    for label, items in groups.items():
        if label == "NONE":
            continue
        k = max(1, int(len(items) * k_frac))
        selected.extend(sent for _, sent in items[:k])
    return selected


# ---------------------------------------------------------------------------
# Coverage / retrieval metrics
# ---------------------------------------------------------------------------

def is_covered(gold_sent: str, highlighted: list[str]) -> bool:
    """Exact string match (case-insensitive)."""
    g = gold_sent.strip().lower()
    return any(g == h.strip().lower() for h in highlighted)


def coverage(gold: list[str], highlighted: list[str]) -> float:
    if not gold:
        return 0.0
    return sum(1 for s in gold if is_covered(s, highlighted)) / len(gold)


def precision_recall(gold: list[str], highlighted: list[str]) -> tuple[float, float]:
    """Precision = relevant highlighted / all highlighted.
    Recall = relevant highlighted / all gold."""
    if not highlighted:
        return 0.0, 0.0
    gold_set = {s.strip().lower() for s in gold}
    hits = sum(1 for h in highlighted if h.strip().lower() in gold_set)
    precision = hits / len(highlighted)
    recall    = hits / len(gold) if gold else 0.0
    return precision, recall


def ndcg(gold: list[str], ranked: list[str]) -> float:
    """nDCG over the full ranked list with binary relevance.
    Rewards finding gold sentences earlier in the ranking."""
    if not gold or not ranked:
        return 0.0
    gold_set = {s.strip().lower() for s in gold}
    # DCG: sum of rel_i / log2(i+2) for i in 0..n-1
    dcg  = sum(1.0 / np.log2(i + 2)
               for i, s in enumerate(ranked)
               if s.strip().lower() in gold_set)
    # Ideal DCG: all gold sentences at the top
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(gold), len(ranked))))
    return dcg / idcg if idcg > 0 else 0.0


def facet_aligned_coverage(gold: list[str], topic: str,
                            sentences: list[str], labels: list[str],
                            highlighted: list[str]) -> float:
    """Coverage counting only highlighted sentences whose classifier label
    falls in the expected facets for this topic."""
    expected = TOPIC_FACETS[topic]
    label_map = dict(zip(sentences, labels))
    facet_highlights = [h for h in highlighted if label_map.get(h, "") in expected]
    return coverage(gold, facet_highlights)


def evaluate_paper(paper, paper_idx: int, total: int,
                   k_frac: float = 0.20,
                   n_random_trials: int = 100,
                   use_topic_groups: bool = False) -> dict | None:
    title = paper.get("id", str(paper_idx))
    print(f"[{paper_idx}/{total}] {title}")

    sentences = paper["source_sentences"]
    if not sentences or len(sentences) < 2:
        print("  Skipped — too few sentences\n")
        return None

    topics = {
        "challenge": paper["challenge_sentences"],
        "approach":  paper["approach_sentences"],
        "outcome":   paper["outcome_sentences"],
    }
    all_gold = [s for sents in topics.values() for s in sents]
    if not all_gold:
        print("  Skipped — no gold sentences\n")
        return None

    print(f"  {len(sentences)} sentences  |  "
          f"challenge:{len(topics['challenge'])}  "
          f"approach:{len(topics['approach'])}  "
          f"outcome:{len(topics['outcome'])}")

    classifications = classify(sentences)
    labels = [c[0] for c in classifications]
    confs  = [c[1] for c in classifications]
    embeddings = embed(sentences)

    # Optionally collapse 5 classifier labels into 3 ACLSum topic groups
    group_labels = _remap_labels(labels, use_topic_groups)

    conf_groups = _facet_conf_groups(sentences, group_labels, confs)
    ftr_groups  = _facet_tr_groups(sentences, group_labels, embeddings)

    # --- Three NONE-handling conditions ---
    # Condition 1: NONE in graph, filtered before top-k
    nb_sorted = _sorted_none_before_topk(sentences, labels, embeddings)
    nb_h      = _select_k(nb_sorted, k_frac)

    # Condition 2: NONE in graph, filtered after top-k
    na_all_sents, na_all_labels = _sorted_none_after_topk(sentences, labels, embeddings)
    na_h = _select_k_filter_after(na_all_sents, na_all_labels, len(sentences), k_frac)

    # Condition 3: NONE excluded from graph entirely
    ne_sorted = _sorted_none_excluded(sentences, labels, confs, embeddings)
    ne_h      = _select_k(ne_sorted, k_frac)

    fa_h  = select_from_groups(conf_groups, k_frac)
    ftr_h = select_from_groups(ftr_groups, k_frac)
    k_abs = max(1, int(len(sentences) * k_frac))

    def _score(highlighted):
        p, r = precision_recall(all_gold, highlighted)
        row = {
            "overall":   coverage(all_gold, highlighted),
            "precision": p,
            "recall":    r,
            "ndcg":      ndcg(all_gold, highlighted),
        }
        for topic, gold in topics.items():
            row[topic] = coverage(gold, highlighted)
            row[f"{topic}_aligned"] = facet_aligned_coverage(
                gold, topic, sentences, labels, highlighted)
        return row

    nb_scores  = _score(nb_h)
    na_scores  = _score(na_h)
    ne_scores  = _score(ne_h)
    fa_scores  = _score(fa_h)
    ftr_scores = _score(ftr_h)

    # Average random baseline over n_random_trials
    rand_acc = None
    for _ in range(n_random_trials):
        s = _score(random.sample(sentences, k_abs))
        if rand_acc is None:
            rand_acc = dict(s)
        else:
            for k, v in s.items():
                rand_acc[k] += v
    rand_scores = {k: v / n_random_trials for k, v in rand_acc.items()}

    def _fmt(scores):
        return (f"P:{scores['precision']:.0%}  R:{scores['recall']:.0%}  "
                f"nDCG:{scores['ndcg']:.3f}  "
                f"challenge:{scores['challenge']:.0%}  "
                f"approach:{scores['approach']:.0%}  "
                f"outcome:{scores['outcome']:.0%}")

    print(f"  NoneBeforeK — {_fmt(nb_scores)}")
    print(f"  NoneAfterK  — {_fmt(na_scores)}")
    print(f"  NoneExcl    — {_fmt(ne_scores)}")
    print(f"  Facet       — {_fmt(fa_scores)}")
    print(f"  Facet+TR    — {_fmt(ftr_scores)}")
    print(f"  Random      — {_fmt(rand_scores)}\n")

    # Confusion data: for each gold sentence, record its classifier label
    label_map = dict(zip(sentences, labels))
    confusion = {
        topic: [label_map[s] for s in gold if s in label_map]
        for topic, gold in topics.items()
    }

    # Full global rankings for curve computation
    fa_ranked   = _flatten_groups(conf_groups)
    ftr_ranked  = _flatten_groups(ftr_groups)
    rand_ranked = random.sample(sentences, len(sentences))

    def _curves_for_gold(gold_sents):
        return {
            "nb":   curve_at_k(nb_sorted,    gold_sents),
            "na":   curve_at_k_filter_after(na_all_sents, na_all_labels, gold_sents),
            "ne":   curve_at_k(ne_sorted,    gold_sents),
            "fa":   curve_at_k(fa_ranked,    gold_sents),
            "ftr":  curve_at_k(ftr_ranked,   gold_sents),
            "rand": curve_at_k(rand_ranked,  gold_sents),
        }

    curves = {
        "overall":   _curves_for_gold(all_gold),
        "challenge": _curves_for_gold(topics["challenge"]),
        "approach":  _curves_for_gold(topics["approach"]),
        "outcome":   _curves_for_gold(topics["outcome"]),
    }

    return {
        "n_sentences": len(sentences),
        "confusion":   confusion,
        "curves":      curves,
        **{f"nb_{k}":   v for k, v in nb_scores.items()},
        **{f"na_{k}":   v for k, v in na_scores.items()},
        **{f"ne_{k}":   v for k, v in ne_scores.items()},
        **{f"fa_{k}":   v for k, v in fa_scores.items()},
        **{f"ftr_{k}":  v for k, v in ftr_scores.items()},
        **{f"rand_{k}": v for k, v in rand_scores.items()},
    }



# ---------------------------------------------------------------------------
# Reporting helpers (extracted from main to keep complexity manageable)
# ---------------------------------------------------------------------------

def _print_tables(results, methods, mean):
    topics = ["overall", "challenge", "approach", "outcome"]

    print()
    print(f"{'Method':<14} " + "  ".join(f"{t:>10}" for t in topics))
    print("-" * 60)
    for label, prefix in methods:
        vals = "  ".join(f"{mean(f'{prefix}_{t}'):>10.1%}" for t in topics)
        print(f"{label:<14} {vals}")

    print()
    print(f"{'Method':<14} {'Precision':>10}  {'Recall':>8}  {'nDCG':>8}")
    print("-" * 48)
    for label, prefix in methods:
        p = mean(f"{prefix}_precision")
        r = mean(f"{prefix}_recall")
        n = mean(f"{prefix}_ndcg")
        print(f"{label:<14} {p:>10.1%}  {r:>8.1%}  {n:>8.3f}")

    print()
    print("Facet-aligned coverage (sentences classified into matching facet)")
    facet_topics = ["challenge", "approach", "outcome"]
    print(f"{'Method':<14} " + "  ".join(f"{t:>10}" for t in facet_topics))
    print("-" * 50)
    for label, prefix in methods[:-1]:  # skip Random
        vals = "  ".join(f"{mean(f'{prefix}_{t}_aligned'):>10.1%}" for t in facet_topics)
        print(f"{label:<14} {vals}")


def _print_significance(results, mean):
    from scipy.stats import wilcoxon, ttest_rel
    rand_overall = [r["rand_overall"] for r in results]

    sig_methods = [
        ("NoneBeforeK", "nb"), ("NoneAfterK", "na"), ("NoneExcl", "ne"),
        ("Facet", "fa"), (FACET_TR, "ftr"),
    ]
    print()
    print(f"Statistical significance — overall coverage (n={len(results)} papers)")
    print(f"  {'Method':<14} {'> random':>9}  {'Wilcoxon p':>11}  {'t-test p':>10}")
    print(f"  {'-'*50}")
    for label, prefix in sig_methods:
        actual   = [r[f"{prefix}_overall"] for r in results]
        diffs    = [a - b for a, b in zip(actual, rand_overall)]
        n_better = sum(1 for d in diffs if d > 0)
        if len(results) >= 5 and len(set(diffs)) > 1:
            _, p_w = wilcoxon(actual, rand_overall, alternative="greater")
            _, p_t = ttest_rel(actual, rand_overall, alternative="greater")
            print(f"  {label:<14} {n_better}/{len(results):>6}  {p_w:>11.4f}  {p_t:>10.4f}")
        else:
            print(f"  {label:<14} {n_better}/{len(results):>6}  {'n/a':>11}  {'n/a':>10}")


def _print_confusion(results):
    from collections import Counter
    ALL_LABELS = ["BACKGROUND", "OBJECTIVE", "METHODS", "RESULTS", "CONCLUSIONS", "NONE"]
    ACL_TOPICS = ["challenge", "approach", "outcome"]

    counts = {topic: Counter() for topic in ACL_TOPICS}
    for r in results:
        for topic in ACL_TOPICS:
            counts[topic].update(r["confusion"][topic])
    totals = {topic: sum(counts[topic].values()) for topic in ACL_TOPICS}

    print()
    print("Classifier label distribution for gold sentences")
    print("(rows = ACLSum topic, columns = predicted classifier label)")
    print()
    col_w  = 13
    header = f"  {'Topic':<12}" + "".join(f"{l:>{col_w}}" for l in ALL_LABELS) + f"  {'Total':>7}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for topic in ACL_TOPICS:
        n   = totals[topic]
        row = f"  {topic:<12}"
        for label in ALL_LABELS:
            c    = counts[topic][label]
            row += f"  {c:>4} ({c/n:>4.0%})" if n else f"  {'—':>9}"
        row += f"  {n:>7}"
        print(row)


def _plot_curves(results, curve_methods, split, n_papers):
    import matplotlib.pyplot as plt

    metrics       = ["precision", "recall", "f1", "ndcg"]
    metric_labels = {"precision": "Precision", "recall": "Recall", "f1": "F1", "ndcg": "nDCG"}
    colors        = {
        "nb": "#1f77b4", "na": "#d62728", "ne": "#2ca02c",
        "fa": "#ff7f0e", "ftr": "#8c564b", "rand": "#9467bd",
    }
    x_frac       = np.linspace(1 / K_BINS, 1.0, K_BINS)
    curve_groups = ["overall", "challenge", "approach", "outcome"]
    group_titles = {"overall": "Overall", "challenge": "Challenge",
                    "approach": "Approach", "outcome": "Outcome"}

    avg_curves = {
        group: {
            prefix: {
                metric: np.mean([r["curves"][group][prefix][metric] for r in results], axis=0)
                for metric in metrics
            }
            for _, prefix in curve_methods
        }
        for group in curve_groups
    }

    saved = []
    for group in curve_groups:
        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        for ax, metric in zip(axes.flatten(), metrics):
            for label, prefix in curve_methods:
                ax.plot(x_frac * 100, avg_curves[group][prefix][metric],
                        label=label, color=colors[prefix],
                        linestyle="--" if prefix == "rand" else "-", linewidth=1.8)
            ax.set_xlabel("k (% of sentences selected)")
            ax.set_ylabel(metric_labels[metric])
            ax.set_title(metric_labels[metric])
            ax.legend()
            ax.set_xlim(0, 100)
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)

        fig.suptitle(f"{group_titles[group]} — ACLSum '{split}' split ({n_papers} papers)",
                     fontsize=13)
        plt.tight_layout()
        fname = f"ranking_curves_{split}_{group}.png"
        plt.savefig(fname, dpi=150)
        plt.show()
        saved.append(fname)

    print(f"\nCurves saved to: {', '.join(saved)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-papers", type=int, default=None)
    parser.add_argument("--split", default="validation",
                        choices=["train", "validation", "test"])
    parser.add_argument("--k", type=float, default=0.20,
                        help="Fraction of sentences to highlight (default: 0.20)")
    parser.add_argument("--random-trials", type=int, default=100)
    parser.add_argument("--topic-groups", action="store_true",
                        help="Group by 3 ACLSum topics instead of 5 classifier labels")
    args = parser.parse_args()

    print("Loading ACLSum dataset (extractive config)...")
    from datasets import load_dataset
    ds = load_dataset("sobamchan/aclsum", "extractive", split=args.split)

    if args.max_papers:
        ds = ds.select(range(min(args.max_papers, len(ds))))

    print(f"Evaluating {len(ds)} papers from '{args.split}' split...\n")

    results, skipped = [], 0
    for i, paper in enumerate(ds, start=1):
        r = evaluate_paper(paper, i, len(ds),
                           k_frac=args.k,
                           n_random_trials=args.random_trials,
                           use_topic_groups=args.topic_groups)
        if r is None:
            skipped += 1
        else:
            results.append(r)

    if not results:
        print("No papers evaluated.")
        return

    def mean(key):
        vals = [r[key] for r in results if key in r]
        return sum(vals) / len(vals) if vals else 0.0

    print("=" * 75)
    print(f"Split            : {args.split}")
    print(f"k (highlight %)  : {args.k:.0%}")
    print(f"Papers evaluated : {len(results)}")
    print(f"Papers skipped   : {skipped}")

    methods = [
        ("NoneBeforeK", "nb"),
        ("NoneAfterK",  "na"),
        ("NoneExcl",    "ne"),
        ("Facet",       "fa"),
        (FACET_TR,      "ftr"),
        ("Random",      "rand"),
    ]

    _print_tables(results, methods, mean)
    _print_significance(results, mean)
    _print_confusion(results)
    _plot_curves(results, methods, args.split, len(results))


if __name__ == "__main__":
    main()
