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


def _rank_then_group(sentences, labels, embeddings, exclude_none_from_graph: bool):
    """Global PageRank first, then group by label.

    If exclude_none_from_graph=True, NONE sentences are removed before building
    the similarity graph. Otherwise they participate in PageRank but are dropped
    from the returned groups.

    Returns {label: [(global_score, sent), ...]} sorted descending, NONE omitted.
    """
    from collections import defaultdict

    if exclude_none_from_graph:
        keep_idx = [i for i, l in enumerate(labels) if l != "NONE"]
        if not keep_idx:
            return {}
        sents_in  = [sentences[i] for i in keep_idx]
        labels_in = [labels[i]    for i in keep_idx]
        embs_in   = embeddings[keep_idx]
    else:
        sents_in  = sentences
        labels_in = labels
        embs_in   = embeddings

    scores = _pagerank(_cosine_similarity_matrix(embs_in))

    groups = defaultdict(list)
    for score, sent, label in zip(scores, sents_in, labels_in):
        if label != "NONE":
            groups[label].append((score, sent))
    for items in groups.values():
        items.sort(reverse=True)
    return dict(groups)


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


def curve_at_k(ranked: list[str], gold: list[str],
               n_total: int | None = None) -> dict[str, list[float]]:
    """Compute P, R, F1, nDCG binned into K_BINS fractions.

    n_total: if provided, use as the x-axis reference so all methods share the
             same scale (k% of all sentences). When k exceeds len(ranked) the
             method has exhausted its pool — metrics flatline at their final value.
    """
    n_ranked = len(ranked)
    n        = n_total if n_total is not None else n_ranked

    if n_ranked == 0 or not gold:
        empty = [0.0] * K_BINS
        return {"k_frac": list(np.linspace(1/max(n, 1), 1.0, K_BINS)),
                "precision": empty, "recall": empty, "f1": empty, "ndcg": empty}

    gold_set = {s.strip().lower() for s in gold}
    n_gold   = len(gold_set)

    # Pre-compute cumulative hits and DCG at every position in the ranked list
    hits_at = [0] * (n_ranked + 1)
    dcg_at  = [0.0] * (n_ranked + 1)
    idcg    = sum(1.0 / np.log2(i + 2) for i in range(min(n_gold, n_ranked)))

    for i, sent in enumerate(ranked):
        hit = int(sent.strip().lower() in gold_set)
        hits_at[i + 1] = hits_at[i] + hit
        dcg_at[i + 1]  = dcg_at[i] + (hit / np.log2(i + 2))

    bin_fracs = np.linspace(1 / n, 1.0, K_BINS)
    result = {"k_frac": [], "precision": [], "recall": [], "f1": [], "ndcg": []}
    for frac in bin_fracs:
        k   = max(1, int(round(frac * n)))
        k_c = min(k, n_ranked)   # clip to pool size — flatlines beyond this
        hits = hits_at[k_c]
        p    = hits / k_c
        r    = hits / n_gold if n_gold else 0.0
        f1   = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        ng   = dcg_at[k_c] / idcg if idcg > 0 else 0.0
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


def curve_at_k_optimal(n_total: int, achievable_gold: list[str],
                        all_gold: list[str], topics_gold: dict[str, list[str]],
                        achievable_topics: dict[str, list[str]],
                        cap: int | None = None) -> dict[str, list[float]]:
    """Oracle upper-bound curve, both variants on the same n_total x-axis.

    cap: maximum sentences the oracle can select (n_non_none for the non-NONE
         oracle, None for the all-sentence oracle). Metrics flatline once k > cap.

    achievable_gold: gold sentences the oracle can actually reach given its pool.
    all_gold: all gold sentences — used as the recall denominator so both oracles
              are comparable on the same recall scale.
    """
    N   = n_total
    m   = len(all_gold)          # recall denominator (same for both oracles)
    m_a = len(achievable_gold)   # hits ceiling given the oracle's pool
    pool_cap = cap if cap is not None else N
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(m_a, pool_cap)))

    bin_fracs = np.linspace(1 / N, 1.0, K_BINS)
    result = {
        "k_frac": [], "precision": [], "recall": [], "f1": [], "ndcg": [],
        **{t: [] for t in topics_gold},
    }
    for frac in bin_fracs:
        k    = max(1, int(round(frac * N)))
        k_c  = min(k, pool_cap)       # flatline beyond the pool
        hits = min(k_c, m_a)
        p    = hits / k_c
        r    = hits / m if m else 0.0
        f1   = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        dcg  = sum(1.0 / np.log2(i + 2) for i in range(hits))
        ng   = dcg / idcg if idcg > 0 else 0.0

        result["k_frac"].append(frac)
        result["precision"].append(p)
        result["recall"].append(r)
        result["f1"].append(f1)
        result["ndcg"].append(ng)
        for topic, tgold in topics_gold.items():
            m_t   = len(tgold)
            m_t_a = len(achievable_topics.get(topic, []))
            result[topic].append(min(k_c, m_t_a) / m_t if m_t else 0.0)

    return result


def score_optimal(all_gold: list[str], topics_gold: dict[str, list[str]],
                  n_total: int, k_frac: float,
                  achievable_gold: list[str],
                  achievable_topics: dict[str, list[str]],
                  cap: int | None = None) -> dict:
    """Fixed-k oracle scores. cap=None → all-sentence oracle; cap=n_non_none → non-NONE oracle."""
    m    = len(all_gold)
    m_a  = len(achievable_gold)
    pool_cap = cap if cap is not None else n_total
    k    = max(1, int(n_total * k_frac))
    k_c  = min(k, pool_cap)
    hits = min(k_c, m_a)
    p    = hits / k_c
    r    = hits / m if m else 0.0
    f1   = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(m_a, pool_cap)))
    dcg  = sum(1.0 / np.log2(i + 2) for i in range(hits))
    row  = {
        "overall":   r,
        "precision": p,
        "recall":    r,
        "f1":        f1,
        "ndcg":      dcg / idcg if idcg > 0 else 0.0,
    }
    for topic, tgold in topics_gold.items():
        m_t   = len(tgold)
        m_t_a = len(achievable_topics.get(topic, []))
        opt_t = min(k_c, m_t_a) / m_t if m_t else 0.0
        row[topic]              = opt_t
        row[f"{topic}_aligned"] = opt_t
    return row


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


def _fmt_scores(scores: dict) -> str:
    return (f"P:{scores['precision']:.0%}  R:{scores['recall']:.0%}  "
            f"F1:{scores['f1']:.0%}  nDCG:{scores['ndcg']:.3f}  "
            f"challenge:{scores['challenge']:.0%}  "
            f"approach:{scores['approach']:.0%}  "
            f"outcome:{scores['outcome']:.0%}")


def _paper_score(highlighted: list[str], all_gold: list[str],
                 topics: dict, sentences: list[str], labels: list[str]) -> dict:
    p, r = precision_recall(all_gold, highlighted)
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    row = {
        "overall":   coverage(all_gold, highlighted),
        "precision": p, "recall": r, "f1": f1,
        "ndcg":      ndcg(all_gold, highlighted),
    }
    for topic, gold in topics.items():
        row[topic] = coverage(gold, highlighted)
        row[f"{topic}_aligned"] = facet_aligned_coverage(
            gold, topic, sentences, labels, highlighted)
    return row


def _topic_curve_from_full(full_curve: dict, topic_key: str) -> dict:
    """Slice a topic-specific recall column out of a full optimal-curve dict."""
    t_recall = full_curve[topic_key]
    return {
        "k_frac":    full_curve["k_frac"],
        "precision": full_curve["precision"],
        "recall":    t_recall,
        "f1":        [2*p*r/(p+r) if (p+r) > 0 else 0.0
                      for p, r in zip(full_curve["precision"], t_recall)],
        "ndcg":      full_curve["ndcg"],
    }


def _build_curves(ranked_lists: tuple, gold_sents: list, n_total: int,
                  opt_curve: dict, onn_curve: dict,
                  topic_key: str | None) -> dict:
    """Build the per-method curve dict for one gold set.

    ranked_lists is the 9-tuple:
        (nb_sorted, na_all_sents, na_all_labels, ne_sorted,
         fa_ranked, ftr_ranked, rgw_ranked, rge_ranked, rand_ranked)
    """
    nb, na_sents, na_labels, ne, fa, ftr, rgw, rge, rand = ranked_lists
    base = {
        "nb":   curve_at_k(nb,       gold_sents, n_total),
        "na":   curve_at_k_filter_after(na_sents, na_labels, gold_sents),
        "ne":   curve_at_k(ne,       gold_sents, n_total),
        "fa":   curve_at_k(fa,       gold_sents, n_total),
        "ftr":  curve_at_k(ftr,      gold_sents, n_total),
        "rgw":  curve_at_k(rgw,      gold_sents, n_total),
        "rge":  curve_at_k(rge,      gold_sents, n_total),
        "rand": curve_at_k(rand,     gold_sents),
    }
    if topic_key is not None:
        base["opt"] = _topic_curve_from_full(opt_curve, topic_key)
        base["onn"] = _topic_curve_from_full(onn_curve, topic_key)
    else:
        base["opt"] = opt_curve
        base["onn"] = onn_curve
    return base


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
    rgw_groups  = _rank_then_group(sentences, group_labels, embeddings, exclude_none_from_graph=False)
    rge_groups  = _rank_then_group(sentences, group_labels, embeddings, exclude_none_from_graph=True)

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
    rgw_h = select_from_groups(rgw_groups, k_frac)
    rge_h = select_from_groups(rge_groups, k_frac)
    k_abs = max(1, int(len(sentences) * k_frac))

    nb_scores  = _paper_score(nb_h,  all_gold, topics, sentences, labels)
    na_scores  = _paper_score(na_h,  all_gold, topics, sentences, labels)
    ne_scores  = _paper_score(ne_h,  all_gold, topics, sentences, labels)
    fa_scores  = _paper_score(fa_h,  all_gold, topics, sentences, labels)
    ftr_scores = _paper_score(ftr_h, all_gold, topics, sentences, labels)
    rgw_scores = _paper_score(rgw_h, all_gold, topics, sentences, labels)
    rge_scores = _paper_score(rge_h, all_gold, topics, sentences, labels)
    # Compute oracle reference values.
    n_total    = len(sentences)
    n_non_none = len(ne_sorted)   # rank_none_excluded already filtered NONE
    sent_set   = {s.strip().lower() for s in sentences}
    label_map  = dict(zip(sentences, labels))

    # Gold reachable by the all-sentence oracle (present in source_sentences)
    achievable_all_gold   = [g for g in all_gold if g.strip().lower() in sent_set]
    achievable_all_topics = {
        t: [g for g in tgold if g.strip().lower() in sent_set]
        for t, tgold in topics.items()
    }
    # Gold reachable by the non-NONE oracle (present AND non-NONE)
    achievable_nn_gold   = [g for g in achievable_all_gold
                            if label_map.get(g, "NONE") != "NONE"]
    achievable_nn_topics = {
        t: [g for g in tgold if g.strip().lower() in sent_set
            and label_map.get(g, "NONE") != "NONE"]
        for t, tgold in topics.items()
    }

    opt_scores = score_optimal(all_gold, topics, n_total, k_frac,
                               achievable_all_gold, achievable_all_topics,
                               cap=None)
    onn_scores = score_optimal(all_gold, topics, n_total, k_frac,
                               achievable_nn_gold, achievable_nn_topics,
                               cap=n_non_none)

    # Average random baseline over n_random_trials
    rand_acc: dict = {}
    for _ in range(n_random_trials):
        s = _paper_score(random.sample(sentences, k_abs), all_gold, topics, sentences, labels)
        for key, val in s.items():
            rand_acc[key] = rand_acc.get(key, 0.0) + val
    rand_scores = {k: v / n_random_trials for k, v in rand_acc.items()}

    print(f"  Opt-All     — {_fmt_scores(opt_scores)}")
    print(f"  Opt-NN      — {_fmt_scores(onn_scores)}")
    print(f"  NoneBeforeK — {_fmt_scores(nb_scores)}")
    print(f"  NoneAfterK  — {_fmt_scores(na_scores)}")
    print(f"  NoneExcl    — {_fmt_scores(ne_scores)}")
    print(f"  Facet       — {_fmt_scores(fa_scores)}")
    print(f"  Facet+TR    — {_fmt_scores(ftr_scores)}")
    print(f"  RankGrp+N   — {_fmt_scores(rgw_scores)}")
    print(f"  RankGrp-N   — {_fmt_scores(rge_scores)}")
    print(f"  Random      — {_fmt_scores(rand_scores)}\n")

    # Confusion data: for each gold sentence, record its classifier label
    label_map = dict(zip(sentences, labels))
    confusion = {
        topic: [label_map[s] for s in gold if s in label_map]
        for topic, gold in topics.items()
    }

    # Full global rankings for curve computation — all use n_total as x-axis
    fa_ranked   = _flatten_groups(conf_groups)
    ftr_ranked  = _flatten_groups(ftr_groups)
    rgw_ranked  = _flatten_groups(rgw_groups)
    rge_ranked  = _flatten_groups(rge_groups)
    rand_ranked = random.sample(sentences, n_total)

    opt_curve = curve_at_k_optimal(n_total, achievable_all_gold, all_gold, topics,
                                   achievable_all_topics, cap=None)
    onn_curve = curve_at_k_optimal(n_total, achievable_nn_gold,  all_gold, topics,
                                   achievable_nn_topics,  cap=n_non_none)

    ranked_lists = (nb_sorted, na_all_sents, na_all_labels, ne_sorted,
                    fa_ranked, ftr_ranked, rgw_ranked, rge_ranked, rand_ranked)
    curves = {
        "overall": _build_curves(ranked_lists, all_gold, n_total,
                                 opt_curve, onn_curve, topic_key=None),
        "challenge": _build_curves(ranked_lists, topics["challenge"], n_total,
                                   opt_curve, onn_curve, topic_key="challenge"),
        "approach": _build_curves(ranked_lists, topics["approach"], n_total,
                                  opt_curve, onn_curve, topic_key="approach"),
        "outcome": _build_curves(ranked_lists, topics["outcome"], n_total,
                                 opt_curve, onn_curve, topic_key="outcome"),
    }

    return {
        "n_sentences": len(sentences),
        "confusion":   confusion,
        "curves":      curves,
        **{f"opt_{k}":  v for k, v in opt_scores.items()},
        **{f"nb_{k}":   v for k, v in nb_scores.items()},
        **{f"na_{k}":   v for k, v in na_scores.items()},
        **{f"ne_{k}":   v for k, v in ne_scores.items()},
        **{f"fa_{k}":   v for k, v in fa_scores.items()},
        **{f"ftr_{k}":  v for k, v in ftr_scores.items()},
        **{f"rgw_{k}":  v for k, v in rgw_scores.items()},
        **{f"rge_{k}":  v for k, v in rge_scores.items()},
        **{f"rand_{k}": v for k, v in rand_scores.items()},
    }



# ---------------------------------------------------------------------------
# Reporting helpers (extracted from main to keep complexity manageable)
# ---------------------------------------------------------------------------

def _print_tables(methods, mean):
    topics = ["overall", "challenge", "approach", "outcome"]

    print()
    print(f"{'Method':<14} " + "  ".join(f"{t:>10}" for t in topics))
    print("-" * 60)
    for label, prefix in methods:
        vals = "  ".join(f"{mean(f'{prefix}_{t}'):>10.1%}" for t in topics)
        print(f"{label:<14} {vals}")

    print()
    print(f"{'Method':<14} {'Precision':>10}  {'Recall':>8}  {'F1':>8}  {'nDCG':>8}")
    print("-" * 57)
    for label, prefix in methods:
        p  = mean(f"{prefix}_precision")
        r  = mean(f"{prefix}_recall")
        f1 = mean(f"{prefix}_f1")
        n  = mean(f"{prefix}_ndcg")
        print(f"{label:<14} {p:>10.1%}  {r:>8.1%}  {f1:>8.1%}  {n:>8.3f}")

    print()
    print("Facet-aligned coverage (sentences classified into matching facet)")
    facet_topics = ["challenge", "approach", "outcome"]
    print(f"{'Method':<14} " + "  ".join(f"{t:>10}" for t in facet_topics))
    print("-" * 50)
    for label, prefix in methods[:-1]:  # skip Random
        vals = "  ".join(f"{mean(f'{prefix}_{t}_aligned'):>10.1%}" for t in facet_topics)
        print(f"{label:<14} {vals}")


def _print_significance(results):
    from scipy.stats import wilcoxon, ttest_rel
    rand_overall = [r["rand_overall"] for r in results]

    sig_methods = [
        ("NoneBeforeK", "nb"), ("NoneAfterK", "na"), ("NoneExcl", "ne"),
        ("Facet", "fa"), (FACET_TR, "ftr"), ("RankGrp+N", "rgw"), ("RankGrp-N", "rge"),
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


_CURVE_COLORS = {
    "opt": "#000000",
    "nb": "#1f77b4", "na": "#d62728", "ne": "#2ca02c",
    "fa": "#ff7f0e", "ftr": "#8c564b",
    "rgw": "#17becf", "rge": "#bcbd22",
    "rand": "#9467bd",
}


def _line_style(prefix):
    if prefix == "opt":
        return ":", 2.2
    if prefix == "rand":
        return "--", 1.8
    return "-", 1.8


def _plot_group_figure(group, title, avg_curves, curve_methods, x_frac,
                        metrics, metric_labels, split, n_papers):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, metric in zip(axes.flatten(), metrics):
        for label, prefix in curve_methods:
            ls, lw = _line_style(prefix)
            ax.plot(x_frac * 100, avg_curves[group][prefix][metric],
                    label=label, color=_CURVE_COLORS[prefix], linestyle=ls, linewidth=lw)
        ax.set_xlabel("k (% of sentences selected)")
        ax.set_ylabel(metric_labels[metric])
        ax.set_title(metric_labels[metric])
        ax.legend()
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"{title} — ACLSum '{split}' split ({n_papers} papers)", fontsize=13)
    plt.tight_layout()
    fname = f"ranking_curves_{split}_{group}.png"
    plt.savefig(fname, dpi=150)
    plt.show()
    return fname


def _plot_curves(results, curve_methods, split, n_papers):
    metrics       = ["precision", "recall", "f1", "ndcg"]
    metric_labels = {"precision": "Precision", "recall": "Recall", "f1": "F1", "ndcg": "nDCG"}
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
        fname = _plot_group_figure(
            group, group_titles[group], avg_curves, curve_methods,
            x_frac, metrics, metric_labels, split, n_papers,
        )
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
    from datasets import load_dataset, Dataset
    raw = load_dataset("sobamchan/aclsum", "extractive", split=args.split)
    assert isinstance(raw, Dataset)
    ds: Dataset = raw

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
        ("Optimal",     "opt"),
        ("NoneBeforeK", "nb"),
        ("NoneAfterK",  "na"),
        ("NoneExcl",    "ne"),
        ("Facet",       "fa"),
        (FACET_TR,      "ftr"),
        ("RankGrp+N",   "rgw"),
        ("RankGrp-N",   "rge"),
        ("Random",      "rand"),
    ]

    _print_tables(methods, mean)
    _print_significance(results)
    _print_confusion(results)
    _plot_curves(results, methods, args.split, len(results))


if __name__ == "__main__":
    main()
