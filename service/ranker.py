import numpy as np


def _cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarity matrix."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-9, norms)
    normed = embeddings / norms
    return normed @ normed.T


def _pagerank(sim_matrix: np.ndarray, damping: float = 0.85, max_iter: int = 100, tol: float = 1e-6) -> np.ndarray:
    """Run PageRank on a similarity matrix used as an adjacency matrix."""
    n = sim_matrix.shape[0]

    # Zero the diagonal (no self-loops)
    np.fill_diagonal(sim_matrix, 0.0)

    # Row-normalize to get transition probabilities
    row_sums = sim_matrix.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1e-9, row_sums)
    transition = sim_matrix / row_sums

    # Power iteration
    scores = np.ones(n) / n
    for _ in range(max_iter):
        new_scores = (1 - damping) / n + damping * transition.T @ scores
        if np.abs(new_scores - scores).sum() < tol:
            break
        scores = new_scores

    return scores


def rank(
    target_sentences: list[str],
    target_labels: list[str],
    target_confidences: list[float],
    target_embeddings: np.ndarray,
    read_sentences: list[str],
    read_embeddings: np.ndarray,
    top_k_fraction: float = 0.20,
    global_top_k: bool = True,
) -> list[dict]:
    """
    Rank target_sentences using TextRank, with read_sentences competing
    in the same graph to push down already-familiar ideas.

    Returns the top_k_fraction of target sentences, sorted by rank score,
    each with sentence, label, confidence, and rank fields.
    """
    n_target = len(target_sentences)
    n_read = len(read_sentences)

    # Build combined embedding matrix: target first, then read
    if n_read > 0:
        all_embeddings = np.vstack([target_embeddings, read_embeddings])
    else:
        all_embeddings = target_embeddings

    print(f"  Building similarity matrix ({n_target} target + {n_read} read sentences)...")
    sim_matrix = _cosine_similarity_matrix(all_embeddings)

    print("  Running PageRank...")
    scores = _pagerank(sim_matrix)

    target_scores = scores[:n_target]

    # k is always relative to the target paper, not the global corpus
    top_k = max(1, int(n_target * top_k_fraction))

    if global_top_k and n_read > 0:
        # Read sentences compete in the graph — only target sentences that rank
        # in the global top-k (out of n_target slots) are returned.
        # k is fixed to n_target so it doesn't grow with read corpus size.
        global_top_indices = set(np.argsort(scores)[::-1][:top_k].tolist())
        ranked_indices = [i for i in np.argsort(target_scores)[::-1]
                          if i in global_top_indices]
    else:
        ranked_indices = np.argsort(target_scores)[::-1][:top_k].tolist()

    results = []
    for rank_position, idx in enumerate(ranked_indices, start=1):
        results.append({
            "sentence": target_sentences[idx],
            "label": target_labels[idx],
            "confidence": round(float(target_confidences[idx]), 4),
            "rank": rank_position,
            "score": round(float(target_scores[idx]), 6),
        })

    return results


def rank_novel(
    target_sentences: list[str],
    target_labels: list[str],
    target_confidences: list[float],
    target_embeddings: np.ndarray,
    read_sentences: list[str],
    read_embeddings: np.ndarray,
    top_k_fraction: float = 0.20,
    novelty_lambda: float = 0.5,
    sim_mode: str = "max",  # "max" | "mean"
    threshold: float | None = None,
) -> list[dict]:
    """
    Rank target sentences by novelty-penalized PageRank score.

    Instead of letting read sentences compete for top-k slots (global_top_k),
    this always selects top-k from target sentences only, but penalizes sentences
    that are similar to already-read content:

        final_score = pagerank_score - lambda * sim_to_read

    where sim_to_read is either max or mean cosine similarity to read sentences.

    If threshold is set, any target sentence with sim_to_read >= threshold is
    excluded entirely regardless of lambda (hard suppression).

    Args:
        novelty_lambda: Weight of the similarity penalty (0 = no pruning, 1 = full penalty).
        sim_mode:       "max" uses max similarity to any read sentence;
                        "mean" uses mean similarity across all read sentences.
        threshold:      If set, hard-exclude sentences above this similarity (0.0–1.0).
    """
    n_target = len(target_sentences)
    n_read   = len(read_sentences)

    # PageRank on target sentences only (not influenced by read graph size)
    print(f"  Building similarity matrix ({n_target} target sentences)...")
    target_sim = _cosine_similarity_matrix(target_embeddings)
    print("  Running PageRank...")
    pagerank_scores = _pagerank(target_sim)

    # Compute similarity of each target sentence to read corpus
    if n_read > 0:
        t_norms = np.linalg.norm(target_embeddings, axis=1, keepdims=True)
        t_norms = np.where(t_norms == 0, 1e-9, t_norms)
        r_norms = np.linalg.norm(read_embeddings, axis=1, keepdims=True)
        r_norms = np.where(r_norms == 0, 1e-9, r_norms)
        t_normed = target_embeddings / t_norms   # (n_target, d)
        r_normed = read_embeddings   / r_norms   # (n_read, d)
        sim_to_read = t_normed @ r_normed.T      # (n_target, n_read)

        if sim_mode == "max":
            read_penalty = sim_to_read.max(axis=1)   # (n_target,)
        else:
            read_penalty = sim_to_read.mean(axis=1)  # (n_target,)
    else:
        read_penalty = np.zeros(n_target)

    final_scores = pagerank_scores - novelty_lambda * read_penalty

    # Hard threshold: exclude sentences too similar to read content
    if threshold is not None and n_read > 0:
        mask = read_penalty >= threshold
        final_scores[mask] = -np.inf

    top_k = max(1, int(n_target * top_k_fraction))
    ranked_indices = np.argsort(final_scores)[::-1][:top_k].tolist()
    # Filter out hard-excluded sentences
    ranked_indices = [i for i in ranked_indices if final_scores[i] > -np.inf]

    results = []
    for rank_position, idx in enumerate(ranked_indices, start=1):
        results.append({
            "sentence":   target_sentences[idx],
            "label":      target_labels[idx],
            "confidence": round(float(target_confidences[idx]), 4),
            "rank":       rank_position,
            "score":      round(float(final_scores[idx]), 6),
            "read_sim":   round(float(read_penalty[idx]), 4),
        })

    return results


if __name__ == "__main__":
    from zotero_db import get_collection_id, get_items_with_pdfs, get_read_item_keys
    from extractor import extract_sentences
    from classifier import classify
    from embedder import embed
    from cache import get as cache_get, store as cache_store

    def process_item(item):
        """Extract, classify, and embed a paper — using cache if available."""
        sents = extract_sentences(item["pdf_path"])
        cached = cache_get(item["attachment_key"], sents)
        if cached:
            return sents, cached["labels"], cached["confidences"], cached["embeddings"]
        print(f"  Classifying {len(sents)} sentences...")
        classifications = classify(sents)
        labels = [c[0] for c in classifications]
        confs = [c[1] for c in classifications]
        print(f"  Embedding {len(sents)} sentences...")
        embs = embed(sents)
        cache_store(item["attachment_key"], sents, embs, labels, confs)
        return sents, labels, confs, embs

    col_id = get_collection_id("Class - Human-Centered AI")
    items = get_items_with_pdfs(col_id)
    read_keys = get_read_item_keys(col_id)

    # Pick first unread paper as target
    target_item = next(i for i in items if i["item_key"] not in read_keys)
    print(f"\nTarget paper: {target_item['pdf_path']}")
    target_sents, target_labels, target_confs, target_embs = process_item(target_item)
    print(f"  {len(target_sents)} sentences")

    # Collect read-paper sentences
    read_items = [i for i in items if i["item_key"] in read_keys]
    all_read_sents = []
    all_read_embs = []

    for item in read_items:
        print(f"Processing read paper: {item['item_key']}...")
        sents, _, _, embs = process_item(item)
        all_read_sents.extend(sents)
        all_read_embs.append(embs)

    read_embs = np.vstack(all_read_embs) if all_read_embs else np.empty((0, 768), dtype=np.float32)

    print("\nRanking...")
    results = rank(
        target_sents, target_labels, target_confs, target_embs,
        all_read_sents, read_embs,
    )


    print(f"\nTop {len(results)} sentences:")
    for r in results:
        print(f"  [{r['rank']}] [{r['label']} {r['confidence']}] {r['sentence'][:80]}")
