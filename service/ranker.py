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
    NONE-labeled sentences are excluded before ranking.
    """
    # Filter out NONE-labeled sentences before ranking
    keep = [i for i, lbl in enumerate(target_labels) if lbl != 'NONE']
    if not keep:
        return []
    target_sentences  = [target_sentences[i]  for i in keep]
    target_labels     = [target_labels[i]      for i in keep]
    target_confidences = [target_confidences[i] for i in keep]
    target_embeddings = target_embeddings[keep]

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
) -> list[dict]:
    """
    Re-rank target sentences using read-history novelty, returning top-k by score.

    Scoring:
        final_score[i] = pr_target[i] - lambda * pr_combined_norm[i]

    where pr_target is PageRank over target sentences only, and pr_combined_norm
    is the normalized PageRank from the target+read graph (target slice, renormed).
    Sentences similar to already-read content are pulled down in the ranking.

    Args:
        novelty_lambda: Re-ranking strength. 0 = pure PageRank, higher = stronger
                        penalty for sentences similar to read content.
    NONE-labeled sentences are excluded before ranking.
    """
    # Filter out NONE-labeled sentences before ranking
    keep = [i for i, lbl in enumerate(target_labels) if lbl != 'NONE']
    if not keep:
        return []
    target_sentences   = [target_sentences[i]   for i in keep]
    target_labels      = [target_labels[i]       for i in keep]
    target_confidences = [target_confidences[i]  for i in keep]
    target_embeddings  = target_embeddings[keep]

    n_target = len(target_sentences)
    n_read   = len(read_sentences)

    # Step 1: PageRank on target sentences only
    print(f"  Building similarity matrix ({n_target} target sentences)...")
    target_sim = _cosine_similarity_matrix(target_embeddings)
    print("  Running PageRank (target only)...")
    pr_target = _pagerank(target_sim)

    top_k = max(1, int(n_target * top_k_fraction))

    # Step 2: PageRank on target + read sentences combined
    if n_read > 0:
        print(f"  Running PageRank (target + {n_read} read sentences)...")
        all_embeddings   = np.vstack([target_embeddings, read_embeddings])
        combined_sim     = _cosine_similarity_matrix(all_embeddings)
        pr_combined_raw  = _pagerank(combined_sim)[:n_target]
        pr_combined_norm = pr_combined_raw / (pr_combined_raw.sum() + 1e-9)

        final_scores    = pr_target - novelty_lambda * pr_combined_norm
        ranked_indices  = np.argsort(final_scores)[::-1][:top_k].tolist()
    else:
        # No read sentences — pure PageRank, exactly k results
        final_scores   = pr_target
        ranked_indices = np.argsort(final_scores)[::-1][:top_k].tolist()

    results = []
    for rank_position, idx in enumerate(ranked_indices, start=1):
        results.append({
            "sentence":   target_sentences[idx],
            "label":      target_labels[idx],
            "confidence": round(float(target_confidences[idx]), 4),
            "rank":       rank_position,
            "score":      round(float(final_scores[idx]), 6),
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
