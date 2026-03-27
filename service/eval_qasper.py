"""
eval_qasper.py — Evaluate highlight coverage on the QASPER dataset.

For each paper in the QASPER validation split:
  1. Flatten full_text paragraphs + figure/table captions into sentences via nltk.
  2. Run classify → embed → rank (top 20% of paper's own sentences, no competing papers).
  3. For each highlighted_evidence sentence, check if it appears as a substring of any
     ranked sentence (or a ranked sentence appears within it).
  4. Compare against a random-20% baseline via Monte Carlo, then run a Wilcoxon
     signed-rank test to assess statistical significance.

Usage:
    python service/eval_qasper.py
    python service/eval_qasper.py --max-papers 20
    python service/eval_qasper.py --split test
    python service/eval_qasper.py --f1-threshold 0.3 --random-trials 200
"""
import sys
import random
import argparse
import numpy as np
import nltk

nltk.download("punkt_tab", quiet=True)

sys.path.insert(0, ".")

from classifier import classify
from embedder import embed
from ranker import rank


def flatten_paper_text(paper) -> list[str]:
    """Flatten full_text paragraphs and figure/table captions into a deduplicated sentence list."""
    sentences = []

    for para_list in paper["full_text"]["paragraphs"]:
        for para in para_list:
            para = para.strip()
            if not para:
                continue
            for sent in nltk.sent_tokenize(para):
                sentences.append(sent)

    for caption in paper["figures_and_tables"]["caption"]:
        caption = caption.strip()
        if caption:
            sentences.append(caption)

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for s in sentences:
        if s not in seen:
            seen.add(s)
            unique.append(s)
    return unique


def _token_f1(a: str, b: str) -> float:
    """Token-level F1 between two strings (standard QA overlap metric)."""
    a_tokens = a.lower().split()
    b_tokens = b.lower().split()
    if not a_tokens or not b_tokens:
        return 0.0
    overlap = len(set(a_tokens) & set(b_tokens))
    if overlap == 0:
        return 0.0
    precision = overlap / len(b_tokens)
    recall = overlap / len(a_tokens)
    return 2 * precision * recall / (precision + recall)


def best_match_f1(sent: str, highlighted: list[str]) -> float:
    """Return the highest token F1 score between sent and any highlighted sentence.
    Substring matches are treated as F1 = 1.0."""
    s = sent.lower().strip()
    if not s:
        return 0.0
    best = 0.0
    for h in highlighted:
        h = h.lower().strip()
        if s in h or h in s:
            return 1.0
        best = max(best, _token_f1(s, h))
    return best


def sentence_covered(sent: str, highlighted: list[str], f1_threshold: float = 0.5) -> bool:
    """True if sent matches any highlighted sentence via substring or token F1 >= threshold."""
    return best_match_f1(sent, highlighted) >= f1_threshold


def evidence_piece_sentences(piece: str) -> list[str]:
    """Split an evidence piece into sentences.
    FLOAT SELECTED entries are kept as a single unit since they reference a caption."""
    if piece.startswith("FLOAT SELECTED:"):
        return [piece]
    return [s for s in nltk.sent_tokenize(piece) if s.strip()]


def compute_coverage(qas, highlighted: list[str], f1_threshold: float) -> dict:
    """Compute the four coverage metrics given a set of highlighted sentences."""
    all_sent_coverages = []
    all_best_f1s = []
    all_piece_coverages = []
    all_answer_coverages = []
    question_coverages = []

    for answer_group in qas["answers"]:
        annotations_covered = []

        for annotation in answer_group["answer"]:
            if annotation["unanswerable"]:
                continue

            piece_covered_flags = []
            for piece in annotation["highlighted_evidence"]:
                piece = piece.strip()
                if not piece:
                    continue
                sents = evidence_piece_sentences(piece)
                sent_flags = [sentence_covered(s, highlighted, f1_threshold) for s in sents]
                sent_f1s   = [best_match_f1(s, highlighted) for s in sents]
                sent_cov   = sum(sent_flags) / len(sent_flags)
                all_sent_coverages.extend(sent_flags)
                all_best_f1s.extend(sent_f1s)
                all_piece_coverages.append(sent_cov)
                piece_covered_flags.append(sent_cov > 0)

            if piece_covered_flags:
                ann_piece_cov = sum(piece_covered_flags) / len(piece_covered_flags)
                all_answer_coverages.append(ann_piece_cov)
                annotations_covered.append(ann_piece_cov > 0)

        if annotations_covered:
            question_coverages.append(sum(annotations_covered) / len(annotations_covered))

    if not question_coverages:
        return None

    return {
        "best_f1s":          all_best_f1s,
        "sentence_coverage": sum(all_sent_coverages) / len(all_sent_coverages),
        "piece_coverage":    sum(1 for c in all_piece_coverages if c > 0) / len(all_piece_coverages),
        "answer_coverage":   sum(all_answer_coverages) / len(all_answer_coverages),
        "question_coverage": sum(question_coverages) / len(question_coverages),
    }


def evaluate_paper(paper, paper_idx: int, total: int,
                   f1_threshold: float = 0.5,
                   n_random_trials: int = 100) -> dict | None:
    title = paper.get("title", paper.get("id", "?"))
    print(f"[{paper_idx}/{total}] {title[:70]}")

    sentences = flatten_paper_text(paper)
    if len(sentences) < 2:
        print("  Skipped — too few sentences\n")
        return None

    has_evidence = any(
        not annotation["unanswerable"] and annotation["highlighted_evidence"]
        for answer_group in paper["qas"]["answers"]
        for annotation in answer_group["answer"]
    )
    if not has_evidence:
        print("  Skipped — no highlighted evidence\n")
        return None

    print(f"  {len(sentences)} sentences extracted")

    classifications = classify(sentences)
    labels = [c[0] for c in classifications]
    confs  = [c[1] for c in classifications]
    embeddings = embed(sentences)

    empty_embs = np.empty((0, embeddings.shape[1]), dtype=np.float32)
    results = rank(
        sentences, labels, confs, embeddings,
        read_sentences=[],
        read_embeddings=empty_embs,
        top_k_fraction=0.20,
        global_top_k=False,
    )
    highlighted = [r["sentence"] for r in results]
    k = len(highlighted)

    actual = compute_coverage(paper["qas"], highlighted, f1_threshold)
    if actual is None:
        print("  Skipped — no scorable questions\n")
        return None

    # Monte Carlo random baseline: sample k sentences uniformly, repeat n_random_trials times
    random_coverages = {
        "sentence_coverage": [],
        "piece_coverage":    [],
        "answer_coverage":   [],
        "question_coverage": [],
    }
    for _ in range(n_random_trials):
        rand_highlighted = random.sample(sentences, k)
        rand = compute_coverage(paper["qas"], rand_highlighted, f1_threshold)
        if rand is None:
            continue
        for key in random_coverages:
            random_coverages[key].append(rand[key])

    rand_means = {
        key: sum(vals) / len(vals) if vals else 0.0
        for key, vals in random_coverages.items()
    }

    print(
        f"  Actual  — Sentence: {actual['sentence_coverage']:.1%}  Piece: {actual['piece_coverage']:.1%}  "
        f"Answer: {actual['answer_coverage']:.1%}  Question: {actual['question_coverage']:.1%}\n"
        f"  Random  — Sentence: {rand_means['sentence_coverage']:.1%}  Piece: {rand_means['piece_coverage']:.1%}  "
        f"Answer: {rand_means['answer_coverage']:.1%}  Question: {rand_means['question_coverage']:.1%}\n"
    )

    return {
        "n_sentences":    len(sentences),
        "n_highlighted":  k,
        "best_f1s":       actual["best_f1s"],
        **{key: actual[key] for key in random_coverages},
        **{f"rand_{key}": rand_means[key] for key in random_coverages},
        "rand_question_coverage_dist": random_coverages["question_coverage"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-papers", type=int, default=None)
    parser.add_argument("--split", default="validation",
                        choices=["train", "validation", "test"])
    parser.add_argument("--f1-threshold", type=float, default=0.5,
                        help="Token F1 threshold for fuzzy sentence matching (default: 0.5)")
    parser.add_argument("--random-trials", type=int, default=100,
                        help="Monte Carlo trials per paper for random baseline (default: 100)")
    args = parser.parse_args()

    print("Loading QASPER dataset...")
    from datasets import load_dataset
    ds = load_dataset("allenai/qasper", split=args.split)

    if args.max_papers:
        ds = ds.select(range(min(args.max_papers, len(ds))))

    total = len(ds)
    print(f"Evaluating {total} papers from '{args.split}' split...\n")

    results = []
    skipped = 0

    for i, paper in enumerate(ds, start=1):
        result = evaluate_paper(paper, i, total,
                                f1_threshold=args.f1_threshold,
                                n_random_trials=args.random_trials)
        if result is None:
            skipped += 1
        else:
            results.append(result)

    if not results:
        print("No papers were evaluated.")
        return

    def mean(key):
        return sum(r[key] for r in results) / len(results)

    def median(key):
        vals = sorted(r[key] for r in results)
        return vals[len(vals) // 2]

    print("=" * 65)
    print(f"Split            : {args.split}")
    print(f"F1 threshold     : {args.f1_threshold}")
    print(f"Random trials    : {args.random_trials}")
    print(f"Papers evaluated : {len(results)}")
    print(f"Papers skipped   : {skipped}")
    print()
    print(f"{'Metric':<35} {'Actual':>7}  {'Random':>7}  {'Delta':>7}")
    print("-" * 60)
    for label, key in [
        ("Sentence cov. (within piece)",   "sentence_coverage"),
        ("Piece cov.    (within answer)",   "piece_coverage"),
        ("Answer cov.   (within question)", "answer_coverage"),
        ("Question cov. (per paper)",       "question_coverage"),
    ]:
        actual_mean = mean(key)
        random_mean = mean(f"rand_{key}")
        delta = actual_mean - random_mean
        print(f"{label:<35} {actual_mean:>7.1%}  {random_mean:>7.1%}  {delta:>+7.1%}")

    # Wilcoxon signed-rank test on question coverage (most meaningful metric)
    # Compares per-paper actual vs. per-paper random mean
    from scipy.stats import wilcoxon, ttest_rel
    actual_q = [r["question_coverage"] for r in results]
    random_q = [r["rand_question_coverage"] for r in results]
    diffs = [a - b for a, b in zip(actual_q, random_q)]
    n_better = sum(1 for d in diffs if d > 0)

    print()
    print(f"Statistical significance (question coverage, n={len(results)} papers)")
    print(f"  Papers where actual > random baseline: {n_better}/{len(results)}")

    if len(results) >= 5 and len(set(diffs)) > 1:
        _, p_wilcoxon = wilcoxon(actual_q, random_q, alternative="greater")
        _, p_ttest    = ttest_rel(actual_q, random_q, alternative="greater")
        print(f"  Wilcoxon signed-rank p-value : {p_wilcoxon:.4f}")
        print(f"  Paired t-test p-value        : {p_ttest:.4f}")
    else:
        print("  (Too few papers for a reliable significance test)")

    # Best-match F1 distribution
    all_f1s = sorted(f1 for r in results for f1 in r["best_f1s"])
    n = len(all_f1s)
    f1_mean   = sum(all_f1s) / n
    f1_median = all_f1s[n // 2]
    buckets       = [0.0, 0.1, 0.25, 0.5, 0.75, 1.01]
    bucket_labels = ["0", "0.1", "0.25", "0.5", "0.75", "1.0"]
    counts = [
        sum(1 for f in all_f1s if buckets[i] <= f < buckets[i+1])
        for i in range(len(buckets) - 1)
    ]
    print()
    print(f"Best-match F1 distribution  (n={n} gold sentences)")
    print(f"  Mean: {f1_mean:.3f}   Median: {f1_median:.3f}")
    print(f"  {'Range':<12} {'Count':>6}  {'%':>6}")
    for i, cnt in enumerate(counts):
        lo = bucket_labels[i]
        hi = bucket_labels[i + 1]
        print(f"  [{lo:>4} – {hi:<4})  {cnt:>6}  {cnt/n:>6.1%}")


if __name__ == "__main__":
    main()
