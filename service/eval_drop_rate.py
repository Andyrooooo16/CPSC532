"""
eval_drop_rate.py — Measure how many top-k sentences would be pruned by a
cosine threshold (τ) on topically-similar papers from Zotero, as context
papers accumulate.

No ground-truth labels needed — this is purely about drop percentage.

Setup:
  - Pulls PDFs from topically-relevant Zotero collections (Human-Centered AI,
    Class-Readings, and Course Project sub-collections), filtered to 10–50 pages.
  - For each paper as "target", samples the others as "read" context.
  - Runs rank_novel (λ=0.6) for each k, then measures what fraction of the
    top-k results would be dropped at each cosine threshold τ.

Usage:
    cd "Project Code"
    python service/eval_drop_rate.py
    python service/eval_drop_rate.py --trials 5 --seed 42
"""

import sys, sqlite3, random, json, argparse
import numpy as np
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, ".")

import fitz
import nltk

from extractor import extract_sentences
from embedder import embed
from classifier import classify
from ranker import rank_novel
from cache import get as cache_get, store as cache_store

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ZOTERO_DB      = Path.home() / "Zotero" / "zotero.sqlite"
ZOTERO_STORAGE = Path.home() / "Zotero" / "storage"
MIN_PAGES = 10
MAX_PAGES = 50

# Topically-relevant collection IDs
RELEVANT_COLLECTION_IDS = [59, 61, 63, 72, 73, 74, 84]

READ_COUNTS = [1, 2, 3, 5, 10]
K_VALUES    = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
LAMBDAS     = [0.6]
TAU_VALUES  = [0.88, 0.90, 0.92, 0.94, 0.96, 0.98, 1.0]


# ---------------------------------------------------------------------------
# Find candidate PDFs via Zotero DB
# ---------------------------------------------------------------------------

def find_candidate_pdfs():
    """Return unique (n_pages, path) pairs from the relevant collections."""
    conn = sqlite3.connect(f"file:{ZOTERO_DB}?immutable=1", uri=True)
    conn.row_factory = sqlite3.Row

    seen = {}
    for col_id in RELEVANT_COLLECTION_IDS:
        rows = conn.execute("""
            SELECT att.key AS attachment_key, ia.path AS raw_path
            FROM collectionItems ci
            JOIN items parent ON ci.itemID = parent.itemID
            JOIN itemAttachments ia ON ia.parentItemID = parent.itemID
            JOIN items att ON att.itemID = ia.itemID
            WHERE ci.collectionID = ? AND ia.contentType = 'application/pdf'
        """, (col_id,)).fetchall()

        for r in rows:
            fname = r["raw_path"].removeprefix("storage:")
            path  = ZOTERO_STORAGE / r["attachment_key"] / fname
            if not path.exists() or str(path) in seen:
                continue
            try:
                doc = fitz.open(str(path))
                n   = len(doc)
                doc.close()
                if MIN_PAGES <= n <= MAX_PAGES:
                    seen[str(path)] = n
            except Exception:
                pass

    conn.close()
    candidates = [(n, p) for p, n in seen.items()]
    candidates.sort(key=lambda x: x[0])
    return candidates


# ---------------------------------------------------------------------------
# Process a single PDF: extract → classify → embed (with cache)
# ---------------------------------------------------------------------------

def process_pdf(pdf_path):
    cache_key = Path(pdf_path).stem
    sentences = extract_sentences(pdf_path)
    if sentences is None or len(sentences) < 5:
        return None

    cached = cache_get(cache_key, sentences)
    if cached is not None:
        return {
            "path":      pdf_path,
            "sentences": cached["sentences"],
            "labels":    cached["labels"],
            "confs":     cached["confidences"],
            "embs":      cached["embeddings"],
        }

    embs   = embed(sentences)
    pairs  = classify(sentences)
    labels = [p[0] for p in pairs]
    confs  = [p[1] for p in pairs]
    cache_store(cache_key, sentences, embs, labels, confs)
    return {
        "path":      pdf_path,
        "sentences": sentences,
        "labels":    labels,
        "confs":     confs,
        "embs":      embs,
    }


# ---------------------------------------------------------------------------
# Drop-rate helpers
# ---------------------------------------------------------------------------

def _max_sim_to_read(target_embs, read_embs):
    """For each target sentence, return max cosine similarity to any read sentence."""
    t = target_embs / np.where(
        (n := np.linalg.norm(target_embs, axis=1, keepdims=True)) == 0, 1e-9, n)
    r = read_embs / np.where(
        (n := np.linalg.norm(read_embs, axis=1, keepdims=True)) == 0, 1e-9, n)
    return (t @ r.T).max(axis=1)


def _drop_frac(ranked_sims, tau):
    if tau >= 1.0:
        return 0.0
    return float((ranked_sims >= tau).sum()) / len(ranked_sims)


def compute_drop_rates(target, read_sents, read_embs):
    """Return {(k, lam, tau): drop_fraction} for all config combinations."""
    dim   = target["embs"].shape[1]
    empty = np.empty((0, dim), dtype=np.float32)

    if read_sents:
        max_sim = _max_sim_to_read(target["embs"], read_embs)
    else:
        max_sim = np.zeros(len(target["sentences"]))

    sent_to_idx = {s: i for i, s in enumerate(target["sentences"])}
    results = {}

    for lam in LAMBDAS:
        for k in K_VALUES:
            ranked = rank_novel(
                target["sentences"], target["labels"], target["confs"], target["embs"],
                read_sents,
                read_embs if read_sents else empty,
                top_k_fraction=k,
                novelty_lambda=lam,
            )
            if not ranked:
                for tau in TAU_VALUES:
                    results[(k, lam, tau)] = 0.0
                continue

            ranked_sims = np.array([
                max_sim[sent_to_idx[r["sentence"]]]
                for r in ranked if r["sentence"] in sent_to_idx
            ])
            for tau in TAU_VALUES:
                results[(k, lam, tau)] = _drop_frac(ranked_sims, tau)

    return results


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def run_simulation(papers, n_trials):
    """Return agg[rc][(k, lam, tau)] -> list of drop fractions."""
    agg = defaultdict(lambda: defaultdict(list))

    for target_idx, target in enumerate(papers):
        print(f"Target [{target_idx+1}/{len(papers)}]: {Path(target['path']).name[:60]}")
        others = [i for i in range(len(papers)) if i != target_idx]

        for rc in READ_COUNTS:
            if rc > len(others):
                print(f"  Skipping rc={rc} (not enough other papers).")
                continue

            for _ in range(n_trials):
                sampled = random.sample(others, rc)
                read_sents = [s for i in sampled for s in papers[i]["sentences"]]
                read_embs  = np.vstack([papers[i]["embs"] for i in sampled])

                for key, frac in compute_drop_rates(target, read_sents, read_embs).items():
                    agg[rc][key].append(frac)

            # Progress hint: k=10%, τ=0.7
            sample_key = (0.10, 0.6, 0.7)
            vals = agg[rc].get(sample_key, [])
            if vals:
                print(f"  rc={rc}: k=10% λ=0.6 τ=0.7 → drop {np.mean(vals):.1%}")

    return agg


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_results(agg, n_papers, n_trials):
    print("\n" + "=" * 80)
    print(f"Drop-rate — Zotero {MIN_PAGES}–{MAX_PAGES} page HCI papers "
          f"({n_papers} papers, {n_trials} trials/rc)")
    print("=" * 80)

    for lam in LAMBDAS:
        for k in K_VALUES:
            print(f"\nλ={lam}  k={k:.0%}")
            header = f"  {'read':>6}  " + "  ".join(f"τ={t:.1f}" for t in TAU_VALUES)
            print(header)
            print("  " + "-" * len(header))
            for rc in READ_COUNTS:
                row = f"  {rc:>6}  "
                for tau in TAU_VALUES:
                    vals = agg[rc].get((k, lam, tau), [])
                    mean = np.mean(vals) if vals else float("nan")
                    row += f"  {mean:>6.1%}"
                print(row)


def save_results(agg, n_papers, n_trials, fname="drop_rate_results.json"):
    out = {
        "meta": {
            "n_papers": n_papers,
            "trials": n_trials,
            "read_counts": READ_COUNTS,
            "k_values": K_VALUES,
            "lambdas": LAMBDAS,
            "tau_values": TAU_VALUES,
            "min_pages": MIN_PAGES,
            "max_pages": MAX_PAGES,
        },
        "results": {},
    }
    for rc in READ_COUNTS:
        out["results"][str(rc)] = {}
        for lam in LAMBDAS:
            for k in K_VALUES:
                for tau in TAU_VALUES:
                    key_str = f"L{lam}_k{k:.2f}_T{tau:.1f}"
                    vals = agg[rc].get((k, lam, tau), [])
                    out["results"][str(rc)][key_str] = {
                        "mean_drop": float(np.mean(vals)) if vals else None,
                        "std_drop":  float(np.std(vals))  if vals else None,
                        "n":         len(vals),
                    }
    with open(fname, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {fname}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=5,
                        help="Trials per read-count (default 5)")
    parser.add_argument("--seed",   type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)

    print(f"Finding {MIN_PAGES}–{MAX_PAGES} page PDFs in relevant Zotero collections…")
    candidates = find_candidate_pdfs()
    print(f"Found {len(candidates)} candidate PDFs.\n")

    papers = []
    for i, (n_pages, path) in enumerate(candidates, 1):
        print(f"[{i}/{len(candidates)}] {Path(path).name[:60]} ({n_pages} pages)")
        data = process_pdf(path)
        if data is None:
            print("  Skipped.")
            continue
        print(f"  {len(data['sentences'])} sentences")
        papers.append(data)

    if len(papers) < 3:
        print("Not enough papers after processing. Exiting.")
        sys.exit(1)

    print(f"\n{len(papers)} papers ready.\n")

    agg = run_simulation(papers, args.trials)
    print_results(agg, len(papers), args.trials)
    save_results(agg, len(papers), args.trials)


if __name__ == "__main__":
    main()
