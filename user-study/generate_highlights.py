"""
generate_highlights.py — Pre-study highlight generation script.

Reads study-config.json to enumerate every (paper, condition, prior-papers)
combination actually needed, then generates a highlight JSON for each one.

Output files in data/highlights/:
  <stem>_all.json               — all_highlights (no prior context)
  <stem>_ctx_.json              — contextual_highlights, first paper (no prior)
  <stem>_ctx_<stem2>_...json   — contextual_highlights with prior papers

Usage:
    cd "Project Code"
    python user-study/generate_highlights.py

Highlight file format:
    [
      { "page": 1, "rects": [[x0, y0, x1, y1], ...], "text": "...", "label": "RESULTS" },
      ...
    ]
    Page numbers are 1-indexed. Rects in PyMuPDF coordinate space (top-left origin,
    y increases downward) — matches how viewer.js draws them.
"""

import json
import sys
import numpy as np
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "service"))

import fitz

from extractor import extract_sentences
from embedder import embed
from classifier import classify
from ranker import rank, rank_novel
from cache import get as cache_get, store as cache_store

STUDY_DIR = Path(__file__).parent
CONFIG_PATH = STUDY_DIR / "study-config.json"
PAPERS_DIR = STUDY_DIR / "public" / "papers"
HIGHLIGHTS_DIR = STUDY_DIR / "data" / "highlights"
HIGHLIGHTS_DIR.mkdir(parents=True, exist_ok=True)

# Cap highlights per paper. None = use all ranked sentences (~20%).
MAX_HIGHLIGHTS = None

# ---------------------------------------------------------------------------
# Load config
# ---------------------------------------------------------------------------

with open(CONFIG_PATH) as f:
    config = json.load(f)

papers_cfg = config["papers"]  # { paperKey: { title, filename } }


def stem_for(paper_key):
    filename = papers_cfg[paper_key]["filename"]
    return Path(filename).stem


# ---------------------------------------------------------------------------
# Enumerate all needed (paper_key, condition, prior_stems_tuple) combos
# ---------------------------------------------------------------------------

def compute_needed_combos():
    combos = set()
    all_configs = list(config["participants"].values()) + [config["defaultParticipant"]]

    for pcfg in all_configs:
        order = pcfg["paperOrder"]
        conditions = pcfg["conditions"]
        ctx_prior = []  # stems of contextual_highlights papers seen so far

        for paper_key, cond in zip(order, conditions):
            if cond == "all_highlights":
                combos.add((paper_key, "all_highlights", ()))
            elif cond == "contextual_highlights":
                combos.add((paper_key, "contextual_highlights", tuple(sorted(ctx_prior))))
                ctx_prior.append(stem_for(paper_key))
            # no_highlights: no file needed

    return combos


def highlight_filename(paper_key, condition, prior_stems):
    stem = stem_for(paper_key)
    if condition == "all_highlights":
        return f"{stem}_all"
    if prior_stems:
        return f"{stem}_ctx_{'_'.join(sorted(prior_stems))}"
    return f"{stem}_ctx_"


# ---------------------------------------------------------------------------
# Sentence extraction + embeddings (with cache)
# ---------------------------------------------------------------------------

def get_paper_data(paper_key):
    pdf_path = PAPERS_DIR / papers_cfg[paper_key]["filename"]
    cache_key = pdf_path.stem

    print(f"  Extracting sentences from {pdf_path.name}…")
    sentences = extract_sentences(str(pdf_path))
    if sentences is None:
        raise RuntimeError(f"PDF too long (>40 pages): {pdf_path.name}")

    cached = cache_get(cache_key, sentences)
    if cached is not None:
        print(f"  Loaded {len(cached['embeddings'])} embeddings from cache.")
        return cached["sentences"], cached["embeddings"], cached["labels"], cached["confidences"]

    print(f"  Embedding {len(sentences)} sentences…")
    embeddings = embed(sentences)
    print(f"  Classifying…")
    pairs = classify(sentences)
    labels = [p[0] for p in pairs]
    confidences = [p[1] for p in pairs]
    cache_store(cache_key, sentences, embeddings, labels, confidences)
    return sentences, embeddings, labels, confidences


# ---------------------------------------------------------------------------
# Bounding box search
# ---------------------------------------------------------------------------

def locate_highlights(pdf_path, ranked):
    doc = fitz.open(str(pdf_path))
    results = []

    for item in ranked:
        sent = item["sentence"]
        label = item.get("label", "NONE")
        for page_num in range(1, len(doc) + 1):
            page = doc[page_num - 1]
            rects = page.search_for(sent)
            if not rects and len(sent) > 80:
                rects = page.search_for(sent[:80])
            if rects:
                results.append({
                    "page": page_num,
                    "rects": [[r.x0, r.y0, r.x1, r.y1] for r in rects],
                    "text": sent,
                    "label": label,
                })
                break
        else:
            print(f"    WARNING: not found on any page: {sent[:60]}…")

    doc.close()
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    combos = compute_needed_combos()
    print(f"Need to generate {len(combos)} highlight file(s).\n")

    # Pre-load all paper data needed across all combos
    needed_keys = {paper_key for paper_key, _, _ in combos}
    paper_data = {}
    for paper_key in needed_keys:
        print(f"[{paper_key}] Loading data…")
        try:
            sents, embs, labels, confs = get_paper_data(paper_key)
            paper_data[paper_key] = dict(sentences=sents, embeddings=embs, labels=labels, confidences=confs)
        except Exception as e:
            print(f"  ERROR: {e}")
            sys.exit(1)
    print()

    # Also need data for prior papers (to build read corpus)
    # Prior papers are identified by stem — find their paper_key
    stem_to_key = {stem_for(k): k for k in papers_cfg}

    for i, (paper_key, condition, prior_stems) in enumerate(sorted(combos), 1):
        fname = highlight_filename(paper_key, condition, prior_stems)
        out_path = HIGHLIGHTS_DIR / f"{fname}.json"

        if out_path.exists():
            print(f"[{i}/{len(combos)}] {fname}.json already exists — skipping.")
            continue

        print(f"[{i}/{len(combos)}] Generating {fname}.json…")

        data = paper_data[paper_key]
        pdf_path = PAPERS_DIR / papers_cfg[paper_key]["filename"]

        # Build read corpus from prior contextual papers
        read_sents = []
        read_embs_list = []
        for prior_stem in prior_stems:
            prior_key = stem_to_key.get(prior_stem)
            if prior_key and prior_key in paper_data:
                read_sents.extend(paper_data[prior_key]["sentences"])
                read_embs_list.append(paper_data[prior_key]["embeddings"])
            else:
                print(f"  WARNING: prior paper '{prior_stem}' not found in paper data, skipping.")

        read_embs = np.vstack(read_embs_list) if read_embs_list else np.empty((0, 768), dtype=np.float32)

        if condition == "contextual_highlights":
            ranked = rank_novel(
                data["sentences"], data["labels"], data["confidences"], data["embeddings"],
                read_sents, read_embs,
                novelty_lambda=0.6,
                top_k_fraction=0.30,
            )
        else:
            ranked = rank(data["sentences"], data["labels"], data["confidences"], data["embeddings"], read_sents, read_embs)
        if MAX_HIGHLIGHTS is not None:
            ranked = ranked[:MAX_HIGHLIGHTS]

        print(f"  {len(ranked)} sentences selected. Locating bounding boxes…")
        highlights = locate_highlights(pdf_path, ranked)
        print(f"  {len(highlights)} sentences located.")

        with open(out_path, "w") as f:
            json.dump(highlights, f, indent=2)
        print(f"  Wrote {out_path.name}\n")

    print("Done.")


if __name__ == "__main__":
    main()
