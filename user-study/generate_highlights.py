"""
generate_highlights.py — Pre-study highlight generation script.

Generates two highlight files for every PDF in public/papers/:
  <stem>_all.json          — all_highlights condition (no prior context)
  <stem>_ctx_.json         — contextual_highlights condition (no prior papers)

Run this once before the study begins.

Usage:
    cd "Project Code"
    python user-study/generate_highlights.py

Highlight file format:
    [
      { "page": 1, "rects": [[x0, y0, x1, y1], ...], "text": "Sentence..." },
      ...
    ]
    Page numbers are 1-indexed. Rects are in PDF coordinate space (origin
    bottom-left), suitable for viewport.convertToViewportRectangle() in PDF.js.
"""

import json
import sys
import numpy as np
from pathlib import Path

# Add service/ to the path
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "service"))

import nltk
import fitz  # PyMuPDF

from extractor import extract_sentences
from embedder import embed
from classifier import classify
from ranker import rank
from cache import get as cache_get, store as cache_store

PAPERS_DIR = Path(__file__).parent / "public" / "papers"
HIGHLIGHTS_DIR = Path(__file__).parent / "data" / "highlights"
HIGHLIGHTS_DIR.mkdir(parents=True, exist_ok=True)

# Optionally cap the number of highlights per paper. None = use all ranked sentences (~20%).
MAX_HIGHLIGHTS = None


# ---------------------------------------------------------------------------
# Sentence extraction + embedding/classification (with cache)
# ---------------------------------------------------------------------------

def get_paper_data(pdf_path: Path):
    """Return (sentences, embeddings, labels, confidences) for a PDF, using cache."""
    cache_key = pdf_path.stem

    print(f"  Extracting sentences…")
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
# Ranking
# ---------------------------------------------------------------------------

def get_highlights(sentences, labels, confidences, embeddings, prior_sentences=None, prior_embeddings=None):
    """Rank sentences and return the top ones."""
    read_sents = prior_sentences or []
    read_embs = prior_embeddings if prior_embeddings is not None else np.empty((0, 768), dtype=np.float32)

    ranked = rank(sentences, labels, confidences, embeddings, read_sents, read_embs)
    if MAX_HIGHLIGHTS is not None:
        ranked = ranked[:MAX_HIGHLIGHTS]
    return ranked  # list of { sentence, label, score, ... }


# ---------------------------------------------------------------------------
# Bounding box search
# ---------------------------------------------------------------------------

def locate_highlights(pdf_path: Path, ranked: list):
    """
    Search each ranked sentence across all pages. Returns list of
    { page, rects, text, label } dicts for sentences that were found.
    ranked is a list of { sentence, label, ... } dicts from the ranker.
    """
    doc = fitz.open(str(pdf_path))
    results = []

    for item in ranked:
        sent = item["sentence"]
        label = item.get("label", "NONE")
        found = False
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
                found = True
                break
        if not found:
            print(f"    WARNING: not found on any page: {sent[:60]}…")

    doc.close()
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)

    pdfs = sorted(PAPERS_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {PAPERS_DIR}. Add your study PDFs and re-run.")
        sys.exit(1)

    print(f"Found {len(pdfs)} PDF(s): {', '.join(p.name for p in pdfs)}\n")

    for pdf_path in pdfs:
        stem = pdf_path.stem
        print(f"=== {pdf_path.name} ===")

        try:
            sentences, embeddings, labels, confidences = get_paper_data(pdf_path)
        except Exception as e:
            print(f"  ERROR: {e}\n")
            continue

        empty_embs = np.empty((0, 768), dtype=np.float32)

        for suffix, prior_sents, prior_embs in [
            ("_all",  [],  empty_embs),
            ("_ctx_", [],  empty_embs),
        ]:
            out_path = HIGHLIGHTS_DIR / f"{stem}{suffix}.json"
            if out_path.exists():
                print(f"  {out_path.name} already exists — skipping.")
                continue

            print(f"  Ranking for {out_path.name}…")
            ranked = get_highlights(sentences, labels, confidences, embeddings, prior_sents, prior_embs)
            print(f"  {len(ranked)} sentences selected. Locating bounding boxes…")

            highlights = locate_highlights(pdf_path, ranked)
            print(f"  {len(highlights)} sentences located.")

            with open(out_path, "w") as f:
                json.dump(highlights, f, indent=2)
            print(f"  Wrote {out_path.name}")

        print()

    print("Done.")


if __name__ == "__main__":
    main()
