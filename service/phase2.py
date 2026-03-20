"""
Phase 2 — Rank sentences and write highlights.json.
Run this whenever read tags change.

Usage:
    python service/phase2.py "Class - Human-Centered AI"
"""
import sys
import json
sys.path.insert(0, ".")

import numpy as np
from pathlib import Path

from zotero_db import get_collection_id, get_items_with_pdfs, get_read_item_keys
from extractor import extract_sentences
from ranker import rank
from cache import get as cache_get

HIGHLIGHTS_PATH = Path(__file__).parent / "highlights.json"


def run(collection_name: str) -> None:
    col_id = get_collection_id(collection_name)
    if col_id is None:
        print(f"Collection not found: {collection_name}")
        sys.exit(1)

    items = get_items_with_pdfs(col_id)
    read_keys = get_read_item_keys(col_id)
    print(f"Found {len(items)} papers, {len(read_keys)} marked read.")

    # Load read-paper data from cache
    read_sents = []
    read_embs_list = []
    for item in items:
        if item["item_key"] not in read_keys:
            continue
        sents = extract_sentences(item["pdf_path"])
        if sents is None:
            continue
        cached = cache_get(item["attachment_key"], sents)
        if cached is None:
            print(f"  WARNING: {item['item_key']} not cached — run phase1 first. Skipping.")
            continue
        read_sents.extend(cached["sentences"])
        read_embs_list.append(cached["embeddings"])

    read_embs = np.vstack(read_embs_list) if read_embs_list else np.empty((0, 768), dtype=np.float32)

    # Load existing highlights and update this collection's section
    if HIGHLIGHTS_PATH.exists():
        with open(HIGHLIGHTS_PATH) as f:
            highlights = json.load(f)
    else:
        highlights = {}

    highlights[collection_name] = {}

    # Rank each unread paper
    unread_items = [i for i in items if i["item_key"] not in read_keys]
    print(f"Ranking {len(unread_items)} unread papers...")

    for idx, item in enumerate(unread_items, start=1):
        print(f"\n[{idx}/{len(unread_items)}] {item['item_key']}")
        sents = extract_sentences(item["pdf_path"])
        cached = cache_get(item["attachment_key"], sents)
        if cached is None:
            print("  WARNING: not cached — run phase1 first. Skipping.")
            continue

        results = rank(
            cached["sentences"],
            cached["labels"],
            cached["confidences"],
            cached["embeddings"],
            read_sents,
            read_embs,
        )
        highlights[collection_name][item["item_key"]] = results
        print(f"  {len(results)} highlights.")

    with open(HIGHLIGHTS_PATH, "w") as f:
        json.dump(highlights, f, indent=2)

    print(f"\nWrote highlights to {HIGHLIGHTS_PATH}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python service/phase2.py <collection_name>")
        sys.exit(1)
    run(sys.argv[1])
