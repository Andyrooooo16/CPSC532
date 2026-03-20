"""
Phase 1 — Extract, classify, and embed all papers in a collection.
Run this once (or when new papers are added) to populate the cache.

Usage:
    python service/phase1.py "Class - Human-Centered AI"
"""
import sys
sys.path.insert(0, ".")

from zotero_db import get_collection_id, get_items_with_pdfs
from extractor import extract_sentences
from classifier import classify
from embedder import embed
from cache import get as cache_get, store as cache_store


def run(collection_name: str) -> None:
    col_id = get_collection_id(collection_name)
    if col_id is None:
        print(f"Collection not found: {collection_name}")
        sys.exit(1)

    items = get_items_with_pdfs(col_id)
    print(f"Found {len(items)} papers in '{collection_name}'")

    for i, item in enumerate(items, start=1):
        key = item["attachment_key"]
        print(f"\n[{i}/{len(items)}] {item['item_key']} — {item['pdf_path']}")

        sents = extract_sentences(item["pdf_path"])
        if sents is None:
            continue
        cached = cache_get(key, sents)
        if cached:
            print(f"  Already cached ({len(sents)} sentences), skipping.")
            continue

        print(f"  Classifying {len(sents)} sentences...")
        classifications = classify(sents)
        labels = [c[0] for c in classifications]
        confs = [c[1] for c in classifications]

        print(f"  Embedding {len(sents)} sentences...")
        embs = embed(sents)

        cache_store(key, sents, embs, labels, confs)
        print("  Cached.")

    print(f"\nPhase 1 complete for '{collection_name}'.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python service/phase1.py <collection_name>")
        sys.exit(1)
    run(sys.argv[1])
