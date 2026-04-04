import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = Path(__file__).parent / "scibert-acl-v3-final"
BATCH_SIZE = 32


def _load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()

    with open(MODEL_DIR / "label_map.json") as f:
        label_map = json.load(f)
    id2label = {int(k): v for k, v in label_map["id2label"].items()}

    return tokenizer, model, id2label


# Load once at import time
_tokenizer, _model, _id2label = _load_model()


def classify(sentences: list[str]) -> list[tuple[str, float]]:
    """
    Classify each sentence into a rhetorical category.

    Returns a list of (label, confidence) tuples in the same order as input.
    Labels: BACKGROUND, CONCLUSIONS, METHODS, NONE, OBJECTIVE, RESULTS
    """
    results = []

    for i in range(0, len(sentences), BATCH_SIZE):
        batch = sentences[i : i + BATCH_SIZE]

        inputs = _tokenizer(
            batch,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt",
        )

        with torch.no_grad():
            logits = _model(**inputs).logits

        probs = torch.softmax(logits, dim=-1)
        top_probs, top_ids = probs.max(dim=-1)

        for label_id, confidence in zip(top_ids.tolist(), top_probs.tolist()):
            results.append((_id2label[label_id], confidence))

    return results


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from extractor import extract_sentences
    from zotero_db import get_collection_id, get_items_with_pdfs
    import nltk
    nltk.download("punkt_tab", quiet=True)

    col_id = get_collection_id("Class - Human-Centered AI")
    if col_id is None:
        print("Collection not found")
        sys.exit(1)

    items = get_items_with_pdfs(col_id)
    first = items[0]

    print(f"Extracting from: {first['pdf_path']}")
    sentences = extract_sentences(first["pdf_path"])
    print(f"Extracted {len(sentences)} sentences\n")

    print("Classifying...")
    labels = classify(sentences)

    print("\n--- Sample classifications ---")
    for sentence, (label, confidence) in zip(sentences[:10], labels[:10]):
        print(f"  [{label} {confidence:.2f}] {sentence[:80]}")
