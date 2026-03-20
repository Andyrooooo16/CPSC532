import numpy as np
from pathlib import Path

CACHE_PATH = Path(__file__).parent / "embeddings_cache.npz"


def _load() -> dict:
    if CACHE_PATH.exists():
        data = np.load(CACHE_PATH, allow_pickle=True)
        return data["cache"].item()
    return {}


def _save(cache: dict) -> None:
    np.savez(CACHE_PATH, cache=cache)


def get(attachment_key: str, sentences: list[str]) -> dict | None:
    """
    Return cached entry for attachment_key if sentences match, else None.
    Entry shape: {"sentences": [...], "embeddings": np.ndarray, "labels": [...], "confidences": [...]}
    """
    cache = _load()
    if attachment_key not in cache:
        return None
    entry = cache[attachment_key]
    if entry["sentences"] != sentences:
        print(f"  Cache mismatch for {attachment_key}, will recompute.")
        return None
    print(f"  Cache hit for {attachment_key}")
    return entry


def store(attachment_key: str, sentences: list[str], embeddings: np.ndarray, labels: list[str], confidences: list[float]) -> None:
    """Save sentences, embeddings, and classifications for attachment_key."""
    cache = _load()
    cache[attachment_key] = {
        "sentences": sentences,
        "embeddings": embeddings,
        "labels": labels,
        "confidences": confidences,
    }
    _save(cache)
