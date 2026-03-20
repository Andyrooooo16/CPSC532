import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

MODEL_NAME = "allenai/specter2_base"
BATCH_SIZE = 32

_tokenizer = None
_model = None


def _load_model():
    print("Loading SPECTER2 base model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME, use_safetensors=True)
    model.eval()
    print("SPECTER2 ready.")
    return tokenizer, model


def _mean_pool(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)


def embed(sentences: list[str]) -> np.ndarray:
    """Return a (N, 768) float32 numpy array of sentence embeddings."""
    global _tokenizer, _model
    if _model is None:
        _tokenizer, _model = _load_model()

    all_embeddings = []
    total_batches = (len(sentences) + BATCH_SIZE - 1) // BATCH_SIZE
    for i in range(0, len(sentences), BATCH_SIZE):
        batch = sentences[i : i + BATCH_SIZE]
        print(f"  Embedding batch {i // BATCH_SIZE + 1}/{total_batches}")
        inputs = _tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
        with torch.no_grad():
            outputs = _model(**inputs)
        vecs = _mean_pool(outputs.last_hidden_state, inputs["attention_mask"])
        all_embeddings.append(vecs.numpy())

    return np.vstack(all_embeddings).astype(np.float32)


if __name__ == "__main__":
    from zotero_db import get_collection_id, get_items_with_pdfs
    from extractor import extract_sentences

    col_id = get_collection_id("Class - Human-Centered AI")
    items = get_items_with_pdfs(col_id)
    first = items[0]

    print(f"Extracting sentences from: {first['pdf_path']}")
    sentences = extract_sentences(first["pdf_path"])
    print(f"Sentences: {len(sentences)}")

    embeddings = embed(sentences)
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"First vector (first 5 dims): {embeddings[0, :5]}")
