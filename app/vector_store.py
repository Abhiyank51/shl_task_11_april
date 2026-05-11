"""
app/vector_store.py

Local FAISS-based vector index over SHL catalog.
Built once at startup and cached globally.
Never rebuilt per-request.
"""
from __future__ import annotations

import os
import numpy as np
import functools
from typing import List, Dict, Any, Tuple, Optional

# --- lazy imports to avoid slow startup penalty ---
_model = None
_index = None
_catalog_items: List[Dict[str, Any]] = []

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SIMILARITY_THRESHOLD = 0.35  # minimum cosine similarity to consider a result relevant


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def _build_index(items: List[Dict[str, Any]]):
    """Build FAISS index from catalog items. Called once."""
    global _index, _catalog_items
    import faiss

    model = _get_model()
    corpus = [item.get("search_text", "") or _fallback_text(item) for item in items]
    embeddings = model.encode(corpus, batch_size=64, show_progress_bar=False, convert_to_numpy=True)
    embeddings = embeddings.astype(np.float32)

    # Normalize for cosine similarity via inner product
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-9, norms)
    embeddings = embeddings / norms

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product = cosine after normalization
    index.add(embeddings)

    _index = index
    _catalog_items = list(items)


def _fallback_text(item: Dict[str, Any]) -> str:
    parts = [f"Name: {item.get('name', '')}"]
    if item.get("description"):
        parts.append(f"Description: {item['description']}")
    if item.get("keys"):
        parts.append(f"Categories: {', '.join(item['keys'])}")
    if item.get("job_levels"):
        parts.append(f"Job Levels: {', '.join(item['job_levels'])}")
    if item.get("duration"):
        parts.append(f"Duration: {item['duration']}")
    return " ".join(parts)


def ensure_index_built():
    """Ensure FAISS index is built from catalog. Safe to call multiple times."""
    global _index
    if _index is None:
        from app.catalog_loader import get_catalog
        items, _ = get_catalog()
        if items:
            _build_index(items)


def search_vector_index(query: str, top_k: int = 20) -> List[Tuple[float, Dict[str, Any]]]:
    """
    Search FAISS index for top_k most semantically similar catalog items.
    Returns list of (cosine_score, catalog_item) tuples, sorted descending.
    """
    ensure_index_built()
    if _index is None or not _catalog_items:
        return []

    model = _get_model()
    q_emb = model.encode([query], convert_to_numpy=True).astype(np.float32)
    norm = np.linalg.norm(q_emb)
    if norm > 0:
        q_emb = q_emb / norm

    k = min(top_k, len(_catalog_items))
    scores, indices = _index.search(q_emb, k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        results.append((float(score), _catalog_items[idx]))
    return results


def get_top_similarity_score(query: str) -> float:
    """Returns cosine similarity of the top-1 result for a query."""
    results = search_vector_index(query, top_k=1)
    if not results:
        return 0.0
    return results[0][0]


def has_relevant_catalog_evidence(query_text: str, context: dict) -> bool:
    """
    Retrieval confidence gate.
    Returns True only if there is meaningful catalog evidence for this query.
    """
    from app.scope_utils import has_assessment_intent, is_known_catalog_query

    text_lower = query_text.lower()

    # Always allow known catalog items by alias (OPQ, GSA, Verify G etc.)
    if is_known_catalog_query(text_lower):
        return True

    # Always allow if comparison context has specific targets
    if context.get("comparisons"):
        return True

    # Allow if there is assessment intent AND vector similarity is strong enough
    if has_assessment_intent(text_lower):
        top_score = get_top_similarity_score(query_text)
        if top_score >= SIMILARITY_THRESHOLD:
            return True
        # Even if vector score is low, allow if context has enough extraction signals
        if context.get("roles") or context.get("skills") or context.get("test_type_preferences"):
            return True

    return False
