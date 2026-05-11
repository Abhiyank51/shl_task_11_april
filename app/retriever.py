"""
app/retriever.py

Hybrid retrieval combining:
  0.55 * vector similarity
  0.25 * TF-IDF cosine similarity
  0.20 * rule-based boosts

Comparison queries use alias-priority matching, bypassing hybrid scoring.
"""

import functools
import re
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.catalog_loader import get_catalog


W_VECTOR = 0.55
W_TFIDF = 0.25
W_RULES = 0.20

MIN_RETRIEVAL_SCORE = 0.03


ALIASES = {
    "gsa": [
        "gsa",
        "global skills assessment",
        "global skills",
    ],
    "opq": [
        "opq",
        "opq32r",
        "opq32",
        "occupational personality questionnaire",
    ],
    "verify g": [
        "verify g",
        "verify-g",
        "verify general ability",
        "verify interactive g",
    ],
    "java 8": ["java 8"],
    "core java": ["core java"],
}


@functools.lru_cache(maxsize=1)
def get_tfidf_index():
    catalog_items, _ = get_catalog()

    if not catalog_items:
        return catalog_items, None, None

    corpus = [item.get("search_text", "") for item in catalog_items]

    vectorizer = TfidfVectorizer(stop_words="english")

    try:
        matrix = vectorizer.fit_transform(corpus)
    except ValueError:
        matrix = None

    return catalog_items, vectorizer, matrix


def is_out_of_scope_solution(item: Dict[str, Any]) -> bool:
    text = f"{item.get('name', '')} {item.get('url', '')}".lower()

    blocked_terms = [
        " solution",
        "-solution",
        "job-focused-assessment",
        "job focused assessment",
        "jfa",
    ]

    return any(term in text for term in blocked_terms)


def comparison_priority(item: Dict[str, Any], aliases: List[str]) -> int:
    name = item.get("name", "").lower()
    description = item.get("description", "").lower()
    search_text = item.get("search_text", "").lower()

    score = 0

    for alias in aliases:
        alias = alias.lower()

        if alias == name:
            score = max(score, 100)
        elif alias in name:
            score = max(score, 80)
        elif alias in description:
            score = max(score, 40)
        elif alias in search_text:
            score = max(score, 20)

    # Prefer direct assessments over report-only artifacts.
    if "assessment" in name:
        score += 15

    if "questionnaire" in name:
        score += 10

    if "report" in name:
        score -= 10

    if "development report" in name:
        score -= 15

    return score


def _rule_boost(item: Dict[str, Any], context: Dict[str, Any]) -> float:
    boost = 0.0

    name_lower = item.get("name", "").lower()
    desc_lower = item.get("description", "").lower()

    for skill in context.get("skills", []):
        skill = skill.lower()

        if skill in name_lower:
            boost += 0.5
        elif skill in desc_lower:
            boost += 0.2

    for role in context.get("roles", []):
        role = role.lower()

        if role in name_lower:
            boost += 0.3
        elif role in desc_lower:
            boost += 0.1

    for category in context.get("test_type_preferences", []):
        if category in item.get("keys", []):
            boost += 0.4

    for level in context.get("seniority", []):
        if level in item.get("job_levels", []):
            boost += 0.3

    if context.get("duration_preference") == "short":
        duration = item.get("duration", "").lower()
        match = re.search(r"(\d+)", duration)

        if match and int(match.group(1)) <= 20:
            boost += 0.4

    if context.get("remote_preference") == "yes" and item.get("remote") == "yes":
        boost += 0.2

    if context.get("adaptive_preference") == "yes" and item.get("adaptive") == "yes":
        boost += 0.2

    return boost


def _dedupe_by_url(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen_urls: Set[str] = set()
    result: List[Dict[str, Any]] = []

    for item in items:
        url = item.get("url", "")

        if not url or url in seen_urls:
            continue

        seen_urls.add(url)
        result.append(item)

    return result


def _comparison_results(
    catalog_items: List[Dict[str, Any]],
    comparisons: List[str],
    top_k: int,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    seen_urls: Set[str] = set()

    for target in [comparison.lower() for comparison in comparisons]:
        aliases = ALIASES.get(target, [target])
        candidates: List[Tuple[int, Dict[str, Any]]] = []

        for item in catalog_items:
            pool = (
                f"{item.get('name', '')} "
                f"{item.get('description', '')} "
                f"{' '.join(item.get('keys', []))} "
                f"{item.get('search_text', '')}"
            ).lower()

            if any(alias in pool for alias in aliases):
                candidates.append((comparison_priority(item, aliases), item))

        if not candidates:
            continue

        candidates.sort(key=lambda pair: pair[0], reverse=True)

        # Add the single strongest match for each requested target.
        best_item = candidates[0][1]
        url = best_item.get("url", "")

        if url and url not in seen_urls:
            seen_urls.add(url)
            results.append(best_item)

    return results[:top_k]


def retrieve_items(
    context: Dict[str, Any],
    query_text: str,
    top_k: int = 10,
    is_compare: bool = False,
) -> List[Dict[str, Any]]:
    catalog_items, vectorizer, tfidf_matrix = get_tfidf_index()

    if not catalog_items:
        return []

    if is_compare and context.get("comparisons"):
        comparison_items = _comparison_results(
            catalog_items=catalog_items,
            comparisons=context.get("comparisons", []),
            top_k=top_k,
        )

        if comparison_items:
            return comparison_items
    if is_compare:
        return []

    try:
        from app.vector_store import search_vector_index

        vector_results = search_vector_index(query_text, top_k=len(catalog_items))
        vector_score_map: Dict[str, float] = {
            item.get("url", ""): score
            for score, item in vector_results
            if item.get("url")
        }

    except Exception:
        vector_score_map = {}

    if vectorizer and tfidf_matrix is not None:
        try:
            query_vector = vectorizer.transform([query_text])
            tfidf_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()
        except ValueError:
            tfidf_scores = np.zeros(len(catalog_items))
    else:
        tfidf_scores = np.zeros(len(catalog_items))

    all_boosts = [_rule_boost(item, context) for item in catalog_items]
    max_boost = max(all_boosts) if any(boost > 0 for boost in all_boosts) else 1.0

    scored_items: List[Tuple[float, Dict[str, Any]]] = []

    for index, item in enumerate(catalog_items):
        vector_score = vector_score_map.get(item.get("url", ""), 0.0)
        tfidf_score = float(tfidf_scores[index])
        rule_score = all_boosts[index] / max_boost if max_boost > 0 else 0.0

        final_score = (
            W_VECTOR * vector_score
            + W_TFIDF * tfidf_score
            + W_RULES * rule_score
        )

        scored_items.append((final_score, item))

    scored_items.sort(key=lambda pair: pair[0], reverse=True)

    top_items = [
        item
        for score, item in scored_items
        if score >= MIN_RETRIEVAL_SCORE
    ]

    if not top_items:
        return []

    preferred = [
        item
        for item in top_items
        if not is_out_of_scope_solution(item)
    ]

    fallback = [
        item
        for item in top_items
        if is_out_of_scope_solution(item)
    ]

    candidates = preferred if preferred else fallback
    deduped_items = _dedupe_by_url(candidates)

    return deduped_items[:top_k]


def find_catalog_item_by_alias(query_text: str) -> Optional[Dict[str, Any]]:
    catalog_items, _, _ = get_tfidf_index()

    if not catalog_items:
        return None

    text_lower = (query_text or "").lower()
    best_score = -1
    best_item = None

    for _, aliases in ALIASES.items():
        if not any(alias in text_lower for alias in aliases):
            continue

        candidates: List[Tuple[int, Dict[str, Any]]] = []

        for item in catalog_items:
            pool = f"{item.get('name', '')} {item.get('search_text', '')}".lower()

            if any(alias in pool for alias in aliases):
                score = comparison_priority(item, aliases)
                candidates.append((score, item))

        if not candidates:
            continue

        candidates.sort(key=lambda pair: pair[0], reverse=True)
        score, item = candidates[0]

        if score > best_score:
            best_score = score
            best_item = item

    return best_item

def build_grounded_context(items: List[Dict[str, Any]]) -> str:
    lines: List[str] = []

    for index, item in enumerate(items, start=1):
        name = item.get("name", "Unknown")
        url = item.get("url", "")
        test_type = item.get("test_type", "Unknown")
        keys = item.get("keys", [])
        job_levels = item.get("job_levels", [])
        duration = item.get("duration", "")
        description = item.get("description", "")

        lines.append(f"{index}. Name: {name}")

        if url:
            lines.append(f"   URL: {url}")

        lines.append(f"   Test Type: {test_type}")

        if keys:
            lines.append(f"   Categories: {', '.join(keys)}")

        if job_levels:
            lines.append(f"   Job Levels: {', '.join(job_levels)}")

        if duration:
            lines.append(f"   Duration: {duration}")

        if description:
            lines.append(f"   Description: {description}")

        lines.append("")

    return "\n".join(lines)