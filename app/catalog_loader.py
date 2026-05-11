import json
import os
from pathlib import Path
from typing import List, Dict, Any, Set

def derive_test_type(keys: List[str]) -> str:
    if not keys:
        return "Unknown"
        
    mapping = {
        "Knowledge & Skills": "K",
        "Ability & Aptitude": "A",
        "Personality & Behavior": "P",
        "Competencies": "C",
        "Development & 360": "D",
        "Biodata & Situational Judgment": "B/S",
        "Assessment Exercises": "E"
    }
    
    mapped_types = []
    for k in keys:
        if k in mapping:
            mapped_types.append(mapping[k])
            
    if not mapped_types:
        return "Unknown"
        
    return ",".join(mapped_types)

def build_search_text(item: Dict[str, Any]) -> str:
    lines = []
    lines.append(f"Name: {item.get('name', '')}")
    lines.append(f"Description: {item.get('description', '')}")
    
    job_levels = item.get('job_levels', [])
    if job_levels:
        lines.append(f"Job Levels: {', '.join(job_levels)}")
        
    languages = item.get('languages', [])
    if languages:
        lines.append(f"Languages: {', '.join(languages)}")
        
    duration = item.get('duration', '')
    if duration:
        lines.append(f"Duration: {duration}")
        
    remote = item.get('remote', '')
    if remote:
        lines.append(f"Remote: {remote}")
        
    adaptive = item.get('adaptive', '')
    if adaptive:
        lines.append(f"Adaptive: {adaptive}")
        
    keys = item.get('keys', [])
    if keys:
        lines.append(f"Categories: {', '.join(keys)}")
        
    return "\n".join(lines)

def load_catalog() -> tuple[List[Dict[str, Any]], Set[str]]:
    default_path = "data/shl_catalog_main.json"
    catalog_path = os.environ.get("SHL_CATALOG_PATH", default_path)
    
    path_obj = Path(catalog_path)
    
    if not path_obj.exists() or path_obj.stat().st_size == 0:
        raise RuntimeError(f"Catalog file not found or empty: {catalog_path}")
        
    with open(catalog_path, 'r', encoding='utf-8') as f:
        raw_items = json.load(f)
        
    normalized_items = []
    valid_urls = set()
    
    for item in raw_items:
        if not item.get("name") or not item.get("link"):
            continue
        # Some catalogs might be wrapped differently, assuming list of dicts based on schema
        norm_item = {
            "entity_id": item.get("entity_id", ""),
            "name": item.get("name", ""),
            "url": item.get("link", ""),
            "description": item.get("description", ""),
            "job_levels": item.get("job_levels", []),
            "languages": item.get("languages", []),
            "duration": item.get("duration", ""),
            "remote": item.get("remote", ""),
            "adaptive": item.get("adaptive", ""),
            "keys": item.get("keys", []),
            "test_type": derive_test_type(item.get("keys", [])),
            "search_text": build_search_text(item)
        }
        normalized_items.append(norm_item)
        if norm_item["url"]:
            valid_urls.add(norm_item["url"])
            
    return normalized_items, valid_urls

import functools

@functools.lru_cache(maxsize=1)
def get_catalog() -> tuple[List[Dict[str, Any]], Set[str]]:
    return load_catalog()
