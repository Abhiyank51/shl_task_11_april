import os
import json
import pytest
from pathlib import Path

MOCK_CATALOG = [
    {
        "entity_id": "1",
        "name": "Java Developer Assessment",
        "link": "https://www.shl.com/products/product-catalog/view/java-dev/",
        "description": "Assesses Java coding skills.",
        "job_levels": ["Mid-Professional"],
        "languages": ["English"],
        "duration": "45 minutes",
        "remote": "yes",
        "adaptive": "no",
        "keys": ["Knowledge & Skills"]
    },
    {
        "entity_id": "2",
        "name": "OPQ Personality Test",
        "link": "https://www.shl.com/products/product-catalog/view/opq/",
        "description": "Measures behavior and personality traits.",
        "job_levels": ["Graduate", "Mid-Professional", "Manager"],
        "languages": ["English", "French"],
        "duration": "30 minutes",
        "remote": "yes",
        "adaptive": "yes",
        "keys": ["Personality & Behavior"]
    },
    {
        "entity_id": "3",
        "name": "GSA Cognitive Ability",
        "link": "https://www.shl.com/products/product-catalog/view/gsa/",
        "description": "General cognitive ability and aptitude.",
        "job_levels": ["Entry-Level", "Mid-Professional"],
        "languages": ["English"],
        "duration": "20 minutes",
        "remote": "yes",
        "adaptive": "no",
        "keys": ["Ability & Aptitude"]
    },
    {
        "entity_id": "4",
        "name": "Pre-packaged Java Job Solution",
        "link": "https://www.shl.com/products/product-catalog/view/job-focused-assessment-java/",
        "description": "A pre-packaged solution for java roles.",
        "job_levels": ["Mid-Professional"],
        "languages": ["English"],
        "duration": "60 minutes",
        "remote": "yes",
        "adaptive": "no",
        "keys": ["Knowledge & Skills"]
    }
]

@pytest.fixture(autouse=True, scope="session")
def setup_mock_catalog(tmp_path_factory):
    # Create a temporary mock catalog file for tests
    temp_dir = tmp_path_factory.mktemp("data")
    mock_file = temp_dir / "shl_catalog_main.json"
    
    with open(mock_file, "w", encoding="utf-8") as f:
        json.dump(MOCK_CATALOG, f)
        
    # Set the environment variable to point to this mock file
    os.environ["SHL_CATALOG_PATH"] = str(mock_file)
    
    # We also need to clear the lru_cache in catalog_loader to ensure it loads our new file
    from app.catalog_loader import get_catalog
    get_catalog.cache_clear()
    
    yield
    
    # Teardown
    get_catalog.cache_clear()
