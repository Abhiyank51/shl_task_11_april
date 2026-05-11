from fastapi.testclient import TestClient
from app.main import app
from app.catalog_loader import get_catalog

client = TestClient(app)

def test_recommendation_urls_are_grounded():
    response = client.post("/chat", json={
        "messages": [
            {"role": "user", "content": "I am hiring a Java developer and need assessments."}
        ]
    })
    
    assert response.status_code == 200
    data = response.json()
    
    _, valid_urls = get_catalog()
    
    # All returned URLs must be in the valid catalog URLs
    for rec in data["recommendations"]:
        assert rec["url"] in valid_urls
