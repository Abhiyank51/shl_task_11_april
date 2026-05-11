from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_refinement_query():
    response = client.post("/chat", json={
        "messages": [
            {"role": "user", "content": "I am hiring a Java developer."},
            {"role": "assistant", "content": "Here are some tests."},
            {"role": "user", "content": "Actually, add personality tests."}
        ]
    })
    
    assert response.status_code == 200
    data = response.json()
    
    # Should recommend items based on updated context
    assert 1 <= len(data["recommendations"]) <= 10
    names = [rec["name"] for rec in data["recommendations"]]
    
    # Should include both Java and Personality based on mock data
    assert any("Personality" in name for name in names)
