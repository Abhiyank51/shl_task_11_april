from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_java_developer_recommendation():
    response = client.post("/chat", json={
        "messages": [
            {"role": "user", "content": "I am hiring a mid-level Java developer."}
        ]
    })
    
    assert response.status_code == 200
    data = response.json()
    
    # Should recommend 1 to 10 items
    assert 1 <= len(data["recommendations"]) <= 10
    
    # Check if the Java Developer Assessment from mock data is in the recommendations
    names = [rec["name"] for rec in data["recommendations"]]
    assert any("Java Developer Assessment" in name for name in names)
