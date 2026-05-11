from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_vague_query():
    response = client.post("/chat", json={
        "messages": [
            {"role": "user", "content": "I need an assessment."}
        ]
    })
    
    assert response.status_code == 200
    data = response.json()
    
    # Should ask clarification, recommendations must be empty
    assert len(data["recommendations"]) == 0
    assert "role" in data["reply"].lower() or "skill" in data["reply"].lower() or "?" in data["reply"]
    assert data["end_of_conversation"] is False
