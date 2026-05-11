from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_comparison_query():
    response = client.post("/chat", json={
        "messages": [
            {"role": "user", "content": "What is the difference between OPQ and GSA?"}
        ]
    })
    
    assert response.status_code == 200
    data = response.json()
    
    assert "reply" in data
    assert isinstance(data["reply"], str)
    
    # Pure compare should not return recommendations in the array, only text
    assert len(data["recommendations"]) == 0
    
    reply_lower = data["reply"].lower()
    
    # Check if it hit the specific comparison or the generic fallback
    is_fallback = "these assessments differ by test type" in reply_lower
    if not is_fallback:
        assert "opq" in reply_lower or "personality" in reply_lower
        assert "gsa" in reply_lower or "global" in reply_lower or "cognitive" in reply_lower
