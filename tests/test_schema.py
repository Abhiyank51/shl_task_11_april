from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_schema_malformed_input():
    # Sending missing 'messages'
    response = client.post("/chat", json={"wrong_key": []})
    assert response.status_code == 422 # Pydantic validation error

def test_schema_invalid_role():
    # Sending invalid role (e.g., 'system' instead of 'user' or 'assistant')
    response = client.post("/chat", json={
        "messages": [
            {"role": "system", "content": "I am hiring a Java developer."}
        ]
    })
    assert response.status_code == 422 # Pydantic validation error

def test_schema_empty_messages():
    # Sending empty messages array
    response = client.post("/chat", json={"messages": []})
    assert response.status_code == 200 # App logic handles it, not pydantic
    data = response.json()
    assert "reply" in data
    assert "Please provide a conversation message" in data["reply"]
    assert data["recommendations"] == []
    assert data["end_of_conversation"] is False

def test_schema_valid_input_structure():
    response = client.post("/chat", json={
        "messages": [
            {"role": "user", "content": "I am hiring a Java developer."}
        ]
    })
    assert response.status_code == 200
    data = response.json()
    
    # Check exact keys
    assert "reply" in data
    assert "recommendations" in data
    assert "end_of_conversation" in data
    
    # Types
    assert isinstance(data["reply"], str)
    assert isinstance(data["recommendations"], list)
    assert isinstance(data["end_of_conversation"], bool)
    
    # If there are recommendations, check their exact keys
    for rec in data["recommendations"]:
        assert "name" in rec
        assert "url" in rec
        assert "test_type" in rec
        assert "description" not in rec
        assert "score" not in rec
