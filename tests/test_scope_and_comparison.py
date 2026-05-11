"""
Tests for generic scope control (out-of-scope refusal vs in-scope questions).
"""
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

REFUSAL_REPLY = "I can only help with SHL assessment recommendations and comparisons based on the SHL catalog."


def test_upsc_is_refused():
    """Generic unrelated question must be refused via scope control, not UPSC-specific hardcoding."""
    response = client.post("/chat", json={
        "messages": [{"role": "user", "content": "What is UPSC?"}]
    })
    assert response.status_code == 200
    data = response.json()
    assert data["recommendations"] == []
    assert data["reply"] == REFUSAL_REPLY


def test_machine_learning_is_refused():
    """Generic technical question unrelated to SHL assessments must be refused."""
    response = client.post("/chat", json={
        "messages": [{"role": "user", "content": "What is machine learning?"}]
    })
    assert response.status_code == 200
    data = response.json()
    assert data["recommendations"] == []
    assert data["reply"] == REFUSAL_REPLY


def test_opq_question_is_not_refused():
    """'What is OPQ?' is about an SHL assessment and must NOT be refused."""
    response = client.post("/chat", json={
        "messages": [{"role": "user", "content": "What is OPQ?"}]
    })
    assert response.status_code == 200
    data = response.json()
    # Must not be refusal
    assert data["reply"] != REFUSAL_REPLY
    # Recommendations can be empty (it's an info/compare query)
    assert isinstance(data["recommendations"], list)


def test_opq_gsa_comparison():
    """OPQ vs GSA comparison must return grounded catalog text, not just generic fallback."""
    response = client.post("/chat", json={
        "messages": [{"role": "user", "content": "What is the difference between OPQ and GSA?"}]
    })
    assert response.status_code == 200
    data = response.json()

    assert data["recommendations"] == []

    reply_lower = data["reply"].lower()
    # Must not be only the generic fallback sentence
    generic = "the requested items differ by their catalog category"
    assert reply_lower != generic

    # Should mention OPQ or Occupational Personality Questionnaire
    assert "opq" in reply_lower or "personality" in reply_lower or "occupational" in reply_lower
    # Should mention GSA or Global Skills Assessment
    assert "gsa" in reply_lower or "global skills" in reply_lower or "global" in reply_lower


def test_vague_in_scope_clarifies():
    """'I need an assessment' is vague but in-scope — must ask for context, not refuse."""
    response = client.post("/chat", json={
        "messages": [{"role": "user", "content": "I need an assessment"}]
    })
    assert response.status_code == 200
    data = response.json()

    assert data["reply"] != REFUSAL_REPLY
    assert data["recommendations"] == []
    reply_lower = data["reply"].lower()
    # Should ask for role/skills/context
    assert "?" in data["reply"] or "role" in reply_lower or "skill" in reply_lower
