"""
Comprehensive scope + intent + vector tests covering all 10 scenarios from the assignment.
"""
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

REFUSAL_REPLY = "I can only help with SHL assessment recommendations and comparisons based on the SHL catalog."


def _is_refusal(data: dict) -> bool:
    return data["reply"] == REFUSAL_REPLY and data["recommendations"] == []


def _post(content: str, history=None):
    messages = history or []
    messages.append({"role": "user", "content": content})
    r = client.post("/chat", json={"messages": messages})
    assert r.status_code == 200
    return r.json()


# ── REFUSE cases ──────────────────────────────────────────────────────────────

def test_refuse_what_is_java():
    """'What is Java?' -> refuse (general CS question, no assessment intent)."""
    data = _post("What is Java?")
    assert _is_refusal(data), f"Expected refusal, got: {data['reply'][:100]}"


def test_refuse_what_is_sales():
    """'What is sales?' -> refuse."""
    data = _post("What is sales?")
    assert _is_refusal(data), f"Expected refusal, got: {data['reply'][:100]}"


def test_refuse_what_is_leadership():
    """'What is leadership?' -> refuse."""
    data = _post("What is leadership?")
    assert _is_refusal(data), f"Expected refusal, got: {data['reply'][:100]}"


def test_refuse_what_is_communication():
    """'What is communication?' -> refuse."""
    data = _post("What is communication?")
    assert _is_refusal(data), f"Expected refusal, got: {data['reply'][:100]}"


def test_refuse_what_is_machine_learning():
    """'What is machine learning?' -> refuse."""
    data = _post("What is machine learning?")
    assert _is_refusal(data), f"Expected refusal, got: {data['reply'][:100]}"


def test_refuse_upsc():
    """'What is UPSC?' -> refuse."""
    data = _post("What is UPSC?")
    assert _is_refusal(data), f"Expected refusal, got: {data['reply'][:100]}"


def test_refuse_explain_recursion():
    """'Explain recursion.' -> refuse."""
    data = _post("Explain recursion.")
    assert _is_refusal(data), f"Expected refusal, got: {data['reply'][:100]}"


def test_refuse_write_code():
    """'Write code for binary search.' -> refuse."""
    data = _post("Write code for binary search.")
    assert _is_refusal(data), f"Expected refusal, got: {data['reply'][:100]}"


# ── CLARIFY cases ─────────────────────────────────────────────────────────────

def test_clarify_i_need_an_assessment():
    """'I need an assessment.' -> clarify (in-scope but vague)."""
    data = _post("I need an assessment")
    assert data["reply"] != REFUSAL_REPLY, "Should not refuse"
    assert data["recommendations"] == [], "Should not recommend yet"


def test_clarify_suggest_tests():
    """'Suggest tests.' -> clarify."""
    data = _post("Suggest tests")
    assert data["reply"] != REFUSAL_REPLY
    assert data["recommendations"] == []


# ── RECOMMEND cases ───────────────────────────────────────────────────────────

def test_recommend_java_assessments():
    """'Recommend SHL assessments for Java developer.' -> 1-10 items."""
    data = _post("Recommend SHL assessments for Java developer")
    assert data["reply"] != REFUSAL_REPLY
    assert len(data["recommendations"]) >= 1
    for rec in data["recommendations"]:
        assert "name" in rec and "url" in rec and "test_type" in rec
        assert "shl.com" in rec["url"]


def test_recommend_java_skills_test():
    """'I need a test for Java programming skills.' -> recommendations."""
    data = _post("I need a test for Java programming skills")
    assert len(data["recommendations"]) >= 1


def test_recommend_sales_manager():
    """'I am hiring a sales manager.' -> recommendations."""
    data = _post("I am hiring a sales manager")
    assert data["reply"] != REFUSAL_REPLY
    assert len(data["recommendations"]) >= 1


def test_recommend_communication_skills():
    """'What assessment should I use for communication skills?' -> recommendations."""
    data = _post("What assessment should I use for communication skills?")
    assert data["reply"] != REFUSAL_REPLY
    assert len(data["recommendations"]) >= 1


# ── EXPLAIN CATALOG ITEM cases ────────────────────────────────────────────────

def test_explain_opq():
    """'What is OPQ?' -> grounded catalog explanation, no refusal."""
    data = _post("What is OPQ?")
    assert data["reply"] != REFUSAL_REPLY
    reply_lower = data["reply"].lower()
    assert "opq" in reply_lower or "personality" in reply_lower or "occupational" in reply_lower


def test_explain_gsa():
    """'What is GSA?' -> grounded catalog explanation."""
    data = _post("What is GSA?")
    assert data["reply"] != REFUSAL_REPLY
    reply_lower = data["reply"].lower()
    assert "gsa" in reply_lower or "global skills" in reply_lower or "assessment" in reply_lower


def test_explain_verify_g():
    """'Tell me about Verify G.' -> grounded catalog explanation."""
    data = _post("Tell me about Verify G")
    assert data["reply"] != REFUSAL_REPLY


# ── COMPARE cases ─────────────────────────────────────────────────────────────

def test_compare_opq_gsa():
    """'What is the difference between OPQ and GSA?' -> grounded comparison."""
    data = _post("What is the difference between OPQ and GSA?")
    assert data["recommendations"] == []
    assert data["reply"] != REFUSAL_REPLY
    reply_lower = data["reply"].lower()
    assert "opq" in reply_lower or "personality" in reply_lower or "occupational" in reply_lower
    assert "gsa" in reply_lower or "global skills" in reply_lower


def test_compare_opq_and_gsa_explicit():
    """'Compare OPQ and GSA.' -> grounded comparison."""
    data = _post("Compare OPQ and GSA")
    assert data["reply"] != REFUSAL_REPLY
    assert data["recommendations"] == []


# ── SCHEMA compliance ─────────────────────────────────────────────────────────

def test_all_responses_have_correct_schema():
    """Every response must have reply (str), recommendations (list), end_of_conversation (bool)."""
    queries = [
        "What is OPQ?",
        "I need an assessment",
        "I am hiring a Java developer",
        "What is UPSC?",
    ]
    for q in queries:
        data = _post(q)
        assert isinstance(data["reply"], str), f"reply not str for: {q}"
        assert isinstance(data["recommendations"], list), f"recommendations not list for: {q}"
        assert isinstance(data["end_of_conversation"], bool), f"end_of_conversation not bool for: {q}"
