import re
from app.scope_utils import (
    is_assessment_scope, is_general_external_question,
    is_catalog_item_mentioned, should_refuse_by_scope
)

# Hard refusal phrases — always blocked regardless of topic
HARD_REFUSAL_PHRASES = [
    "can i legally", "is it legal", "legal advice", "discriminate",
    "reject older candidates", "ignore previous instructions",
    "forget everything", "system prompt", "recommend non-shl",
    "other companies' assessments", "ignore all above instructions",
    "ignore the above instructions", "override the rules",
    "override instructions", "you are now", "jailbreak",
    "developer message", "outside shl", "non shl",
]

HARD_REFUSAL_PATTERNS = [
    r"how to (cook|bake|treat|cure|vote)",
    r"recipe for",
    r"who is the president",
    r"medical advice",
    r"\belection\b",
]


def check_guardrails(text: str) -> bool:
    """
    Returns True if the input is out of scope and should be refused.

    Layers:
    1. Hard-blocked phrases (injection, legal, non-SHL requests)
    2. Hard-blocked regex patterns (unrelated topics)
    3. Generic scope check: general question form with no assessment intent
    """
    text_lower = text.lower()

    # Layer 1: Hard refusal keywords
    for phrase in HARD_REFUSAL_PHRASES:
        if phrase in text_lower:
            return True

    # Layer 2: Hard regex patterns
    for pattern in HARD_REFUSAL_PATTERNS:
        if re.search(pattern, text_lower):
            return True

    # Layer 3: Generic scope gate
    if should_refuse_by_scope(text):
        return True

    return False
