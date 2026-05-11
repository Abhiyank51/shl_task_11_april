"""
app/scope_utils.py

Strict scope detection helpers for SHL catalog-grounded RAG.

Core rule:
- The assistant only answers SHL assessment recommendation, comparison,
  refinement, clarification, and known SHL catalog item explanation queries.
- General knowledge questions are refused unless they clearly ask about SHL
  assessments/catalog or mention a known SHL catalog product alias.
- Broad topics like Java, Python, DBMS, sales, leadership, communication,
  machine learning, etc. are NOT in-scope by themselves.
"""

import re


REFUSAL_REPLY = (
    "I can only help with SHL assessment recommendations and comparisons "
    "based on the SHL catalog."
)


KNOWN_CATALOG_ALIASES = {
    "opq",
    "opq32",
    "opq32r",
    "occupational personality questionnaire",
    "gsa",
    "global skills assessment",
    "verify g",
    "verify-g",
    "verify general ability",
    "verify interactive g",
    "verify numerical",
    "verify verbal",
    "verify deductive",
    "verify inductive",
    "sjt",
    "situational judgment",
    "situational judgement",
    "motivational questionnaire",
}


ASSESSMENT_PHRASES = {
    "shl",
    "shl catalog",
    "in shl catalog",
    "available in shl",
    "assessment",
    "assessments",
    "candidate assessment",
    "candidate assessments",
    "hiring assessment",
    "hiring assessments",
    "recruitment assessment",
    "recruitment assessments",
    "screening test",
    "screening tests",
    "screening assessment",
    "screening assessments",
    "evaluate candidate",
    "evaluate candidates",
    "assess candidate",
    "assess candidates",
    "skills to assess",
    "recommend assessment",
    "recommend assessments",
    "recommend test",
    "recommend tests",
    "suggest assessment",
    "suggest assessments",
    "suggest test",
    "suggest tests",
    "which assessment",
    "which assessments",
    "which test",
    "which tests",
    "what assessment",
    "what assessments",
    "what test",
    "what tests",
    "shortlist",
    "job description",
    "jd",
    "i am hiring",
    "we are hiring",
    "hiring for",
    "hiring a",
    "hiring an",
    "need an assessment",
    "need assessments",
    "need a test",
    "need tests",
    "test for",
    "tests for",
    "assessment for",
    "assessments for",
}


def _contains_phrase(text: str, phrase: str) -> bool:
    """
    Safe phrase matcher.

    Multi-word phrases are matched as substrings.
    Single-word phrases are matched with word boundaries so:
    - "test" does not match inside "best"
    - "shl" matches only as a separate token
    """

    text = (text or "").lower().strip()
    phrase = (phrase or "").lower().strip()

    if not text or not phrase:
        return False

    if " " in phrase:
        return phrase in text

    return re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", text) is not None


def is_known_catalog_query(text_lower: str) -> bool:
    """
    Returns True only when the query mentions a strong SHL catalog product alias.

    This intentionally does NOT allow broad single tokens like:
    java, python, sales, leadership, communication, dbms, developer, manager.
    """

    text_lower = (text_lower or "").lower().strip()
    return any(_contains_phrase(text_lower, alias) for alias in KNOWN_CATALOG_ALIASES)


def is_general_question_form(text_lower: str) -> bool:
    """
    Returns True for generic knowledge-seeking or task-completion questions.

    These must be refused unless the query has explicit SHL assessment intent
    or mentions a known SHL catalog product alias.
    """

    text_lower = (text_lower or "").lower().strip()

    patterns = [
        r"^\s*what is\b",
        r"^\s*what are\b",
        r"^\s*what's\b",
        r"^\s*who is\b",
        r"^\s*who are\b",
        r"^\s*where is\b",
        r"^\s*when is\b",
        r"^\s*why is\b",
        r"^\s*explain\b",
        r"^\s*define\b",
        r"^\s*describe\b",
        r"^\s*tell me about\b",
        r"^\s*give me information about\b",
        r"^\s*how to\b",
        r"^\s*how do i\b",
        r"^\s*write code\b",
        r"^\s*write a\b",
        r"^\s*solve\b",
        r"^\s*make\b",
        r"^\s*create\b",
        r"^\s*generate\b",
        r"^\s*summarize\b",
        r"^\s*translate\b",
    ]

    return any(re.search(pattern, text_lower) for pattern in patterns)


def is_comparison_form(text_lower: str) -> bool:
    """
    Returns True if the user is asking for a comparison.
    Generic comparisons should be refused unless they have SHL/catalog context.
    """

    text_lower = (text_lower or "").lower().strip()

    patterns = [
        r"\bcompare\b",
        r"\bcomparison\b",
        r"\bdifference between\b",
        r"\bdifferences between\b",
        r"\bvs\.?\b",
        r"\bversus\b",
    ]

    return any(re.search(pattern, text_lower) for pattern in patterns)


def has_assessment_intent(text_lower: str) -> bool:
    """
    Returns True only when the user clearly wants SHL assessment selection,
    recommendation, comparison, screening, or candidate evaluation.

    Broad words alone are not enough:
    - "What is Java?" -> False
    - "What is DBMS?" -> False
    - "What is sales?" -> False
    - "Compare Java and Python" -> False

    SHL-assessment intent is enough:
    - "What SHL assessment should I use for Java developer?" -> True
    - "Recommend tests for sales manager" -> True
    - "I need an assessment" -> True
    """

    text_lower = (text_lower or "").lower().strip()

    if not text_lower:
        return False

    if is_known_catalog_query(text_lower):
        return True

    for phrase in ASSESSMENT_PHRASES:
        if _contains_phrase(text_lower, phrase):
            return True

    action_patterns = [
        r"\bi (am|need|want|require|would like)\b.*\b(test|tests|assessment|assessments|evaluation|evaluate|hire|hiring|recruit|screening)\b",
        r"\bwe (are|need|want|require|would like)\b.*\b(test|tests|assessment|assessments|evaluation|evaluate|hire|hiring|recruit|screening)\b",
        r"\bhiring\s+(a|an|for)\b",
        r"\bhelp\s+(me\s+)?(hire|assess|evaluate|screen)\b",
        r"\b(select|choose|find)\s+(a\s+|an\s+)?(test|tests|assessment|assessments)\b",
        r"\b(best|suitable|right)\s+(test|tests|assessment|assessments)\b",
        r"\b(test|tests|assessment|assessments)\s+(for|to assess|to evaluate)\b",
        r"\b(assess|evaluate|screen)\s+(for\s+)?(a\s+|an\s+)?(candidate|candidates|role|job)\b",
    ]

    return any(re.search(pattern, text_lower) for pattern in action_patterns)


def is_catalog_item_mentioned(text_lower: str) -> bool:
    """
    Strict catalog mention check.

    Do NOT perform loose token matching against catalog names.
    Only strong SHL catalog aliases are allowed here.
    """

    return is_known_catalog_query(text_lower)


def is_assessment_scope(text_lower: str) -> bool:
    """
    Backward-compatible helper used by guardrails.py.

    A query is in assessment scope only if:
    - it has explicit SHL assessment/recommendation/hiring intent, OR
    - it mentions a known SHL catalog/product alias.
    """

    text_lower = (text_lower or "").lower().strip()
    return has_assessment_intent(text_lower) or is_known_catalog_query(text_lower)


def is_general_external_question(text_lower: str) -> bool:
    """Backward-compatible alias."""

    return is_general_question_form(text_lower)


def should_refuse_by_scope(text: str) -> bool:
    """
    Generic strict scope gate.

    Refuse any general knowledge/instructional/comparison query unless it has:
    - explicit SHL assessment intent, OR
    - a known SHL catalog alias like OPQ/GSA/Verify.

    This catches:
    - What is the full form of UPSC?
    - What is DBMS?
    - What is Java?
    - What is the best recipe to make biryani?
    - Explain recursion.
    - Compare Java and Python.
    """

    text_lower = (text or "").lower().strip()

    if not text_lower:
        return False
    if is_known_catalog_query(text_lower):
        return False
    if has_assessment_intent(text_lower):
        return False
    if is_general_question_form(text_lower):
        return True
    is_comparison = (
        re.search(r"\bcompare\b", text_lower)
        or re.search(r"\bcomparison\b", text_lower)
        or "difference between" in text_lower
        or "differences between" in text_lower
        or re.search(r"\bvs\.?\b", text_lower)
        or re.search(r"\bversus\b", text_lower)
    )

    if is_comparison:
        return True
    return False