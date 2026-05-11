import re
from typing import Dict, Any, List
from langgraph.graph import StateGraph, END
from app.schemas import AgentState
from app.guardrails import check_guardrails
from app.context_extractor import extract_context, has_enough_context
from app.retriever import retrieve_items, build_grounded_context, find_catalog_item_by_alias
from app.llm_client import GeminiClient
from app.prompts import (
    RECOMMENDATION_PROMPT, REFINEMENT_PROMPT, COMPARISON_PROMPT,
    CLARIFICATION_PROMPT, FALLBACKS
)
from app.response_utils import build_deterministic_comparison_reply
from app.scope_utils import has_assessment_intent, is_known_catalog_query

llm_client = GeminiClient()

# ── Prompts for explain intent ────────────────────────────────────────────────
EXPLAIN_PROMPT = """CATALOG CONTEXT:
{context}

CONVERSATION:
{conversation}

TASK:
The user wants to understand this SHL catalog item.
Explain it clearly using ONLY the provided catalog context fields:
name, description, test_type, categories, job levels, duration.
Do not use outside knowledge. Do not invent details.
"""


# ── Nodes ─────────────────────────────────────────────────────────────────────

def validate_input_node(state: AgentState) -> AgentState:
    messages = state.get("messages", [])
    if not messages:
        return state

    conversation_text = ""
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        conversation_text += f"{role.capitalize()}: {content}\n"

    latest_user = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            latest_user = msg.get("content", "")
            break

    return {
        "conversation_text": conversation_text,
        "latest_user_message": latest_user,
    }


def guardrail_node(state: AgentState) -> AgentState:
    text = state.get("latest_user_message", "")
    if check_guardrails(text):
        return {"refused": True, "intent": "refuse"}
    return {"refused": False}


def intent_node(state: AgentState) -> AgentState:
    if state.get("refused"):
        return state

    text_lower = state.get("latest_user_message", "").lower()

    # Done signals
    if re.search(r"^\s*(thanks|thank you|done|enough|bye)\b", text_lower):
        return {"intent": "done", "end_of_conversation": True}

    # Compare signals
    if "compare" in text_lower or "difference between" in text_lower or " vs " in text_lower:
        return {"intent": "compare"}

    # Explain catalog item: "What is OPQ?", "Tell me about GSA", "What is Verify G?"
    is_general_question = re.search(
        r"^\s*(what is|what are|tell me about|explain|describe|how does)\b", text_lower
    )
    if is_general_question and is_known_catalog_query(text_lower):
        return {"intent": "explain_catalog_item"}

    # Refine signals (multi-turn)
    if any(kw in text_lower for kw in ["actually", "instead", "remove", "add", "include"]):
        if len(state.get("messages", [])) > 2:
            return {"intent": "refine"}

    return {"intent": "recommend"}


def context_extraction_node(state: AgentState) -> AgentState:
    context = extract_context(state.get("conversation_text", ""))

    intent = state.get("intent")
    if intent in ["recommend", "refine"]:
        if not has_enough_context(context, state.get("latest_user_message", "")):
            intent = "clarify"

    return {"context": context, "intent": intent}


def retrieval_confidence_node(state: AgentState) -> AgentState:
    """
    After context extraction, check if we have real catalog evidence.
    If not, refuse rather than guess.
    """
    intent = state.get("intent")
    if intent in ["clarify", "refuse", "done", "compare", "explain_catalog_item"]:
        return state  # skip confidence check for these

    context = state.get("context", {})
    query_text = state.get("conversation_text", "")

    try:
        from app.vector_store import has_relevant_catalog_evidence
        has_evidence = has_relevant_catalog_evidence(query_text, context)
    except Exception:
        has_evidence = True  # fail open if vector store is unavailable

    if not has_evidence:
        return {"refused": True, "intent": "refuse"}

    return state


def clarification_node(state: AgentState) -> AgentState:
    prompt = CLARIFICATION_PROMPT.format(conversation=state.get("conversation_text", ""))
    reply = llm_client.generate_reply(prompt, "clarify")
    return {"reply": reply, "recommendations": [], "selected_items": []}


def retrieval_node(state: AgentState) -> AgentState:
    context = state.get("context", {})
    query_text = state.get("conversation_text", "")
    is_compare = state.get("intent") == "compare"
    items = retrieve_items(context, query_text, top_k=20, is_compare=is_compare)
    return {"retrieved_items": items}


def rerank_node(state: AgentState) -> AgentState:
    items = state.get("retrieved_items", [])
    selected = items[:10]
    return {"selected_items": selected}


def comparison_node(state: AgentState) -> AgentState:
    context = state.get("context", {})
    query_text = state.get("conversation_text", "")
    items = retrieve_items(context, query_text, top_k=5, is_compare=True)
    return {"selected_items": items}


def explain_catalog_item_node(state: AgentState) -> AgentState:
    """Handle 'What is OPQ?'-style queries using catalog data only."""
    query = state.get("latest_user_message", "")
    item = find_catalog_item_by_alias(query)
    if item:
        items = [item]
        grounded = build_grounded_context(items)
        prompt = EXPLAIN_PROMPT.format(
            context=grounded, conversation=state.get("conversation_text", "")
        )
        reply = llm_client.generate_reply(prompt, "recommend")
        # Deterministic fallback if LLM fails
        if reply == FALLBACKS.get("recommend", ""):
            reply = (
                f"Based on the SHL catalog: **{item['name']}** is categorized under "
                f"{', '.join(item.get('keys', []))}. "
                f"{item.get('description', '')} "
                f"Duration: {item.get('duration', 'Not specified')}. "
                f"Job levels: {', '.join(item.get('job_levels', []))}."
            )
    else:
        reply = "I could not find that item in the SHL catalog. Please try a different assessment name."

    return {"reply": reply, "selected_items": [], "recommendations": []}


def llm_reply_node(state: AgentState) -> AgentState:
    intent = state.get("intent")
    if intent in ["clarify", "refuse", "done", "explain_catalog_item"]:
        return state

    items = state.get("selected_items", [])
    grounded_context = build_grounded_context(items)

    if intent == "compare":
        prompt = COMPARISON_PROMPT.format(context=grounded_context, conversation=state.get("conversation_text", ""))
    elif intent == "refine":
        prompt = REFINEMENT_PROMPT.format(context=grounded_context, conversation=state.get("conversation_text", ""))
    else:
        prompt = RECOMMENDATION_PROMPT.format(context=grounded_context, conversation=state.get("conversation_text", ""))

    reply = llm_client.generate_reply(prompt, intent)

    # Comparison deterministic fallback
    if intent == "compare" and items:
        generic_fallback = FALLBACKS.get("compare", "")
        reply_lower = reply.lower()
        item_mentioned = any(
            token in reply_lower
            for item in items
            for token in item.get("name", "").lower().split()
            if len(token) > 3
        )
        if reply == generic_fallback or not item_mentioned:
            reply = build_deterministic_comparison_reply(items)

    return {"reply": reply, "grounded_context": grounded_context}


def response_validation_node(state: AgentState) -> AgentState:
    intent = state.get("intent")
    if intent == "refuse" or state.get("refused"):
        reply = "I can only help with SHL assessment recommendations and comparisons based on the SHL catalog."
    elif intent == "done":
        reply = "You're welcome! Let me know if you need any more SHL assessment recommendations."
    else:
        reply = state.get("reply", "")

    return {"reply": reply}


# ── Routing ────────────────────────────────────────────────────────────────────

def route_after_guardrail(state: AgentState) -> str:
    if state.get("refused"):
        return "refusal"
    return "intent"


def route_after_intent(state: AgentState) -> str:
    if state.get("refused"):
        return "refusal"
    intent = state.get("intent")
    if intent == "done":
        return "done"
    if intent == "explain_catalog_item":
        return "explain"
    return "context"


def route_after_context(state: AgentState) -> str:
    intent = state.get("intent")
    if intent == "clarify":
        return "clarify"
    if intent == "compare":
        return "compare"
    return "retrieve"


def route_after_confidence(state: AgentState) -> str:
    if state.get("refused"):
        return "refusal"
    return "retrieve_exec"


# ── Graph assembly ─────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    workflow = StateGraph(AgentState)

    workflow.add_node("validate_input", validate_input_node)
    workflow.add_node("guardrail", guardrail_node)
    workflow.add_node("intent", intent_node)
    workflow.add_node("context", context_extraction_node)
    workflow.add_node("retrieval_confidence", retrieval_confidence_node)
    workflow.add_node("clarify", clarification_node)
    workflow.add_node("retrieve", retrieval_node)
    workflow.add_node("rerank", rerank_node)
    workflow.add_node("compare", comparison_node)
    workflow.add_node("explain_catalog_item", explain_catalog_item_node)
    workflow.add_node("llm_reply", llm_reply_node)
    workflow.add_node("response_validation", response_validation_node)

    workflow.set_entry_point("validate_input")
    workflow.add_edge("validate_input", "guardrail")

    workflow.add_conditional_edges(
        "guardrail",
        route_after_guardrail,
        {"refusal": "response_validation", "intent": "intent"},
    )

    workflow.add_conditional_edges(
        "intent",
        route_after_intent,
        {
            "refusal": "response_validation",
            "done": "response_validation",
            "explain": "explain_catalog_item",
            "context": "context",
        },
    )

    workflow.add_conditional_edges(
        "context",
        route_after_context,
        {"clarify": "clarify", "compare": "compare", "retrieve": "retrieval_confidence"},
    )

    workflow.add_conditional_edges(
        "retrieval_confidence",
        route_after_confidence,
        {"refusal": "response_validation", "retrieve_exec": "retrieve"},
    )

    workflow.add_edge("clarify", "response_validation")
    workflow.add_edge("explain_catalog_item", "response_validation")

    workflow.add_edge("retrieve", "rerank")
    workflow.add_edge("rerank", "llm_reply")
    workflow.add_edge("compare", "llm_reply")

    workflow.add_edge("llm_reply", "response_validation")
    workflow.add_edge("response_validation", END)

    return workflow.compile()


app_graph = build_graph()
