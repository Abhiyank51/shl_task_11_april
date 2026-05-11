from typing import List, Dict, Any
from app.graph import app_graph
from app.response_utils import validate_and_format_response

def run_agent(messages: List[Dict[str, str]]) -> dict:
    """
    Run the LangGraph workflow given the conversation messages.
    """
    # Initialize state
    initial_state = {
        "messages": messages,
        "conversation_text": "",
        "latest_user_message": "",
        "intent": "",
        "context": {},
        "retrieved_items": [],
        "selected_items": [],
        "grounded_context": "",
        "reply": "",
        "recommendations": [],
        "refused": False,
        "end_of_conversation": False,
        "errors": []
    }
    
    # LangGraph returns the final state
    final_state = app_graph.invoke(initial_state)
    
    # Extract data for the final response
    reply = final_state.get("reply", "")
    items = final_state.get("selected_items", [])
    intent = final_state.get("intent", "recommend")
    end_of_conv = final_state.get("end_of_conversation", False)
    
    # Assemble and validate the final JSON dictionary using the response_utils
    return validate_and_format_response(reply, items, intent, end_of_conv)
