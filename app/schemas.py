from typing import List, Dict, Any, TypedDict, Literal
from pydantic import BaseModel, Field

class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]

class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str

class ChatResponse(BaseModel):
    reply: str
    recommendations: List[Recommendation]
    end_of_conversation: bool

class AgentState(TypedDict):
    messages: List[Dict[str, str]]
    conversation_text: str
    latest_user_message: str
    intent: str
    context: Dict[str, Any]
    retrieved_items: List[Dict[str, Any]]
    selected_items: List[Dict[str, Any]]
    grounded_context: str
    reply: str
    recommendations: List[Dict[str, str]]
    refused: bool
    end_of_conversation: bool
    errors: List[str]
