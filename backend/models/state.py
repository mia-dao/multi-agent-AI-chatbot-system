"""
State management for the multi-agent chatbot system
"""
from typing import TypedDict, List, Optional, Annotated
from operator import add


class Message(TypedDict):
    role: str
    content: str
    timestamp: Optional[str]


class AgentState(TypedDict):
    """State shared across all agents in the workflow"""
    messages: Annotated[List[Message], add]
    user_input: str
    intent: Optional[str]
    confidence: Optional[float]
    retrieved_info: Optional[str]
    response: Optional[str]
    quality_score: Optional[float]
    needs_refinement: bool
    iteration_count: int
    context: Optional[dict]