from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    """
    State schema for the Negotiator Agent with Reflexion pattern.
    
    This state is shared across all nodes and persisted via checkpointer.
    Uses Annotated with add_messages to append to message history.
    """
    
    # Conversation history (using add_messages reducer)
    messages: Annotated[list[BaseMessage], add_messages]
    
    # Initial search context
    city: str
    place_type: str
    user_intent: str
    user_query: str  # Original user query
    budget: float | None
    
    # SERP & External Data
    serp_results: list[dict]  # Raw SERP API results
    tavily_reviews: list[dict]  # Tavily search results for reviews
    reviews: dict[str, list[str]] # Detailed reviews for specific shops
    
    # Human-in-the-Loop
    human_approved: bool
    human_notes: str | None
    
    # Agent Workflow State
    route: str  # "negotiation", "info_only", "comparison"
    show_all: bool  # True if user wants all results, False if only best one
    current_step: str  # Current node in workflow
    
    # Analysis & Reflexion
    initial_analysis: list[dict]  # First pass analysis
    shop_responses: list[dict]  # Simulated shop communications
    refined_analysis: list[dict]  # After reflexion
    iteration: int  # Reflexion loop counter
    
    # Negotiation State
    negotiation_active: bool
    negotiation_history: list[dict]  # Chat history with shop: [{"role": "agent", "content": "..."}]
    negotiation_status: str  # "ongoing", "success", "failed", "rejected"
    planned_message: str  # Draft message for HITL review
    shop_persona: str  # Simulated shopkeeper personality (e.g. "stubborn", "friendly")
    target_price: float | None # User's desired price
    negotiation_start_time: int | None  # Unix timestamp when negotiation started (for filtering old messages)
    
    # Final Output
    recommendations: list[dict] | None
    is_complete: bool
    
    # Metadata
    thread_id: str
