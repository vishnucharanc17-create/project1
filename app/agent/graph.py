from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing import Literal
from app.agent.state import AgentState
from app.agent.nodes import (
    revisor_node,
    simple_best_reviewed_node,
    negotiation_path_node,
    review_extraction_node,
    analyze_reviews_node,
    decide_path,
    strategy_node,
    human_review_node,
    negotiation_manager_node
)
from app.config import settings


def create_negotiator_agent():
    """
    YOUR DESIGN: STEP 5 Revisor Node Implementation
    
    Workflow:
    STEP 4 (SERPSTACK results ready) →
    STEP 5 (Revisor Node - Gemini decides path) →
        PATH A: Simple Best Reviewed (sort by rating/reviews) → END
        PATH B: Negotiation Required (Tavily search + negotiation) → [Under construction]
    Returns:
        Compiled LangGraph application with MemorySaver persistence
    """
    
    # Create StateGraph
    workflow = StateGraph(AgentState)
    
    # ==========================================
    # ADD NODES 
    # ==========================================
    
    # STEP 5: Revisor Node - Gemini decides which path
    workflow.add_node("revisor", revisor_node)
    
    # PATH A: Simple best reviewed analysis
    workflow.add_node("simple_best_reviewed", simple_best_reviewed_node)
    
    # STEP 6: Review Extraction
    workflow.add_node("review_extraction", review_extraction_node)
    
    # STEP 7: Analyze Reviews
    workflow.add_node("analyze_reviews", analyze_reviews_node)
    
    # PATH B: Negotiation workflow (Real SMS - no simulation)
    workflow.add_node("negotiation_path", negotiation_path_node)
    workflow.add_node("strategy", strategy_node)
    workflow.add_node("human_review", human_review_node)
    workflow.add_node("negotiation_manager", negotiation_manager_node)
    # Note: shop_simulation removed - we use real SMS
    
    # ==========================================
    # SET ENTRY POINT
    # ==========================================
    
    workflow.set_entry_point("revisor")
    
    # ==========================================
    # ADD EDGES (YOUR DESIGN)
    # ==========================================
    
    # Revisor → Conditional (PATH A or PATH B)
    workflow.add_conditional_edges(
        "revisor",
        decide_path,
        {
            "path_a": "simple_best_reviewed",
            "path_b": "negotiation_path"
        }
    )
    
    # PATH A: Simple Best Reviewed → Review Extraction
    workflow.add_edge("simple_best_reviewed", "review_extraction")
    
    # Review Extraction → Analyze Reviews
    workflow.add_edge("review_extraction", "analyze_reviews")
    
    # Analyze Reviews → END
    workflow.add_edge("analyze_reviews", END)
    
    # PATH B: Negotiation Workflow Edges
    # Flow: negotiation_path → negotiation_manager → strategy → human_review → negotiation_manager (loop)
    
    # Init -> Manager (to check history/status)
    workflow.add_edge("negotiation_path", "negotiation_manager")
    
    # Strategy -> Human Review (HITL - pauses here for approval)
    workflow.add_edge("strategy", "human_review")
    
    # Human Review -> Manager (after SMS sent, check for reply)
    workflow.add_edge("human_review", "negotiation_manager")
    
    # Manager -> Conditional (Continue to next strategy or End)
    def check_negotiation_status(state: AgentState) -> Literal["strategy", "end"]:
        status = state.get("negotiation_status", "continue")
        if status == "continue":
            return "strategy"
        return "end"
        
    workflow.add_conditional_edges(
        "negotiation_manager",
        check_negotiation_status,
        {
            "strategy": "strategy",
            "end": END
        }
    )
    
    # ==========================================
    # CONFIGURE PERSISTENCE (Memory)
    # ==========================================
    
    # Use MemorySaver - All state persisted automatically
    checkpointer = MemorySaver()
    
    # ==========================================
    # COMPILE GRAPH
    # ==========================================
    
    app = workflow.compile(
        checkpointer=checkpointer,
        interrupt_before=["human_review"] # PAUSE before human review node runs
    )
    
    return app


# ==========================================
# GRAPH VISUALIZATION (Optional)
# ==========================================

def visualize_graph():
    """
    Returns ASCII representation of the agent workflow.
    Useful for debugging and documentation.
    """
    return """
    ┌─────────────────────────────────────────────────┐
    │         NEGOTIATOR AGENT WORKFLOW               │
    │         (Reflexion + HITL Pattern)              │
    └─────────────────────────────────────────────────┘
    
    START
      ↓
    ┌─────────────┐
    │   Router    │  ← Classify intent (negotiation/info/comparison)
    └──────┬──────┘
           ↓
    ┌─────────────┐
    │ Fetch Data  │  ← Get Tavily reviews for top places
    └──────┬──────┘
           ↓
    ┌─────────────┐
    │Human Review │  ← 🔴 HITL: User approves/modifies data
    │   (HITL)    │
    └──────┬──────┘
           ↓
    ┌─────────────┐
    │  Analyze    │  ← Initial analysis of all places
    └──────┬──────┘
           ↓
         ┌─┴─┐
         │ ? │  Needs negotiation?
         └─┬─┘
      ┌────┴────┐
      │         │
      NO       YES
      │         ↓
      │    ┌─────────────┐
      │    │  Negotiate  │  ← Simulate shop contact
      │    └──────┬──────┘
      │           ↓
      │         ┌─┴─┐
      │         │ ? │  Need refinement?
      │         └─┬─┘
      │      ┌────┴────┐
      │      │         │
      │      NO       YES
      │      │         ↓
      │      │    ┌─────────────┐
      │      │    │   Refine    │  ← Reflexion: Self-critique
      │      │    └──────┬──────┘
      │      │           │
      └──────┴───────────┘
             ↓
    ┌─────────────┐
    │ Recommend   │  ← Final ranked recommendations
    └──────┬──────┘
           ↓
         END
    
    🔑 KEY FEATURES:
    - Router: Determines workflow path
    - HITL: Human validates data before processing
    - Reflexion: Iterative improvement (max 3 loops)
    - Memory: SQLite persistence across sessions
    - Streaming: Real-time token output
    """


# ==========================================
# HELPER FUNCTIONS
# ==========================================

def get_thread_config(thread_id: str) -> dict:
    """
    Generate config for thread-specific execution.
    Required for memory persistence.
    """
    return {
        "configurable": {
            "thread_id": thread_id
        },
        "recursion_limit": settings.recursion_limit
    }


async def stream_agent_response(app, input_state: dict, thread_id: str):
    """
    Stream agent execution with real-time updates.
    
    Args:
        app: Compiled LangGraph application
        input_state: Initial state dict
        thread_id: Conversation thread ID
    
    Yields:
        Dict with node updates and messages
    """
    config = get_thread_config(thread_id)
    
    async for event in app.astream(input_state, config):
        # Each event is {node_name: output}
        node_name = list(event.keys())[0]
        node_output = event[node_name]
        
        yield {
            "node": node_name,
            "output": node_output,
            "timestamp": None  # Can add timestamp if needed
        }


# Initialize the agent
negotiator_agent = create_negotiator_agent()
