"""
YOUR DESIGN: STEP 5 Revisor Node Implementation

STEP 5 Logic:
- Receives: user_query + LLM parsed params + SERPSTACK results (from checkpoint memory)
- Uses: Gemini LLM to decide PATH A or PATH B
- PATH A: User wants best reviewed → Sort by rating/reviews → Return top result
- PATH B: User wants negotiation/pricing → Use Tavily API → Deep search → Negotiation workflow
"""

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from app.agent.state import AgentState
from app.config import settings
from pydantic import BaseModel, Field
from typing import Literal
import logging
import json

logger = logging.getLogger(__name__)


# ==========================================
# HELPER: Get Groq LLM
# ==========================================

def get_llm(temperature: float = 0.3, use_key_2: bool = False):
    """
    Get configured Groq LLM
    
    Args:
        temperature: LLM temperature (0-1)
        use_key_2: If True, uses GROQ_API_KEY_2, else uses GROQ_API_KEY
    """
    api_key = settings.groq_api_key_2 if use_key_2 else settings.groq_api_key
    
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=temperature,
        api_key=api_key or settings.groq_api_key  # Fallback to main key
    )


# ==========================================
# STEP 5: REVISOR NODE (Decision Point)
# ==========================================

class PathDecision(BaseModel):
    """Gemini's decision on which path to follow"""
    path: Literal["path_a", "path_b"] = Field(
        description="path_a for simple best reviewed, path_b for negotiation"
    )
    show_all: bool = Field(
        description="True (boolean) if user wants to see ALL results, False (boolean) if only best/top one. Do not return a string."
    )
    reasoning: str = Field(
        description="Why this path was chosen"
    )
    confidence: float = Field(
        description="Confidence score 0.0 to 1.0 (number). Do not return a string."
    )


async def revisor_node(state: AgentState) -> dict:
    """
    STEP 5: Revisor Node - ALWAYS goes PATH A first
    
    NEW DESIGN:
    - ALWAYS return path_a to show results first
    - path_b ONLY when user clicks NEGOTIATE button (route forced externally)
    
    Analyzes:
    - user_query (original natural language)
    - serp_results (SERPSTACK data from STEP 4)
    """
    
    logger.info("="*50)
    logger.info("🔍 STEP 5: Revisor Node - Analyzing query intent...")
    logger.info(f"   User Query: {state.get('user_query', 'N/A')}")
    logger.info(f"   SERP Results Count: {len(state.get('serp_results', []))}")
    logger.info(f"   Current Route: {state.get('route', 'not set')}")
    logger.info("="*50)
    
    # Check if route is already forced (e.g. by NEGOTIATE button click)
    if state.get("route") == "path_b":
        logger.info("⏩ Revisor: Route FORCED to path_b (User clicked NEGOTIATE button)")
        return {
            "route": "path_b",
            "show_all": state.get("show_all", False),
            "current_step": "revisor",
            "messages": state["messages"] + [AIMessage(content="🤖 Revisor: Starting negotiation as requested.")]
        }

    # NEW DESIGN: ALWAYS go PATH A first to show results
    # User will click NEGOTIATE button to enter PATH B
    logger.info("✅ Revisor: Going PATH A (Show results first, user can click NEGOTIATE later)")
    
    # Get data from checkpoint memory
    user_query = state.get("user_query", "")
    user_intent = state.get("user_intent", "")
    serp_results = state.get("serp_results", [])
    parsed_params = state.get("parsed_params", {})
    
    # Determine if user wants ALL results or just the BEST one
    llm = get_llm(temperature=0.3, use_key_2=True)
    structured_llm = llm.with_structured_output(PathDecision)
    
    system_prompt = """You are a decision-making agent for a place discovery system.

Your job: Decide TWO things:
1. PATH: path_a (simple search) or path_b (negotiation/pricing)
2. SHOW_ALL: True (show all results) or False (show only best one)

IMPORTANT: Return raw JSON booleans (true/false) and numbers (0.8), NOT strings ("true"/"0.8").

PATH A Indicators:
- "best reviewed", "top rated", "highest rating", "most popular", "show me the best"
- Generic queries like "gyms in X", "restaurants in Y" (without "best")

PATH B Indicators:
- "negotiate", "price", "pricing", "cost", "fees"
- "budget", "cheap", "affordable"
- "contact", "call", "phone number"
- "deals", "discount"

SHOW_ALL Rules:
- show_all = True if user says: "gyms", "restaurants", "show all", "list", "display other", "2nd and 3rd positions", "top 3", "top 5", "top 10", "multiple"
- show_all = False if user says: "best", "top one", "the highest", "most reviewed" (UNLESS they specify a number > 1 like "top 3")

Examples:
- "gyms in Bangalore" → path_a, show_all=True (user wants to see ALL gyms)
- "best gym in Bangalore" → path_a, show_all=False (user wants THE BEST one)
- "top 3 gyms in Bangalore" → path_a, show_all=True (user wants a LIST)
- "gyms with pricing in Bangalore" → path_b, show_all=True (negotiation needed for all)
"""
    
    user_prompt = f"""
Original User Query: "{user_query}"
Parsed Intent: "{user_intent}"
Number of SERPSTACK Results: {len(serp_results)}

Sample Result (if available):
{serp_results[0] if serp_results else "No results"}

Decision: Which path should we follow?
"""
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    decision: PathDecision = structured_llm.invoke(messages)
    
    # OVERRIDE: Always go PATH A, but respect show_all decision
    logger.info(f"✅ Revisor Decision: PATH_A (forced) | Show All: {decision.show_all}")
    logger.info(f"💭 LLM Reasoning: {decision.reasoning}")
    logger.info(f"📊 Confidence: {decision.confidence:.2f}")
    logger.info(f"📝 Note: path_b only via NEGOTIATE button, not from query text")
    
    return {
        "route": "path_a",  # ALWAYS path_a first - user clicks NEGOTIATE for path_b
        "show_all": decision.show_all,  # Respect show_all decision from LLM
        "current_step": "revisor",
        "messages": state["messages"] + [
            AIMessage(content=f"🤖 Revisor Analysis:\n- Showing results (click NEGOTIATE on any place to start negotiation)\n- Show All Results: {decision.show_all}\n- Analysis: {decision.reasoning}")
        ]
    }


# ==========================================
# DECISION FUNCTION (for conditional edge)
# ==========================================

def decide_path(state: AgentState) -> Literal["path_a", "path_b"]:
    """
    Conditional edge function: Returns which path to follow
    """
    route = state.get("route", "path_a")
    logger.info(f"📍 Routing to: {route}")
    return route


# ==========================================
# PATH A: SIMPLE BEST REVIEWED
# ==========================================

async def simple_best_reviewed_node(state: AgentState) -> dict:
    """
    PATH A: Simple Best Reviewed Analysis
    
    Process:
    1. Get SERPSTACK results from state
    2. Check if user wants ALL results or just BEST one (from revisor decision)
    3. If show_all=True: Return ALL sorted by rating
    4. If show_all=False: Use Tavily for deep analysis, return top 1
    5. END workflow
    """
    
    logger.info("⭐ PATH A: Analyzing places...")
    
    serp_results = state.get("serp_results", [])
    show_all = state.get("show_all", False)  # Get revisor's decision
    
    if not serp_results:
        return {
            "current_step": "simple_best_reviewed",
            "is_complete": True,
            "recommendations": [],
            "messages": state["messages"] + [
                AIMessage(content="❌ No places found in SERPSTACK results.")
            ]
        }
    
    # Sort by rating (primary) and reviews_count (secondary)
    sorted_places = sorted(
        serp_results,
        key=lambda x: (x.get("rating", 0), x.get("reviews_count", 0)),
        reverse=True
    )
    
    # DECISION POINT: Show all or just best?
    # Force show_all if user explicitly asked for "top X" or "list" in query
    user_query_lower = state.get("user_query", "").lower()
    logger.info(f"DEBUG: user_query='{user_query_lower}'")
    force_show_all = any(x in user_query_lower for x in ["top 3", "top 5", "top 10", "list", "all", "multiple"])
    logger.info(f"DEBUG: show_all={show_all}, force_show_all={force_show_all}")
    
    if show_all or force_show_all:
        logger.info(f"📋 Showing ALL {len(sorted_places)} results (user wants complete list)")
        
        # Return all results sorted by rating
        result_text = f"✅ **ALL PLACES IN YOUR AREA** (Sorted by Rating):\n\n"
        
        for idx, place in enumerate(sorted_places, 1):
            reviews_count = place.get('reviews_count', 0)
            # Safely convert to int for formatting
            try:
                reviews_count = int(reviews_count) if reviews_count else 0
            except (ValueError, TypeError):
                reviews_count = 0
            
            result_text += f"""
**{idx}. {place.get('name', 'Unknown')}**
⭐ Rating: {place.get('rating', 'N/A')}/5.0 ({reviews_count:,} reviews)
📍 Address: {place.get('address', 'N/A')}
📞 Phone: {place.get('phone', 'Not available')}
---
"""
        
        return {
            "current_step": "simple_best_reviewed",
            "is_complete": True,
            "recommendations": sorted_places,  # Return all
            "messages": state["messages"] + [
                AIMessage(content=result_text)
            ]
        }
    
    else:
        logger.info(f"🏆 Finding THE BEST option (user wants top recommendation)")
        
        # Get top 3 candidates for deep analysis
        top_candidates = sorted_places[:min(3, len(sorted_places))]
        
        logger.info(f"📊 Top candidates for deep analysis: {[p.get('name') for p in top_candidates]}")
        
        # We return these 3 candidates so the next node (review_extraction) can fetch reviews for ALL of them
        return {
            "current_step": "simple_best_reviewed",
            "is_complete": False, # Not complete yet
            "recommendations": top_candidates,  # Return TOP 3
            "messages": state["messages"] + [
                AIMessage(content=f"Found {len(top_candidates)} top candidates. Fetching detailed reviews to pick the winner...")
            ]
        }


# ==========================================
# PATH B: NEGOTIATION WORKFLOW
# ==========================================

async def negotiation_path_node(state: AgentState) -> dict:
    """
    PATH B: Negotiation Workflow Initialization
    
    1. Identify target place (from recommendations set by /agent/negotiate/start)
    2. Initialize negotiation state
    3. Route to strategy formulation
    """
    logger.info("="*60)
    logger.info("🚀 PATH B: NEGOTIATION WORKFLOW STARTED")
    logger.info("="*60)
    
    # Get target place from recommendations (set by /agent/negotiate/start endpoint)
    recommendations = state.get("recommendations", [])
    serp_results = state.get("serp_results", [])
    
    # Use recommendation if available, otherwise first serp result
    if recommendations:
        target_place = recommendations[0]
        logger.info(f"📍 Target Place (from button click): {target_place.get('name')}")
    elif serp_results:
        target_place = serp_results[0]
        logger.info(f"📍 Target Place (fallback to first): {target_place.get('name')}")
    else:
        target_place = {"name": "Unknown Place", "address": "Unknown"}
        logger.warning("⚠️ No target place found!")
    
    logger.info(f"   Phone: {target_place.get('phone', 'N/A')}")
    logger.info(f"   Address: {target_place.get('address', 'N/A')}")
    logger.info(f"   Target Price: {state.get('target_price', 'Not specified')}")
    logger.info(f"   User Goal: {state.get('user_intent', 'Not specified')}")
    
    # Preserve existing history if available (e.g. when resuming)
    existing_history = state.get("negotiation_history", [])
    logger.info(f"📜 Existing negotiation history: {len(existing_history)} messages")
    
    return {
        "current_step": "negotiation_init",
        "negotiation_active": True,
        "negotiation_status": "ongoing",
        "negotiation_history": existing_history if existing_history else [],
        "shop_persona": "friendly but firm",
        "recommendations": [target_place],
        "messages": state["messages"] + [
            AIMessage(content=f"🎯 Starting negotiation for: {target_place.get('name', 'Unknown Place')}\n📞 Phone: {target_place.get('phone', 'N/A')}")
        ]
    }


# ==========================================
# STEP 8: STRATEGY NODE
# ==========================================

async def strategy_node(state: AgentState) -> dict:
    """
    Formulates a negotiation strategy and drafts a message.
    Uses Llama 3.3 70B and leverages reviews for leverage.
    """
    logger.info("="*60)
    logger.info("🧠 STRATEGY NODE: Formulating negotiation plan...")
    logger.info("="*60)
    
    target_place = state.get("recommendations", [{}])[0]
    user_intent = state.get("user_intent", "")
    negotiation_history = state.get("negotiation_history", [])
    target_price = state.get("target_price")
    
    logger.info(f"   Target: {target_place.get('name', 'Unknown')}")
    logger.info(f"   User Goal: {user_intent}")
    logger.info(f"   Target Price: {target_price}")
    logger.info(f"   History Length: {len(negotiation_history)} messages")
    
    # Get reviews for context
    reviews_data = target_place.get("reviews_data", [])
    reviews_context = "\n".join(reviews_data) if reviews_data else "No specific reviews available."
    
    logger.info(f"   Reviews for leverage: {len(reviews_data)} reviews")
    
    # Use Llama 3.3 (via Groq)
    llm = get_llm(temperature=0.6)
    
    system_prompt = f"""You are a skilled negotiator acting on behalf of a user.
Target Place: {target_place.get('name')}
Address: {target_place.get('address')}
User Goal: {user_intent}
Target Price: {target_price if target_price else "Get the best deal possible"}

Reviews Context:
{reviews_context}

Your Strategy:
1. Analyze the negotiation history.
2. Use the reviews as leverage (e.g., if reviews mention "crowded", ask for a discount; if "great equipment", acknowledge value but push for price).
3. Be persuasive, professional, but firm.
4. Draft the NEXT message to send to the shopkeeper via SMS.
5. Keep the message concise (under 160 chars preferred, max 2 sentences).
"""

    history_text = "\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in negotiation_history])
    
    user_prompt = f"""
Negotiation History:
{history_text if history_text else "No history yet (Start of conversation)"}

Draft the next message.
"""

    class StrategyOutput(BaseModel):
        thought_process: str = Field(description="Your reasoning, including how you used reviews")
        draft_message: str = Field(description="The exact SMS message to send")

    structured_llm = llm.with_structured_output(StrategyOutput)
    strategy = structured_llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])
    
    logger.info(f"📝 Draft Message: \"{strategy.draft_message}\"")
    logger.info(f"💭 Thought Process: {strategy.thought_process[:100]}...")
    
    return {
        "current_step": "strategy",
        "planned_message": strategy.draft_message,
        "messages": state["messages"] + [
            AIMessage(content=f"📝 Strategy: {strategy.thought_process}\n\n💬 Draft Message: \"{strategy.draft_message}\"")
        ]
    }


# ==========================================
# STEP 9: HUMAN REVIEW NODE (HITL)
# ==========================================

async def human_review_node(state: AgentState) -> dict:
    """
    Executes the approved message sending via MessagingService.
    """
    logger.info("="*60)
    logger.info("⏸️ HUMAN REVIEW NODE")
    logger.info("="*60)
    logger.info(f"   Human Approved: {state.get('human_approved', False)}")
    logger.info(f"   Planned Message: {state.get('planned_message', 'N/A')}")
    
    # Check if approved
    if state.get("human_approved"):
        logger.info("✅ APPROVED by User - Sending SMS...")
        
        planned_msg = state.get("planned_message")
        target_place = state.get("recommendations", [{}])[0]
        place_phone = target_place.get("phone", "") if isinstance(target_place, dict) else ""
        place_name = target_place.get("name", "Unknown Place") if isinstance(target_place, dict) else "Unknown Place"

        # Normalize phone numbers to keep digits (and leading +)
        def normalize_phone(num: str) -> str:
            if not num:
                return ""
            num = num.strip()
            if num.startswith("+"):
                return "+" + re.sub(r"[^0-9]", "", num)
            return re.sub(r"[^0-9]", "", num)
        
        # Update history
        new_history_entry = {"role": "agent", "content": planned_msg}
        
        # Send Real SMS via MessagingService
        from app.messaging.service import MessagingService
        from app.config import settings
        import re
        
        # Prefer the shop's phone from SerpStack; fallback to configured agent recipient
        target_number = normalize_phone(place_phone) or normalize_phone(settings.default_target_number) or normalize_phone(settings.twilio_target_number)

        if not target_number:
            log_msg = "❌ No target phone available (missing SerpStack phone and default recipient)."
            logger.error(log_msg)
            return {
                "current_step": "human_review",
                "messages": state["messages"] + [AIMessage(content=log_msg)],
                "human_approved": False
            }
        
        logger.info("="*40)
        logger.info("📱 SENDING SMS")
        logger.info(f"   To: {target_number}")
        logger.info(f"   Message: {planned_msg}")
        logger.info("="*40)
        
        sms_sent = MessagingService.send_message(target_number, planned_msg)
        
        if sms_sent:
            log_msg = f"✅ SMS SENT to {place_name} ({target_number}): \"{planned_msg}\""
            logger.info(log_msg)
        else:
            log_msg = f"❌ FAILED to send SMS to {place_name} ({target_number}). Check logs."
            logger.error(log_msg)

        return {
            "current_step": "human_review",
            "negotiation_history": state["negotiation_history"] + [new_history_entry],
            "human_approved": False, # Reset for next turn
            "messages": state["messages"] + [AIMessage(content=log_msg)]
        }
    else:
        # This path is taken when the graph resumes but approval wasn't explicitly set to True
        # or if we are just entering the node to pause.
        logger.info("⏳ Waiting for human approval...")
        return {"current_step": "human_review"}


# ==========================================
# STEP 10: SHOP SIMULATION NODE
# ==========================================

async def shop_simulation_node(state: AgentState) -> dict:
    """
    Simulates the shopkeeper's response.
    """
    logger.info("🎭 Shop Simulation Node: Generating response...")
    
    negotiation_history = state.get("negotiation_history", [])
    target_place = state.get("recommendations", [{}])[0]
    persona = state.get("shop_persona", "friendly")
    
    # Get the last message from agent
    last_agent_msg = negotiation_history[-1]["content"] if negotiation_history else ""
    
    llm = get_llm(temperature=0.6)
    
    system_prompt = f"""You are a shopkeeper at {target_place.get('name')}.
Location: {target_place.get('address')}
Persona: {persona}

Your goal: Maximize profit but don't lose the customer if the offer is reasonable.
Current Market Rates (Assumed): Gyms ~2000-5000/month.

Respond to the customer's query. Be realistic.
"""

    user_prompt = f"Customer says: \"{last_agent_msg}\""
    
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])
    
    shop_msg = response.content
    
    new_history_entry = {"role": "shop", "content": shop_msg}
    
    return {
        "current_step": "shop_simulation",
        "negotiation_history": state["negotiation_history"] + [new_history_entry],
        "messages": state["messages"] + [
            AIMessage(content=f"🏪 Shop: \"{shop_msg}\"")
        ]
    }


# ==========================================
# STEP 11: NEGOTIATION MANAGER NODE
# ==========================================

async def negotiation_manager_node(state: AgentState) -> dict:
    """
    Decides whether to continue negotiating, accept deal, or walk away.
    Also handles the 'Wait for Reply' state.
    """
    logger.info("⚖️ Negotiation Manager: Evaluating status...")
    
    negotiation_history = state.get("negotiation_history", [])
    
    # If history is empty, we are just starting -> Go to strategy
    if not negotiation_history:
        return {
            "current_step": "negotiation_manager",
            "negotiation_status": "continue"
        }
        
    last_msg = negotiation_history[-1]
    
    # If last message was from US (agent), we need to WAIT for a reply.
    if last_msg["role"] == "agent":
        logger.info("⏳ Last message was from Agent. Waiting for reply...")
        # In a real async graph, we might return a special status or just END.
        # Returning "end" here effectively pauses the graph until we resume it with a new message.
        return {
            "current_step": "negotiation_manager",
            "negotiation_status": "end", # End this run, wait for external trigger
            "messages": state["messages"] + [AIMessage(content="⏳ Waiting for reply from shopkeeper...")]
        }
    
    # If last message was from SHOP (reply received), we analyze and continue
    if last_msg["role"] == "shop":
        logger.info("📩 Reply received from Shop. Analyzing...")
        
        # Analyze if we should continue or stop
        last_shop_msg = last_msg["content"]
        llm = get_llm(temperature=0.1)
        
        class ManagerDecision(BaseModel):
            status: Literal["continue", "success", "failed"] = Field(description="Status of negotiation")
            reasoning: str = Field(description="Why this status?")
            
        structured_llm = llm.with_structured_output(ManagerDecision)
        
        decision = structured_llm.invoke([
            SystemMessage(content="Analyze the negotiation status. Did we reach a deal? Did they reject us? Should we keep trying?"),
            HumanMessage(content=f"Last Shop Message: \"{last_shop_msg}\"")
        ])
        
        return {
            "current_step": "negotiation_manager",
            "negotiation_status": decision.status,
            "messages": state["messages"] + [
                AIMessage(content=f"⚖️ Status: {decision.status.upper()} ({decision.reasoning})")
            ]
        }
        
    return {"negotiation_status": "continue"}



# ==========================================
# STEP 6: REVIEW EXTRACTION NODE
# ==========================================

async def review_extraction_node(state: AgentState) -> dict:
    """
    STEP 6: Review Extraction Node
    
    Fetches detailed reviews for the recommended places using WebScraping.AI.
    """
    logger.info("🔍 STEP 6: Review Extraction Node")
    
    recommendations = state.get("recommendations", [])
    if not recommendations:
        logger.info("No recommendations to fetch reviews for.")
        return {
            "current_step": "review_extraction",
            "messages": state["messages"] + [AIMessage(content="No recommendations available for review extraction.")]
        }
        
    from app.agent.tools import fetch_reviews_webscraping_ai
    
    reviews_map = {}
    updated_recommendations = []
    
    for place in recommendations:
        name = place.get("name", "Unknown")
        address = place.get("address", "")
        
        # Construct query for WebScraping.AI
        query = f"{name} {address}".strip()
        
        logger.info(f"Fetching reviews for: {name}")
        
        # Call the tool directly (it returns a list of strings)
        # Note: In a real async environment, we might want to run these in parallel
        try:
            # The tool is decorated, so we invoke it
            reviews = fetch_reviews_webscraping_ai.invoke({"query": query, "limit": 3})
        except Exception as e:
            logger.error(f"Failed to invoke fetch_reviews_webscraping_ai: {e}")
            reviews = ["Could not fetch reviews."]
            
        reviews_map[name] = reviews
        
        # Enrich recommendation with reviews
        place["reviews_data"] = reviews
        updated_recommendations.append(place)
        
    return {
        "reviews": reviews_map,
        "recommendations": updated_recommendations,
        "current_step": "review_extraction",
        "messages": state["messages"] + [AIMessage(content=f"Retrieved reviews for {len(updated_recommendations)} places.")]
    }


# ==========================================
# STEP 7: ANALYZE REVIEWS NODE
# ==========================================

async def analyze_reviews_node(state: AgentState) -> dict:
    """
    STEP 7: Analyze Reviews Node
    
    Uses LLM to analyze the fetched reviews for the top candidates and pick the winner.
    """
    logger.info("🧠 STEP 7: Analyze Reviews Node")
    
    recommendations = state.get("recommendations", [])
    user_query = state.get("user_query", "")
    
    if not recommendations:
        return {
            "is_complete": True,
            "current_step": "analyze_reviews",
            "messages": state["messages"] + [AIMessage(content="No recommendations to analyze.")],
            "recommendations": []
        }
        
    # Prepare data for LLM
    candidates_data = []
    for place in recommendations:
        candidates_data.append({
            "name": place.get("name"),
            "rating": place.get("rating"),
            "reviews_count": place.get("reviews_count"),
            "reviews": place.get("reviews_data", [])
        })
        
    llm = get_llm(temperature=0.2)
    
    system_prompt = """You are an expert local guide. Your task is to pick the SINGLE BEST place from the provided candidates based on the user's request and the actual content of the reviews.

Analyze the reviews for:
1. Sentiment (are people happy?)
2. Specific mentions relevant to the user's query (e.g., "clean", "friendly", "equipment")
3. Red flags (e.g., "hidden fees", "rude staff")

Output Format:
Return the name of the winning place exactly as it appears in the input.
Then, provide a short "Why this is the winner" explanation.
"""

    user_prompt = f"""
User Query: "{user_query}"

Candidates and their Reviews:
{json.dumps(candidates_data, indent=2)}

Who is the winner?
"""

    # We use a structured output or just parse the text. Let's use text for flexibility and parse the name.
    # Actually, let's use structured output for the name to be safe.
    
    class WinnerSelection(BaseModel):
        winner_name: str = Field(description="The exact name of the winning place")
        explanation: str = Field(description="Why this place was chosen based on reviews")
        key_pros: list[str] = Field(description="List of 3 key positive points from reviews")
        key_cons: list[str] = Field(description="List of 1-2 negative points or warnings (if any)")

    structured_llm = llm.with_structured_output(WinnerSelection)
    
    try:
        decision = structured_llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])
        
        winner_name = decision.winner_name
        explanation = decision.explanation
        
        # Find the winner object
        winner = None
        for place in recommendations:
            if place.get("name") == winner_name:
                winner = place
                break
        
        if not winner:
            # Fallback fuzzy match
            for place in recommendations:
                if place.get("name") in winner_name or winner_name in place.get("name"):
                    winner = place
                    break
        
        if not winner:
            winner = recommendations[0] # Fallback to first
        
        # Safely extract rating value
        winner_rating = winner.get('rating')
        if isinstance(winner_rating, dict):
            winner_rating = winner_rating.get('value') or winner_rating.get('rating') or 'N/A'
        winner_rating = winner_rating or 'N/A'
        
        # Safely extract reviews count
        reviews_count = winner.get('reviews_count', 0)
        try:
            reviews_count = int(reviews_count) if reviews_count else 0
        except (ValueError, TypeError):
            reviews_count = 0
            
        # Format the final output
        result_text = f"""
🏆 **WINNER SELECTED:** {winner.get('name')}

⭐ **Rating:** {winner_rating}/5.0 ({reviews_count:,} reviews)
📍 **Address:** {winner.get('address')}

**Why it won:**
{explanation}

**Key Highlights:**
{chr(10).join([f"- {pro}" for pro in decision.key_pros])}

**Things to Note:**
{chr(10).join([f"- {con}" for con in decision.key_cons])}
"""
        
        return {
            "current_step": "analyze_reviews",
            "is_complete": True,
            "recommendations": [winner], # Return just the winner
            "messages": state["messages"] + [AIMessage(content=result_text)]
        }
        
    except Exception as e:
        logger.error(f"Error in analyze_reviews_node: {e}")
        # Fallback
        return {
            "current_step": "analyze_reviews",
            "is_complete": True,
            "recommendations": [recommendations[0]],
            "messages": state["messages"] + [AIMessage(content="Could not analyze reviews in detail. Returning top rated option.")]
        }
