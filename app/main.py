from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from typing import Optional
from app.models import (
    SearchRequest,
    SearchResult,
    AgentActivationRequest,
    AgentChatRequest,
    HumanApprovalRequest,
    AgentResponse,
    PlaceType,
    AgentResponse,
    PlaceType,
    InitialQueryRequest,
    NegotiationStartRequest
)
from app.agent.graph import negotiator_agent, get_thread_config, visualize_graph
from app.agent.tools import search_places
from app.config import settings
import uuid
import json
import logging
import re

# Configure logging
logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)

from fastapi.staticfiles import StaticFiles

# ==========================================
# INITIALIZE FASTAPI
# ==========================================

app = FastAPI(
    title="Negotiator Agent API",
    description="AI Agent for place discovery and negotiation with HITL support",
    version="1.0.0"
)

# Mount static files
app.mount("/app/static", StaticFiles(directory="app/static"), name="static")
app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# HEALTH CHECK
# ==========================================

@app.get("/")
async def root():
    """Health check and API info"""
    return {
        "status": "healthy",
        "service": "Negotiator Agent API",
        "version": "1.0.0",
        "environment": settings.environment,
        "workflow": visualize_graph()
    }


@app.get("/dashboard.html")
async def dashboard():
    """Serve dashboard HTML"""
    from fastapi.responses import FileResponse
    return FileResponse("frontend/dashboard.html")


# ==========================================
# ENDPOINT 1: INITIAL SEARCH (SERP API)
# ==========================================

@app.post("/search", response_model=list[SearchResult])
async def search_endpoint(request: SearchRequest):
    """
    Initial place search using SERP API.
    Returns raw results before agent mode.
    
    Flow: User searches → Gets results → Can activate agent mode
    """
    try:
        logger.info(f"Search request: {request.city}, {request.place_type}")
        
        # Use SERP tool
        results = search_places.invoke({
            "city": request.city,
            "place_type": request.place_type.value,
            "query": request.query
        })
        
        if not results:
            raise HTTPException(status_code=404, detail="No places found")
        
        # Convert to SearchResult models
        search_results = [
            SearchResult(
                name=place.get("name", "Unknown"),
                address=place.get("address", ""),
                phone=place.get("phone"),
                rating=place.get("rating"),
                price_level=place.get("price_level"),
                type=place.get("type", request.place_type.value)
            )
            for place in results
            if "error" not in place
        ]
        
        logger.info(f"Found {len(search_results)} places")
        return search_results
    
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ==========================================
# ENDPOINT 2: START AGENT (YOUR DESIGN: STEP 1-2-3)
# ==========================================

@app.post("/agent/start")
async def start_agent(request: InitialQueryRequest):
    """
    STEP 1-3 Implementation from MY_DESIGN_README
    
    STEP 1: User submits natural language query
    STEP 2: Responder LLM analyzes intent
    STEP 3: Shows "Searching SERPSTACK API with params"
    """
    try:
        from langchain_groq import ChatGroq
        from pydantic import BaseModel, Field
        
        # STEP 1: Receive user query
        user_query = request.user_query
        logger.info(f"STEP 1: User query received: {user_query}")
        
        # STEP 2: Responder LLM - Parse query into structured format
        class ParsedQuery(BaseModel):
            """Structured query extraction"""
            city: str = Field(description="City name extracted from query")
            area: str = Field(description="Area/locality within city, if mentioned")
            place_type: str = Field(description="Type of place (gym, cafe, restaurant, etc.)")
            intent: str = Field(description="What user wants: best reviewed, negotiation, pricing info, etc.")
            budget: Optional[str] = Field(None, description="Budget mentioned, if any")
        
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            api_key=settings.groq_api_key  # GROQ_API_KEY
        )
        
        structured_llm = llm.with_structured_output(ParsedQuery)
        
        responder_prompt = f"""
        You are a query parser for a place discovery system.
        
        User query: "{user_query}"
        
        Extract:
        - city: The city name
        - area: Specific area/locality if mentioned (otherwise use city name)
        - place_type: Type of place (gym, cafe, restaurant, hotel, etc.)
        - intent: What the user wants (best reviewed, negotiation, pricing, etc.)
        - budget: Any budget mentioned (extract number if present)
        
        Examples:
        "I need gyms in koramangala in bangalore" → city: bangalore, area: koramangala, type: gym
        "Find cafes in MG Road" → city: MG Road, area: MG Road, type: cafe
        """
        
        logger.info("STEP 2: Responder LLM analyzing query...")
        parsed_query = structured_llm.invoke(responder_prompt)
        
        logger.info(f"STEP 2: Parsed - City: {parsed_query.city}, Area: {parsed_query.area}, Type: {parsed_query.place_type}")
        
        # Generate thread_id
        thread_id = str(uuid.uuid4()) #here we are saving to memory using thread_id
        
        # STEP 3: Prepare for SERPSTACK API call
        # Combine city + area for better search results
        search_location = f"{parsed_query.area} {parsed_query.city}" if parsed_query.area and parsed_query.area != parsed_query.city else parsed_query.city
        
        # Construct a clean search query
        final_query = f"{parsed_query.place_type} in {search_location}"
        
        search_params = {
            "city": search_location,
            "place_type": parsed_query.place_type,
            "query": final_query
        }
        
        logger.info(f"STEP 3: Searching SERPSTACK API with: Query='{final_query}'")
        
        # Call SERPSTACK (STEP 4 will execute in streaming)
        results = search_places.invoke(search_params)
        
        if not results or any("error" in str(r) for r in results):
            logger.warning("SERPSTACK returned no results or error")
            results = []
        
        # Initialize agent state
        initial_state = {
            "messages": [HumanMessage(content=user_query)],
            "city": parsed_query.city,
            "place_type": parsed_query.place_type,
            "user_intent": parsed_query.intent,
            "budget": None,
            "serp_results": results,
            "thread_id": thread_id,
            "iteration": 0,
            "is_complete": False,
            "show_all": False,  # Will be set by Revisor node
            "user_query": user_query,  # Store original query
            "parsed_params": {
                "city": parsed_query.city,
                "area": parsed_query.area,
                "type": parsed_query.place_type
            }
        }
        
        config = get_thread_config(thread_id)
        
        # Save initial state to memory (do NOT execute agent yet)
        # Frontend will immediately call /stream which will start execution
        logger.info(f"Saving initial state for thread {thread_id}")
        await negotiator_agent.aupdate_state(config, initial_state)
        
        return {
            "thread_id": thread_id,
            "message": "Agent initialized successfully",
            "step": "search_params_ready",
            "parsed_params": {
                "city": parsed_query.city,
                "area": parsed_query.area,
                "type": parsed_query.place_type,
                "intent": parsed_query.intent
            },
            "places_found": len(results)
        }
    
    except Exception as e:
        logger.error(f"Start agent error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ==========================================
# ENDPOINT 3: ACTIVATE AGENT MODE
# ==========================================

@app.post("/agent/activate", response_model=AgentResponse)
async def activate_agent(request: AgentActivationRequest):
    """
    Activates agent mode with user's search results.
    Starts the LangGraph workflow and returns thread_id.
    
    Will pause at HITL checkpoint for human approval.
    """
    try:
        # Generate thread ID if not provided
        thread_id = request.thread_id or str(uuid.uuid4())
        
        logger.info(f"Activating agent for thread: {thread_id}")
        
        # Convert SearchResult models to dicts
        serp_results = [result.model_dump() for result in request.search_results]
        
        # Prepare initial state
        initial_state = {
            "messages": [HumanMessage(content=request.user_intent)],
            "city": request.search_results[0].address.split(",")[-1].strip() if request.search_results else "Unknown",
            "place_type": request.search_results[0].type if request.search_results else "unknown",
            "user_intent": request.user_intent,
            "budget": request.budget,
            "serp_results": serp_results,
            "tavily_reviews": [],
            "human_approved": False,
            "human_notes": None,
            "route": "",
            "current_step": "start",
            "initial_analysis": [],
            "shop_responses": [],
            "refined_analysis": [],
            "iteration": 0,
            "recommendations": None,
            "is_complete": False,
            "thread_id": thread_id
        }
        
        # Run agent until HITL interrupt
        config = get_thread_config(thread_id)
        
        # Invoke will run until interrupt_before=["human_review"]
        state = await negotiator_agent.ainvoke(initial_state, config)
        
        # Get last message from agent
        last_message = state["messages"][-1].content if state["messages"] else "Agent initialized"
        
        return AgentResponse(
            message=last_message,
            thread_id=thread_id,
            data={
                "route": state.get("route"),
                "current_step": state.get("current_step"),
                "serp_results": serp_results
            },
            requires_approval=True,  # HITL checkpoint
            is_complete=False
        )
    
    except Exception as e:
        logger.error(f"Agent activation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ==========================================
# ENDPOINT 3: HUMAN APPROVAL (HITL)
# ==========================================

# ==========================================
# ENDPOINT 3: HUMAN APPROVAL (HITL)
# ==========================================

@app.post("/agent/approve", response_model=AgentResponse)
async def approve_and_continue(request: HumanApprovalRequest):
    """
    Human-in-the-Loop approval endpoint.
    User reviews data and approves/modifies before agent continues.
    
    Resumes graph execution after interrupt.
    """
    try:
        logger.info("="*60)
        logger.info("✅ HITL APPROVAL RECEIVED")
        logger.info(f"   Thread: {request.thread_id}")
        logger.info(f"   Approved: {request.approved}")
        logger.info(f"   Edited Message: {request.edited_message}")
        logger.info("="*60)
        
        config = get_thread_config(request.thread_id)
        
        # Get current state
        current_state = await negotiator_agent.aget_state(config)
        
        if not current_state:
            logger.error("Thread not found!")
            raise HTTPException(status_code=404, detail="Thread not found")
        
        logger.info(f"   Current Step: {current_state.values.get('current_step')}")
        logger.info(f"   Next Node: {current_state.next}")
        
        if request.approved:
            # User approved - update state with approval
            logger.info("📝 Updating state with human_approved=True")
            
            update_state = {
                "human_approved": True,
                "human_notes": request.user_notes
            }
            
            # If user edited negotiation message, use that
            if request.edited_message:
                update_state["planned_message"] = request.edited_message
                logger.info(f"   Using edited message: {request.edited_message}")
            
            # Update state first
            await negotiator_agent.aupdate_state(config, update_state)
            
            # Now invoke to continue from human_review node
            logger.info("▶️ Resuming graph execution...")
            state = await negotiator_agent.ainvoke(None, config)
            
            logger.info(f"   After invoke - Current Step: {state.get('current_step')}")
            logger.info(f"   Negotiation Status: {state.get('negotiation_status')}")
                
        else:
            # User rejected
            logger.info("❌ User rejected negotiation")
            return AgentResponse(
                message="Negotiation cancelled by user",
                thread_id=request.thread_id,
                is_complete=True
            )
        
        # Check current state after execution
        final_state = await negotiator_agent.aget_state(config)
        is_complete = final_state.values.get("is_complete", False)
        last_message = final_state.values.get("messages", [])[-1].content if final_state.values.get("messages") else "Processing..."
        negotiation_status = final_state.values.get("negotiation_status", "")
        
        logger.info(f"   Final Status: {negotiation_status}")
        logger.info(f"   Is Complete: {is_complete}")
        
        # Check if waiting for reply
        requires_approval = False
        if final_state.next and "human_review" in final_state.next:
            requires_approval = True
            logger.info("⏸️ Paused at human_review again - next turn")
        
        return AgentResponse(
            message=last_message,
            thread_id=request.thread_id,
            data={
                "recommendations": final_state.values.get("recommendations"),
                "current_step": final_state.values.get("current_step"),
                "negotiation_status": negotiation_status,
                "planned_message": final_state.values.get("planned_message")
            },
            requires_approval=requires_approval,
            is_complete=is_complete or negotiation_status == "end"
        )
    
    except Exception as e:
        logger.error(f"❌ Approval error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


# ==========================================
# ENDPOINT: CHECK FOR REPLY (Chat Interface Polling)
# ==========================================

@app.get("/agent/check-reply")
async def check_for_reply(thread_id: str):
    """
    Poll endpoint for checking if shopkeeper replied.
    Called by the chat interface every 10 seconds.
    
    CRITICAL FILTERING:
    1. Only messages FROM the salesperson's phone (9391060967 - your phone)
    2. Only messages AFTER the negotiation started
    3. Messages must be numeric phone numbers (filters out "JD-EKARTL-S", "AD-ICICIT-S", etc.)
    """
    try:
        from app.messaging.service import get_messaging_service
        import time
        
        logger.info(f"🔍 Checking for replies - Thread: {thread_id}")
        
        config = get_thread_config(thread_id)
        current_state = await negotiator_agent.aget_state(config)
        
        if not current_state:
            return {"has_reply": False, "status": "thread_not_found"}
        
        # Get the SALESPERSON'S phone number (your phone - where SMS was sent TO)
        # This is stored in settings.twilio_target_number
        salesperson_phone = settings.twilio_target_number.replace("+91", "").replace("+", "").replace(" ", "").replace("-", "")
        logger.info(f"📞 Salesperson phone (your phone): {salesperson_phone}")
        
        # Get negotiation start time from state (to filter old messages)
        negotiation_start_time = current_state.values.get("negotiation_start_time")
        if not negotiation_start_time:
            # Set it now if not set
            negotiation_start_time = int(time.time())
            await negotiator_agent.aupdate_state(config, {
                "negotiation_start_time": negotiation_start_time
            })
        
        logger.info(f"⏰ Negotiation started at: {negotiation_start_time}")
        
        # Get target place name for display
        recommendations = current_state.values.get("recommendations", [])
        target_place_name = recommendations[0].get("name", "Shopkeeper") if recommendations else "Shopkeeper"
        
        # Get messaging service and fetch messages
        messaging = get_messaging_service()
        all_messages = messaging.get_messages()
        
        logger.info(f"📬 Found {len(all_messages)} total messages in inbox")
        
        # Get negotiation history from state
        history = current_state.values.get("negotiation_history", [])
        
        # Filter messages - ONLY from the salesperson's phone AND after negotiation started
        new_replies = []
        for sms in all_messages:
            # SMSMobileAPI returns 'number' field (e.g., "919391060967" or "JD-EKARTL-S")
            sms_number = str(sms.get("number", ""))
            sms_text = sms.get("message", "") or sms.get("body", "") or sms.get("text", "")
            sms_timestamp = sms.get("timestamp_unix", "0")
            sms_date = sms.get("date", "")
            sms_hour = sms.get("hour", "")
            
            # FILTER 1: Must be from a NUMERIC phone number (filters out "JD-EKARTL-S", "AD-ICICIT-S", etc.)
            # Real phone numbers are numeric, spam/service messages have alphanumeric senders
            clean_number = sms_number.replace("+", "").replace("-", "").replace(" ", "")
            if not clean_number.isdigit():
                continue  # Skip - not a real phone number (it's a service like Ekart, ICICI, etc.)
            
            # FILTER 2: Must be from the salesperson's phone number
            if salesperson_phone not in clean_number and clean_number not in salesperson_phone:
                continue  # Skip - not from the salesperson
            
            # FILTER 3: Must be AFTER the negotiation started
            try:
                msg_time = int(sms_timestamp) if sms_timestamp else 0
                if msg_time > 0 and msg_time < negotiation_start_time:
                    logger.info(f"⏭️ Skipping old message (before negotiation): {sms_text[:30]}...")
                    continue  # Skip - this message is from before the negotiation started
            except:
                pass  # If timestamp parsing fails, include the message
            
            logger.info(f"✅ Valid reply from salesperson: {sms_text[:50]}...")
            
            # Check if we've already processed this message
            already_in_history = any(
                h.get("content") == sms_text and h.get("role") == "shopkeeper"
                for h in history
            )
            
            if not already_in_history and sms_text:
                new_replies.append({
                    "text": sms_text,
                    "from": sms_number,
                    "time": f"{sms_date} {sms_hour}" if sms_date else sms_timestamp
                })
        
        if new_replies:
            # Found new reply from the salesperson!
            latest_reply = new_replies[0]
            logger.info(f"📩 NEW REPLY from salesperson: {latest_reply['text']}")
            
            # Update negotiation history with shopkeeper's reply
            new_history = history + [{
                "role": "shopkeeper",
                "content": latest_reply["text"],
                "timestamp": latest_reply["time"]
            }]
            
            await negotiator_agent.aupdate_state(config, {
                "negotiation_history": new_history,
                "negotiation_status": "ongoing"  # Reset to ongoing so agent can continue
            })
            
            # Generate agent's counter-offer using LLM
            agent_response = await generate_negotiation_response(
                thread_id=thread_id,
                shopkeeper_reply=latest_reply["text"],
                history=new_history,
                current_state=current_state
            )
            
            return {
                "has_reply": True,
                "reply_text": latest_reply["text"],
                "reply_time": latest_reply["time"],
                "from_phone": latest_reply["from"],
                "status": "reply_received",
                "agent_response": agent_response.get("message"),
                "thought_process": agent_response.get("thought")
            }
        
        return {
            "has_reply": False,
            "status": "waiting",
            "message": f"Waiting for reply from salesperson ({salesperson_phone})"
        }
        
    except Exception as e:
        logger.error(f"❌ Check reply error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return {"has_reply": False, "status": "error", "error": str(e)}


# ==========================================
# HELPER: Generate Negotiation Response
# ==========================================

async def generate_negotiation_response(thread_id: str, shopkeeper_reply: str, history: list, current_state) -> dict:
    """
    Use LLM to generate a smart negotiation counter-offer based on shopkeeper's reply.
    """
    try:
        from langchain_groq import ChatGroq
        from pydantic import BaseModel, Field
        
        # Get context from state
        target_price = current_state.values.get("target_price", 3000)
        user_goal = current_state.values.get("user_intent", "Get the best price")
        recommendations = current_state.values.get("recommendations", [])
        place_name = recommendations[0].get("name", "the place") if recommendations else "the place"
        
        # Build conversation history for context
        conversation = "\n".join([
            f"{'Agent' if h['role'] == 'agent' else 'Shopkeeper'}: {h['content']}"
            for h in history
        ])
        
        class NegotiationResponse(BaseModel):
            """Negotiation response"""
            message: str = Field(description="The counter-offer or response message to send")
            thought: str = Field(description="Brief explanation of the negotiation strategy")
            should_accept: bool = Field(description="True if the offer is acceptable and we should accept")
            should_reject: bool = Field(description="True if negotiation should end without deal")
        
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.7,
            api_key=settings.groq_api_key
        )
        
        structured_llm = llm.with_structured_output(NegotiationResponse)
        
        prompt = f"""You are a skilled negotiator helping a customer get the best price.

CONTEXT:
- Place: {place_name}
- Customer's Target Price: ₹{target_price}/month
- Customer's Goal: {user_goal}

CONVERSATION SO FAR:
{conversation}

SHOPKEEPER'S LATEST REPLY: "{shopkeeper_reply}"

YOUR TASK:
1. Analyze the shopkeeper's response
2. Decide if we should:
   - Accept (if price is at or below target)
   - Counter-offer (if there's room for negotiation)
   - Politely decline (if price is too far from target and shopkeeper won't budge)

3. Generate a short, polite, and persuasive response (1-2 sentences max)

GUIDELINES:
- Be polite but firm
- If shopkeeper offers close to target (within 500), consider accepting
- If shopkeeper is firm, try one more counter before accepting/declining
- Keep messages SHORT and conversational (like real SMS)
- Don't be too pushy - 2-3 counter-offers max

Generate the next message to send."""

        logger.info("🧠 Generating negotiation counter-offer...")
        response = structured_llm.invoke(prompt)
        
        logger.info(f"📝 Agent response: {response.message}")
        logger.info(f"💭 Thought: {response.thought}")
        logger.info(f"✅ Should accept: {response.should_accept}")
        
        return {
            "message": response.message,
            "thought": response.thought,
            "should_accept": response.should_accept,
            "should_reject": response.should_reject
        }
        
    except Exception as e:
        logger.error(f"❌ Error generating negotiation response: {str(e)}")
        return {
            "message": "Can you do any better on the price?",
            "thought": "Fallback response due to error",
            "should_accept": False,
            "should_reject": False
        }


# ==========================================
# ENDPOINT: SEND MANUAL MESSAGE (User types in chat)
# ==========================================

@app.post("/agent/send-manual")
async def send_manual_message(request: dict):
    """
    Send a manual message typed by user in chat interface.
    Bypasses the agent strategy - direct user control.
    """
    try:
        from app.messaging.service import get_messaging_service
        
        thread_id = request.get("thread_id")
        message = request.get("message")
        
        if not thread_id or not message:
            raise HTTPException(status_code=400, detail="thread_id and message required")
        
        logger.info(f"📤 MANUAL MESSAGE from user")
        logger.info(f"   Thread: {thread_id}")
        logger.info(f"   Message: {message}")
        
        config = get_thread_config(thread_id)
        current_state = await negotiator_agent.aget_state(config)
        
        if not current_state:
            raise HTTPException(status_code=404, detail="Thread not found")
        
        # Resolve target phone: prefer place phone from state, fallback to configured default
        state_values = current_state.values if hasattr(current_state, "values") else {}

        def normalize_phone(num: str) -> str:
            if not num:
                return ""
            num = num.strip()
            if num.startswith("+"):
                return "+" + re.sub(r"[^0-9]", "", num)
            return re.sub(r"[^0-9]", "", num)

        recs = state_values.get("recommendations") or state_values.get("serp_results") or []
        place_phone = recs[0].get("phone") if recs and isinstance(recs[0], dict) else ""
        target_number = (
            normalize_phone(place_phone)
            or normalize_phone(settings.default_target_number)
            or normalize_phone(settings.twilio_target_number)
        )

        if not target_number:
            raise HTTPException(status_code=400, detail="No target phone available to send SMS")

        messaging = get_messaging_service()
        
        success = messaging.send_message(to=target_number, message=message)
        
        if success:
            logger.info(f"✅ Manual message sent to {target_number}")
            
            # Update negotiation history
            history = current_state.values.get("negotiation_history", [])
            history.append({
                "role": "agent",
                "content": message,
                "timestamp": str(uuid.uuid4())[:8]  # Simple timestamp
            })
            
            await negotiator_agent.aupdate_state(config, {
                "negotiation_history": history
            })
            
            return {"success": True, "message": "Message sent"}
        else:
            logger.error("❌ Failed to send manual message")
            return {"success": False, "message": "Failed to send"}
            
    except Exception as e:
        logger.error(f"❌ Send manual error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


# ==========================================
# ENDPOINT: SEND CHAT CONTINUATION MESSAGE (Agent suggested response)
# ==========================================

@app.post("/agent/send-chat")
async def send_chat_message(request: dict):
    """
    Send an agent-suggested chat message (for negotiation continuation).
    Called when user approves the agent's suggested response.
    
    This is different from /agent/approve - it doesn't invoke the graph.
    It directly sends the SMS and updates negotiation history.
    """
    try:
        from app.messaging.service import get_messaging_service
        import time
        
        thread_id = request.get("thread_id")
        message = request.get("message")
        
        if not thread_id or not message:
            raise HTTPException(status_code=400, detail="thread_id and message required")
        
        logger.info("="*60)
        logger.info("📤 CHAT CONTINUATION MESSAGE")
        logger.info(f"   Thread: {thread_id}")
        logger.info(f"   Message: {message}")
        logger.info("="*60)
        
        config = get_thread_config(thread_id)
        current_state = await negotiator_agent.aget_state(config)
        
        if not current_state:
            raise HTTPException(status_code=404, detail="Thread not found")
        
        # Resolve target phone: prefer place phone from state, fallback to configured default
        state_values = current_state.values if hasattr(current_state, "values") else {}

        def normalize_phone(num: str) -> str:
            if not num:
                return ""
            num = num.strip()
            if num.startswith("+"):
                return "+" + re.sub(r"[^0-9]", "", num)
            return re.sub(r"[^0-9]", "", num)

        recs = state_values.get("recommendations") or state_values.get("serp_results") or []
        place_phone = recs[0].get("phone") if recs and isinstance(recs[0], dict) else ""
        target_number = (
            normalize_phone(place_phone)
            or normalize_phone(settings.default_target_number)
            or normalize_phone(settings.twilio_target_number)
        )

        if not target_number:
            raise HTTPException(status_code=400, detail="No target phone available to send SMS")

        # Send SMS
        messaging = get_messaging_service()
        
        logger.info(f"📱 Sending SMS to {target_number}...")
        success = messaging.send_message(to=target_number, message=message)
        
        if success:
            logger.info(f"✅ Chat message sent to {target_number}")
            
            # Update negotiation history with agent's message
            history = current_state.values.get("negotiation_history", [])
            history.append({
                "role": "agent",
                "content": message,
                "timestamp": str(int(time.time()))
            })
            
            await negotiator_agent.aupdate_state(config, {
                "negotiation_history": history,
                "negotiation_status": "ongoing"
            })
            
            return {
                "success": True, 
                "message": "Message sent",
                "sent_message": message
            }
        else:
            logger.error("❌ Failed to send chat message")
            return {"success": False, "message": "Failed to send SMS"}
            
    except Exception as e:
        logger.error(f"❌ Send chat error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


# ==========================================
# ENDPOINT: TERMINATE NEGOTIATION
# ==========================================

@app.post("/agent/terminate")
async def terminate_negotiation(request: dict):
    """
    Terminate an ongoing negotiation.
    Called when user clicks the 'End' button in chat interface.
    """
    try:
        thread_id = request.get("thread_id")
        
        if not thread_id:
            raise HTTPException(status_code=400, detail="thread_id required")
        
        logger.info("="*60)
        logger.info("🛑 NEGOTIATION TERMINATED BY USER")
        logger.info(f"   Thread: {thread_id}")
        logger.info("="*60)
        
        config = get_thread_config(thread_id)
        current_state = await negotiator_agent.aget_state(config)
        
        if not current_state:
            return {"success": True, "message": "Thread not found (already terminated)"}
        
        # Update state to mark negotiation as ended
        await negotiator_agent.aupdate_state(config, {
            "negotiation_active": False,
            "negotiation_status": "terminated",
            "is_complete": True
        })
        
        logger.info("✅ Negotiation state updated to terminated")
        
        return {
            "success": True,
            "message": "Negotiation terminated successfully",
            "status": "terminated"
        }
        
    except Exception as e:
        logger.error(f"❌ Terminate error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return {"success": False, "message": str(e)}


# ==========================================
# ENDPOINT 8: START NEGOTIATION (User clicked NEGOTIATE button)
# ==========================================

@app.post("/agent/negotiate/start", response_model=AgentResponse)
async def start_negotiation(request: NegotiationStartRequest):
    """
    Triggers the negotiation workflow for a specific place.
    Called when user clicks NEGOTIATE button on a place.
    
    Flow: Updates state with route=path_b, then re-invokes graph.
    The revisor_node will see route=path_b and skip to negotiation_path.
    """
    try:
        logger.info("="*60)
        logger.info("🤝 NEGOTIATE BUTTON CLICKED")
        logger.info(f"   Place: {request.place_name}")
        logger.info(f"   Thread: {request.thread_id}")
        logger.info(f"   Target Price: {request.target_price}")
        logger.info(f"   Goal: {request.initial_message}")
        logger.info("="*60)
        
        config = get_thread_config(request.thread_id)
        
        # Get current state to find the place object
        current_state = await negotiator_agent.aget_state(config)
        if not current_state:
             raise HTTPException(status_code=404, detail="Thread not found")
             
        serp_results = current_state.values.get("serp_results", [])
        logger.info(f"📋 Found {len(serp_results)} places in state")
        
        # Find the place by name (fuzzy match)
        target_place = None
        for p in serp_results:
            if p.get("name") == request.place_name or request.place_name in p.get("name", ""):
                target_place = p
                break
        
        if not target_place:
            logger.warning(f"⚠️ Place '{request.place_name}' not found in serp_results, creating minimal object")
            target_place = {"name": request.place_name, "address": "Unknown", "phone": "Unknown"}
        else:
            logger.info(f"✅ Found place: {target_place.get('name')}")
            logger.info(f"   Phone: {target_place.get('phone', 'N/A')}")
            logger.info(f"   Address: {target_place.get('address', 'N/A')}")
        
        # Get current timestamp for filtering messages later
        import time
        negotiation_start_time = int(time.time())
        logger.info(f"⏰ Negotiation start time: {negotiation_start_time}")
        
        # Build the state update for negotiation
        negotiation_state = {
            "route": "path_b",  # Force PATH B (Negotiation)
            "is_complete": False,
            "recommendations": [target_place],
            "target_price": request.target_price,
            "user_intent": request.initial_message or f"Negotiate price for {request.place_name}",
            "negotiation_active": True,
            "negotiation_status": "ongoing",
            "negotiation_history": [],  # Fresh start
            "negotiation_start_time": negotiation_start_time,  # Track when negotiation started
            "messages": [HumanMessage(content=f"User wants to negotiate with {request.place_name}. Goal: {request.initial_message or 'Get best price'}")]
        }
        
        logger.info("📝 Updating state for negotiation...")
        
        # Update state and invoke graph from beginning
        # The revisor_node will see route=path_b and go to negotiation_path
        result = await negotiator_agent.ainvoke(negotiation_state, config)
        
        logger.info("✅ Negotiation flow started!")
        logger.info(f"   Current step: {result.get('current_step', 'unknown')}")
        
        # Check if we hit the HITL interrupt
        state_after = await negotiator_agent.aget_state(config)
        next_node = state_after.next if state_after else None
        
        response_msg = "Negotiation started"
        requires_approval = False
        planned_message = None
        
        if next_node and "human_review" in next_node:
            logger.info("⏸️ Graph paused at human_review - waiting for approval")
            requires_approval = True
            planned_message = state_after.values.get("planned_message", "")
            response_msg = f"Review negotiation message before sending"
        
        return AgentResponse(
            message=response_msg,
            thread_id=request.thread_id,
            data={
                "planned_message": planned_message,
                "target_place": target_place,
                "requires_approval": requires_approval
            },
            requires_approval=requires_approval,
            is_complete=False
        )

    except Exception as e:
        logger.error(f"❌ Negotiation start error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


# ==========================================
# ENDPOINT 4: CONVERSATIONAL CHAT
# ==========================================

@app.post("/agent/chat", response_model=AgentResponse)
async def chat_with_agent(request: AgentChatRequest):
    """
    Continue conversation with agent after recommendations.
    User can ask follow-up questions, request refinements.
    
    Uses existing thread_id to maintain context (memory).
    """
    try:
        logger.info(f"Chat request for thread: {request.thread_id}")
        
        config = get_thread_config(request.thread_id)
        
        # Get current state
        current_state = negotiator_agent.get_state(config)
        
        if not current_state:
            raise HTTPException(status_code=404, detail="Thread not found. Please start a new session.")
        
        # Add user message to existing conversation
        new_input = {
            "messages": [HumanMessage(content=request.message)]
        }
        
        # Invoke agent with new message
        state = await negotiator_agent.ainvoke(new_input, config)
        
        # Get response
        last_message = state["messages"][-1].content if state["messages"] else "No response"
        
        return AgentResponse(
            message=last_message,
            thread_id=request.thread_id,
            data={
                "recommendations": state.get("recommendations"),
                "current_step": state.get("current_step")
            },
            is_complete=state.get("is_complete", False)
        )
    
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ==========================================
# ENDPOINT 5: STREAMING RESPONSE
# ==========================================

@app.get("/agent/stream")
async def stream_agent(thread_id: str):
    """
    Streaming endpoint for real-time agent responses with verbose logging.
    Returns Server-Sent Events (SSE) for frontend with detailed API/LLM calls.
    Uses existing thread_id from /agent/start.
    """
    
    async def event_generator():
        try:
            config = get_thread_config(thread_id)
            
            # Get current state
            state_snapshot = await negotiator_agent.aget_state(config)
            
            if not state_snapshot or not state_snapshot.values:
                error_data = json.dumps({"type": "error", "error": "Thread not found"})
                yield f"data: {error_data}\n\n"
                return
            
            logger.info(f"Streaming for thread: {thread_id}")
            
            # Send initial state info
            init_data = json.dumps({
                "type": "init",
                "city": state_snapshot.values.get("city"),
                "place_type": state_snapshot.values.get("place_type"),
                "serp_results_count": len(state_snapshot.values.get("serp_results", []))
            })
            yield f"data: {init_data}\n\n"
            
            # Show the SerpStack results first (from STEP 4)
            serp_results = state_snapshot.values.get("serp_results", [])
            if serp_results:
                results_preview = json.dumps({
                    "type": "data_fetched",
                    "source": "SerpStack",
                    "count": len(serp_results),
                    "sample": serp_results[:3] if len(serp_results) > 3 else serp_results
                })
                yield f"data: {results_preview}\n\n"
                logger.info(f"Showed {len(serp_results)} SerpStack results")
            
            # Check if we need to resume or if it's complete
            if not state_snapshot.next:
                # Graph hasn't started yet or is complete
                if state_snapshot.values.get("is_complete"):
                    complete_data = json.dumps({
                        "type": "complete",
                        "recommendations": state_snapshot.values.get("recommendations", []),
                        "all_places": state_snapshot.values.get("serp_results", []),
                        "iteration": state_snapshot.values.get("iteration", 0)
                    })
                    yield f"data: {complete_data}\n\n"
                    return
                
                # Graph hasn't started - kick it off now
                logger.info("Starting agent execution...")
            
            # Check if we're paused at human_review interrupt - DON'T continue, just show HITL UI
            if state_snapshot.next and "human_review" in state_snapshot.next:
                logger.info("Graph paused at human_review - showing HITL panel")
                planned_msg = state_snapshot.values.get("planned_message", "")
                interrupt_data = json.dumps({
                    "type": "interrupt",
                    "node": "human_review",
                    "plan": "Review negotiation message",
                    "message": planned_msg,
                    "target_place": state_snapshot.values.get("recommendations", [{}])[0] if state_snapshot.values.get("recommendations") else {}
                })
                yield f"data: {interrupt_data}\n\n"
                return  # Don't continue streaming - wait for approval
            
            # Check if negotiation is waiting for reply
            if state_snapshot.values.get("negotiation_status") == "end":
                # Get target place name for chat header
                recs = state_snapshot.values.get("recommendations", [])
                target_place_name = recs[0].get("name", "Shopkeeper") if recs else "Shopkeeper"
                
                # Get last sent message
                history = state_snapshot.values.get("negotiation_history", [])
                last_message = ""
                for msg in reversed(history):
                    if msg.get("role") == "agent":
                        last_message = msg.get("content", "")
                        break
                
                # If no history, check planned_message
                if not last_message:
                    last_message = state_snapshot.values.get("planned_message", "Message sent")
                
                waiting_data = json.dumps({
                    "type": "waiting",
                    "message": "SMS sent! Waiting for reply from shopkeeper...",
                    "target_place": target_place_name,
                    "last_message": last_message
                })
                yield f"data: {waiting_data}\n\n"
                return
            
            # Stream state updates using stream_mode="updates" to track node changes
            previous_step = state_snapshot.values.get("current_step", "")
            
            async for chunk in negotiator_agent.astream(None, config, stream_mode="updates"):
                # chunk is a dict like: {"node_name": updated_state}
                for node_name, node_state in chunk.items():
                    logger.info(f"Node update: {node_name}")
                    
                    # Node started
                    start_data = json.dumps({
                        "type": "node_start",
                        "node": node_name,
                        "timestamp": str(uuid.uuid4())[:8]
                    })
                    yield f"data: {start_data}\n\n"
                    
                    # Check for specific node types to extract details
                    current_step = node_state.get("current_step", "")
                    
                    # Router node - show decision
                    if node_name == "revisor":
                        router_decision = node_state.get("show_all", False)
                        decision_data = json.dumps({
                            "type": "router_decision",
                            "requires_negotiation": False, # Revisor doesn't decide negotiation yet in this path
                            "reason": "Showing ALL results" if router_decision else "Showing BEST result"
                        })
                        yield f"data: {decision_data}\n\n"
                    
                    # Review Extraction node
                    elif node_name == "review_extraction":
                        reviews = node_state.get("reviews", {})
                        review_data = json.dumps({
                            "type": "reviews_fetched",
                            "count": len(reviews),
                            "reviews": reviews  # Send the full reviews map {PlaceName: [Review1, Review2]}
                        })
                        yield f"data: {review_data}\n\n"
                    
                    # Analyze Reviews node
                    elif node_name == "analyze_reviews":
                        messages = node_state.get("messages", [])
                        if messages:
                            last_msg = messages[-1]
                            if hasattr(last_msg, "content"):
                                llm_data = json.dumps({
                                    "type": "llm_response",
                                    "model": "llama-3.3-70b",
                                    "content": str(last_msg.content)
                                })
                                yield f"data: {llm_data}\n\n"

                    # Analyze node - show LLM analysis
                    elif node_name == "analyze_node":
                        messages = node_state.get("messages", [])
                        if messages:
                            last_msg = messages[-1]
                            if hasattr(last_msg, "content"):
                                llm_data = json.dumps({
                                    "type": "llm_response",
                                    "model": "gemini-2.5-flash",
                                    "content": str(last_msg.content)[:500]  # First 500 chars
                                })
                                yield f"data: {llm_data}\n\n"
                    
                    # Negotiate node - show simulation
                    elif node_name == "negotiate_node":
                        negotiations = node_state.get("negotiations", [])
                        if negotiations:
                            nego_data = json.dumps({
                                "type": "negotiation",
                                "count": len(negotiations),
                                "sample": negotiations[-1] if negotiations else None
                            })
                            yield f"data: {nego_data}\n\n"
                    
                    # Reflexion node - show iteration
                    elif node_name == "reflexion_node":
                        iteration = node_state.get("iteration", 0)
                        reflexion_data = json.dumps({
                            "type": "reflexion",
                            "iteration": iteration,
                            "status": "Refining recommendations..."
                        })
                        yield f"data: {reflexion_data}\n\n"
                    
                    # Node completed
                    end_data = json.dumps({
                        "type": "node_end",
                        "node": node_name,
                        "iteration": node_state.get("iteration", 0)
                    })
                    yield f"data: {end_data}\n\n"
                
                # Check for interrupts after each update
                current_state = await negotiator_agent.aget_state(config)
                
                # Check if interrupted for HITL (Human Review)
                if current_state.next and current_state.next == ("human_review",):
                    # Extract planned message
                    planned_msg = current_state.values.get("planned_message", "No plan generated")
                    
                    interrupt_data = json.dumps({
                        "type": "interrupt",
                        "node": "human_review",
                        "plan": "Review negotiation strategy",
                        "message": planned_msg,
                        "serp_results": current_state.values.get("serp_results", [])
                    })
                    yield f"data: {interrupt_data}\n\n"
                    logger.info("HITL interrupt detected (Negotiation)")
                    return
                
                # Check if complete
                if not current_state.next and current_state.values.get("is_complete"):
                    complete_data = json.dumps({
                        "type": "complete",
                        "recommendations": current_state.values.get("recommendations", []),
                        "all_places": current_state.values.get("serp_results", []),
                        "iteration": current_state.values.get("iteration", 0)
                    })
                    yield f"data: {complete_data}\n\n"
                    logger.info("Agent completed")
                    return
                
                # Check if waiting for reply (Negotiation Loop)
                negotiation_status = current_state.values.get("negotiation_status")
                if negotiation_status == "end" and not current_state.values.get("is_complete"):
                    # Get target place name for chat header
                    recs = current_state.values.get("recommendations", [])
                    target_place_name = recs[0].get("name", "Shopkeeper") if recs else "Shopkeeper"
                    
                    # Get last sent message
                    history = current_state.values.get("negotiation_history", [])
                    last_message = ""
                    for msg in reversed(history):
                        if msg.get("role") == "agent":
                            last_message = msg.get("content", "")
                            break
                    
                    if not last_message:
                        last_message = current_state.values.get("planned_message", "Message sent")
                    
                    waiting_data = json.dumps({
                        "type": "waiting",
                        "message": "SMS sent! Waiting for reply from shopkeeper...",
                        "target_place": target_place_name,
                        "last_message": last_message
                    })
                    yield f"data: {waiting_data}\n\n"
                    logger.info("Agent waiting for reply")
                    return
        
        except Exception as e:
            logger.error(f"Streaming error: {str(e)}")
            error_data = json.dumps({"type": "error", "error": str(e)})
            yield f"data: {error_data}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


# ==========================================
# ENDPOINT 6: GET CONVERSATION HISTORY
# ==========================================

@app.get("/agent/history/{thread_id}")
async def get_history(thread_id: str):
    """
    Retrieve conversation history for a thread.
    Shows all checkpoints and state transitions.
    """
    try:
        config = get_thread_config(thread_id)
        
        # Get state history
        history = []
        for state in negotiator_agent.get_state_history(config):
            history.append({
                "checkpoint_id": state.config.get("configurable", {}).get("checkpoint_id"),
                "step": state.metadata.get("step", 0),
                "current_node": state.values.get("current_step"),
                "is_complete": state.values.get("is_complete", False),
                "message_count": len(state.values.get("messages", []))
            })
        
        return {
            "thread_id": thread_id,
            "total_checkpoints": len(history),
            "history": history
        }
    
    except Exception as e:
        logger.error(f"History retrieval error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ==========================================
# ENDPOINT 7: RESET CONVERSATION
# ==========================================

@app.delete("/agent/reset/{thread_id}")
async def reset_thread(thread_id: str):
    """
    Clear conversation history for a thread.
    Useful for starting fresh.
    """
    try:
        # Note: LangGraph doesn't have direct delete,
        # but starting with new thread_id effectively resets
        return {
            "message": f"Thread {thread_id} marked for reset. Start new conversation with different thread_id.",
            "new_thread_id": str(uuid.uuid4())
        }
    
    except Exception as e:
        logger.error(f"Reset error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ==========================================
# ENDPOINT 9: QUICK START (SEARCH + ACTIVATE)
# ==========================================

@app.post("/agent/start", response_model=AgentResponse)
async def quick_start_agent(
    city: str,
    place_type: str,
    budget: Optional[float] = None,
    user_preferences: Optional[str] = None
):
    """
    Convenience endpoint that combines /search and /agent/activate.
    Searches for places and immediately activates agent mode.
    
    Usage:
        POST /agent/start?city=Bangalore&place_type=gym&budget=5000&user_preferences=Looking for good equipment
    """
    try:
        logger.info(f"Quick start: {city}, {place_type}")
        
        # Step 1: Search for places
        search_req = SearchRequest(
            city=city,
            place_type=PlaceType(place_type.lower()),
            query=user_preferences
        )
        
        results = search_places.invoke({
            "city": search_req.city,
            "place_type": search_req.place_type.value,
            "query": search_req.query
        })
        
        if not results or any("error" in r for r in results):
            raise HTTPException(
                status_code=404, 
                detail=f"No places found or API error: {results[0].get('error', 'Unknown') if results else 'No results'}"
            )
        
        # Convert to SearchResult models
        search_results = [
            SearchResult(
                name=place.get("name", "Unknown"),
                address=place.get("address", ""),
                phone=place.get("phone"),
                rating=place.get("rating"),
                price_level=place.get("price_level"),
                type=place.get("type", place_type)
            )
            for place in results
            if "error" not in place
        ]
        
        if not search_results:
            raise HTTPException(status_code=404, detail="No valid places found")
        
        logger.info(f"Found {len(search_results)} places")
        
        # Step 2: Activate agent with results
        user_intent = user_preferences or f"Find the best {place_type} in {city}"
        if budget:
            user_intent += f" within budget of ₹{budget}"
        
        activation_req = AgentActivationRequest(
            search_results=search_results,
            user_intent=user_intent,
            budget=budget
        )
        
        # Activate agent
        return await activate_agent(activation_req)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Quick start error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ==========================================
# RUN SERVER
# ==========================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.environment == "development"
    )
