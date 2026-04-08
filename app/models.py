from pydantic import BaseModel, Field
from typing import Literal, Optional
from enum import Enum


# ==========================================
# NEW: Simple query model for natural language input
# ==========================================

class InitialQueryRequest(BaseModel):
    """Natural language query from user (STEP 1)"""
    user_query: str = Field(..., description="Natural language query like 'I need gyms in koramangala in bangalore'")


class PlaceType(str, Enum):
    """Supported place types"""
    CAFE = "cafe"
    RESTAURANT = "restaurant"
    HOSPITAL = "hospital"
    HOTEL = "hotel"
    MENS_PG = "mens_pg"
    WOMENS_PG = "womens_pg"
    PAYING_GUEST = "paying_guest"
    GYM = "gym"
    PHARMACY = "pharmacy"
    BANK = "bank"
    ATM = "atm"
    GAS_STATION = "gas_station"


class SearchRequest(BaseModel):
    """Initial search request"""
    city: str = Field(..., description="City or area name")
    place_type: PlaceType = Field(..., description="Type of place to search")
    query: Optional[str] = Field(None, description="Additional search query")
    user_preferences: Optional[str] = Field("", description="User requirements (can include budget if needed)")


class SearchResult(BaseModel):
    """Single place from SERP results"""
    name: str
    address: str
    phone: Optional[str] = None
    rating: Optional[float] = None
    price_level: Optional[str] = None
    type: str


class AgentActivationRequest(BaseModel):
    """Request to activate agent mode"""
    thread_id: Optional[str] = Field(None, description="Conversation thread ID")
    search_results: list[SearchResult] = Field(..., description="SERP results to analyze")
    user_intent: str = Field(..., description="What user wants to achieve")
    budget: Optional[float] = Field(None, description="Budget constraint if any")


class AgentChatRequest(BaseModel):
    """Conversational message to agent"""
    thread_id: str = Field(..., description="Conversation thread ID")
    message: str = Field(..., description="User message")


class HumanApprovalRequest(BaseModel):
    """HITL approval or modification"""
    thread_id: str
    approved: bool = Field(default=True)
    modified_results: Optional[list[SearchResult]] = None
    user_notes: Optional[str] = None
    edited_message: Optional[str] = None # For negotiation message editing


class NegotiationStartRequest(BaseModel):
    """Request to start negotiation for a specific place"""
    thread_id: str
    place_name: str
    target_price: Optional[float] = None
    initial_message: Optional[str] = None


class RouteDecision(BaseModel):
    """Router decision for agent workflow"""
    route: Literal["negotiation", "info_only", "comparison"] = Field(
        description="Type of assistance needed"
    )
    reasoning: str = Field(description="Why this route was chosen")
    can_negotiate: bool = Field(description="Whether negotiation is applicable")


class PlaceAnalysis(BaseModel):
    """Analysis of a single place"""
    name: str
    pros: list[str] = Field(description="Advantages")
    cons: list[str] = Field(description="Disadvantages")
    price_estimate: Optional[str] = None
    suitability_score: float = Field(ge=0, le=10, description="0-10 rating")
    reasoning: str


class ShopResponse(BaseModel):
    """Simulated or real shop response"""
    place_name: str
    response_type: Literal["pricing", "availability", "features", "negotiation"]
    message: str
    pricing_info: Optional[dict] = None
    available: bool = True


class FinalRecommendation(BaseModel):
    """Agent's final recommendation"""
    top_choice: str
    ranked_list: list[PlaceAnalysis]
    reasoning: str
    total_score: float
    negotiation_summary: Optional[str] = None


class AgentResponse(BaseModel):
    """Standard agent response"""
    message: str
    thread_id: str
    data: Optional[dict] = None
    requires_approval: bool = False
    is_complete: bool = False
