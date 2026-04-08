from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
import requests
from typing import Optional
from app.config import settings
from app.models import ShopResponse
import json
import logging
from bs4 import BeautifulSoup
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)


# ==========================================
# SerpStack API Tool
# ==========================================

@tool
def search_places(city: str, place_type: str, query: Optional[str] = None) -> list[dict]:
    """
    Search for places using SerpStack API (Google Search).
    
    Args:
        city: City or area name
        place_type: Type of place (gym, restaurant, cafe, etc.)
        query: Optional additional search terms
    
    Returns:
        List of places with name, address, phone, rating, price_level
    """
    import re
    
    try:
        # Build search query
        if query:
            search_query = query
        else:
            search_query = f"{place_type} in {city}"
        
        # SerpStack API parameters
        params = {
            "access_key": settings.serp_api_key,
            "query": search_query,
            "type": "web",
            "num": 20,  # Request more results to get better data
            "auto_location": 1, # Ensure auto-location is on
            "google_domain": "google.co.in", # Use Google India for better local results
            "gl": "in", # Country code for India
            "hl": "en" # Language English
        }
        
        response = requests.get(settings.serp_api_url, params=params, timeout=15)
        response.raise_for_status()
        
        data = response.json()
        
        # Check for API errors
        if not data.get("request", {}).get("success", True):
            error_info = data.get("error", {})
            return [{"error": f"SerpStack API error: {error_info.get('info', 'Unknown error')}"}]
        
        results = []
        
        # Helper function to extract phone from text
        def extract_phone(text):
            """Extract phone number from text like '+91 99885 93333' or '99885 93333'"""
            if not text:
                return ""
            # Match Indian phone patterns: +91 XXXXX XXXXX or XXXXX XXXXX
            phone_match = re.search(r'(\+91\s*\d{5}\s*\d{5}|\d{5}\s*\d{5})', text)
            return phone_match.group(1) if phone_match else ""
        
        # Create a mapping of place names to phone numbers from related_places
        phone_mapping = {}
        if "related_places" in data:
            logger.info(f"Processing {len(data['related_places'])} related_places for phone extraction")
            for place_obj in data["related_places"]:
                place_text = place_obj.get("places", "")
                phone = extract_phone(place_text)
                if phone:
                    place_title = place_obj.get("title", "").lower().strip()
                    # Store phone number keyed by place title
                    phone_mapping[place_title] = phone
                    logger.info(f"Phone mapping: '{place_title}' -> '{phone}'")
        
        # PRIORITY 1: Parse local_results (has best structured data)
        if "local_results" in data:
            logger.info(f"Processing {len(data['local_results'])} local_results")
            for place in data["local_results"][:10]:
                place_name = place.get("title", "Unknown")
                place_name_lower = place_name.lower().strip()
                
                # Try to get phone from related_places mapping first
                phone = phone_mapping.get(place_name_lower, "")
                
                if not phone:
                    logger.warning(f"No phone found in mapping for '{place_name_lower}'. Available keys: {list(phone_mapping.keys())}")
                
                # Fallback: try to extract from "type" field or address
                if not phone:
                    phone = extract_phone(place.get("type", ""))
                if not phone:
                    phone = extract_phone(place.get("address", ""))
                
                logger.info(f"Final phone for '{place_name}': '{phone}'")
                
                results.append({
                    "name": place_name,
                    "address": place.get("address", ""),
                    "phone": phone,
                    "rating": place.get("rating", None),
                    "reviews_count": place.get("reviews", 0),
                    "price_level": place.get("price", {}),
                    "type": place_type,
                    "extensions": place.get("extensions", {})
                })
        
        # PRIORITY 2: Parse related_places (fallback if no local_results)
        elif "related_places" in data:
            for place_obj in data["related_places"][:10]:
                place_text = place_obj.get("places", "")
                phone = extract_phone(place_text)
                
                # Extract rating from text like "4.8(407)"
                rating_match = re.search(r'(\d+\.\d+)\((\d+)\)', place_text)
                rating = float(rating_match.group(1)) if rating_match else None
                reviews = int(rating_match.group(2)) if rating_match else 0
                
                # Extract address (text between business info and "Closed/Open")
                address_match = re.search(r'business\s*·\s*([^·]+?)(?:Closed|Open)', place_text)
                address = address_match.group(1).strip() if address_match else ""
                
                results.append({
                    "name": place_obj.get("title", "Unknown"),
                    "address": address,
                    "phone": phone,
                    "rating": rating,
                    "reviews_count": reviews,
                    "price_level": {},
                    "type": place_type,
                    "extensions": {}
                })
        
        # PRIORITY 3: Parse organic_results (last resort)
        if not results and "organic_results" in data:
            for idx, result in enumerate(data["organic_results"][:10], 1):
                results.append({
                    "name": result.get("title", "Unknown"),
                    "address": result.get("snippet", "")[:100],
                    "phone": "",
                    "rating": None,
                    "reviews_count": 0,
                    "price_level": {},
                    "type": place_type,
                    "url": result.get("url", ""),
                    "position": idx,
                    "extensions": {}
                })
        
        # If no results found
        if not results:
            return [{
                "name": "No results found",
                "address": f"Try searching for '{search_query}' manually",
                "type": place_type
            }]
        
        return results
    
    except requests.exceptions.RequestException as e:
        return [{"error": f"SerpStack API request error: {str(e)}"}]
    except Exception as e:
        return [{"error": f"SerpStack API error: {str(e)}"}]


# ==========================================
# Tavily Search Tool (Reviews & Sentiment)
# ==========================================

def get_tavily_tool():
    """
    Returns Tavily search tool for reviews and ratings.
    Only initialize if API key is provided.
    """
    if settings.tavily_api_key:
        return TavilySearchResults(
            max_results=5,
            search_depth="advanced",
            include_answer=True,
            include_raw_content=False,
            api_key=settings.tavily_api_key
        )
    return None


@tool
def search_reviews(place_name: str, city: str) -> dict:
    """
    Search for reviews and ratings of a specific place using Tavily.
    
    Args:
        place_name: Name of the establishment
        city: City location
    
    Returns:
        Dictionary with reviews, sentiment, and ratings info
    """
    try:
        tavily_tool = get_tavily_tool()
        
        if not tavily_tool:
            return {
                "source": "fallback",
                "message": "Tavily API not configured, using basic analysis"
            }
        
        query = f"{place_name} {city} reviews ratings customer feedback"
        results = tavily_tool.invoke({"query": query})
        
        return {
            "source": "tavily",
            "place_name": place_name,
            "results": results,
            "summary": "Review data fetched successfully"
        }
    
    except Exception as e:
        return {
            "source": "error",
            "error": str(e)
        }


# ==========================================
# Simulated Shop Contact Tool
# ==========================================

@tool
def contact_shop_simulation(
    place_name: str,
    place_type: str,
    question_type: str,
    user_budget: Optional[float] = None
) -> str:
    """
    Simulate contacting a shop/business for information.
    Uses LLM to generate realistic shop responses based on place type.
    
    Args:
        place_name: Name of the business
        place_type: Type (gym, restaurant, etc.)
        question_type: What to ask (pricing, availability, features, negotiation)
        user_budget: User's budget constraint if any
    
    Returns:
        Simulated shop response as JSON string
    """
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.messages import SystemMessage, HumanMessage
    
    try:
        llm = ChatGoogleGenerativeAI(
            model=settings.llm_model,
            temperature=0.8,  # More creative for realistic responses
            google_api_key=settings.google_api_key
        )
        
        # Craft realistic simulation prompt
        system_prompt = f"""You are simulating a {place_type} business owner/manager responding to a customer inquiry.
        Business name: {place_name}
        Be professional, realistic, and provide specific details.
        """
        
        user_prompt = f"""Generate a realistic response for this inquiry:
        
Question type: {question_type}
{'Customer budget: ₹' + str(user_budget) if user_budget else 'No budget mentioned'}

Provide response in this JSON format:
{{
    "response_type": "{question_type}",
    "message": "Your detailed response here",
    "pricing_info": {{"monthly": 3000, "quarterly": 8000}} (if applicable),
    "available": true/false,
    "features": ["feature1", "feature2"] (if applicable)
}}
"""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        response = llm.invoke(messages)
        return response.content
    
    except Exception as e:
        # Fallback response
        fallback = {
            "response_type": question_type,
            "message": f"We'd be happy to help! Please call us for details about {question_type}.",
            "available": True
        }
        return json.dumps(fallback)


# ==========================================
# Price Comparison Tool
# ==========================================

@tool
def compare_prices(places: list[dict], budget: Optional[float] = None) -> dict:
    """
    Compare pricing across multiple places.
    
    Args:
        places: List of place dictionaries
        budget: Optional budget constraint
    
    Returns:
        Price comparison analysis
    """
    try:
        analysis = {
            "within_budget": [],
            "above_budget": [],
            "no_price_info": [],
            "average_price": None
        }
        
        prices = []
        
        for place in places:
            # Check if place has pricing info
            if "pricing_info" in place and place["pricing_info"]:
                price = place["pricing_info"].get("monthly") or place["pricing_info"].get("base")
                
                if price:
                    prices.append(price)
                    
                    if budget:
                        if price <= budget:
                            analysis["within_budget"].append({
                                "name": place["name"],
                                "price": price
                            })
                        else:
                            analysis["above_budget"].append({
                                "name": place["name"],
                                "price": price
                            })
            else:
                analysis["no_price_info"].append(place.get("name", "Unknown"))
        
        if prices:
            analysis["average_price"] = sum(prices) / len(prices)
        
        return analysis
    
    except Exception as e:
        return {"error": str(e)}


# ==========================================
# WebScraping.AI Tool (Review Extraction)
# ==========================================

@tool
def fetch_reviews_webscraping_ai(query: str, limit: int = 3) -> list[str]:
    """
    Fetch reviews for a shop using WebScraping.AI to get HTML and Groq to extract reviews.
    
    Args:
        query: Place name and address
        limit: Number of reviews to fetch (default 3)
        
    Returns:
        List of review strings
    """
    if not settings.webscraping_ai_api_key:
        return ["Error: WebScraping.AI API Key not configured."]

    try:
        # 1. Fetch HTML via WebScraping.AI
        search_query = f"reviews for {query}"
        target_url = f"https://www.google.com/search?q={search_query}"
        
        api_url = "https://api.webscraping.ai/html"
        
        params = {
            "api_key": settings.webscraping_ai_api_key,
            "url": target_url,
            "device": "desktop",
            "proxy": "residential",
            "js": "true" # Google needs JS
        }
        
        logger.info(f"Fetching HTML for '{query}' via WebScraping.AI...")
        response = requests.get(api_url, params=params, timeout=60)
        
        if response.status_code != 200:
            logger.error(f"WebScraping.AI Error: {response.status_code} - {response.text}")
            return [f"Error fetching page: {response.status_code}"]
            
        html_content = response.text
        
        # 2. Parse HTML with BeautifulSoup
        soup = BeautifulSoup(html_content, "html.parser")
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
            
        # Get text
        text = soup.get_text(separator=" ", strip=True)
        
        # Truncate text to fit in LLM context (approx 15k chars should be enough for top reviews)
        # Google Search results are noisy, but the reviews are usually in the main content.
        # We'll take the first 20,000 characters.
        text_preview = text[:20000]
        
        # 3. Use Groq to extract reviews
        logger.info("Extracting reviews using Groq...")
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0,
            api_key=settings.groq_api_key
        )
        
        prompt = (
            f"Here is the text content of a Google Search page for '{query}'. "
            f"Extract the top {limit} most relevant and detailed user reviews for this place. "
            "Look for text that looks like user feedback, ratings, or comments. "
            "Return ONLY a raw JSON list of strings. Example: [\"Great coffee!\", \"Service was slow.\"]. "
            "If no reviews are found, return [].\n\n"
            f"PAGE TEXT:\n{text_preview}"
        )
        
        msg = HumanMessage(content=prompt)
        ai_response = llm.invoke([msg])
        content = ai_response.content.strip()
        
        # Clean up markdown
        if content.startswith("```json"):
            content = content.replace("```json", "").replace("```", "")
        elif content.startswith("```"):
            content = content.replace("```", "")
            
        try:
            reviews = json.loads(content)
            if isinstance(reviews, list):
                return reviews[:limit]
            else:
                return [str(reviews)]
        except json.JSONDecodeError:
            return [content]
            
    except Exception as e:
        logger.error(f"Exception in fetch_reviews_webscraping_ai: {str(e)}")
        return [f"Error: {str(e)}"]
