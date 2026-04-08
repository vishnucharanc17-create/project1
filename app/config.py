from pydantic_settings import BaseSettings
from typing import Literal


class Settings(BaseSettings):
    """Application configuration with environment variable support"""
    
    # Groq API Keys (replacing Gemini)
    groq_api_key: str
    groq_api_key_2: str = ""
    serp_api_key: str
    tavily_api_key: str = ""  # Optional
    webscraping_ai_api_key: str = "" # For AI-powered Review Extraction
    
    # Application
    environment: Literal["development", "production"] = "development"
    database_path: str = "./data/checkpoints.db"
    log_level: str = "INFO"
    
    # LLM Settings
    llm_model: str = "llama-3.3-70b-versatile"  # Groq's best model
    llm_temperature: float = 0.7
    max_tokens: int = 2048
    
    # Agent Settings
    max_iterations: int = 3
    recursion_limit: int = 50
    
    # SerpStack API
    serp_api_url: str = "http://api.serpstack.com/search"

    # Default SMS routing (gateway-agnostic)
    default_sender_number: str = ""  # Optional; depends on provider
    default_target_number: str = ""  # Fallback recipient if place has no phone (E.164)

    # (Legacy Twilio fields retained for compatibility; unused when messaging_provider=smsmobileapi)
    twilio_account_sid: str = ""
    twilio_auth_token: str = ""
    twilio_messaging_service_sid: str = ""
    twilio_phone_number: str = ""
    twilio_target_number: str = ""

    # Messaging Settings
    messaging_provider: Literal["twilio", "smsmobileapi"] = "smsmobileapi"
    smsmobileapi_key: str = ""  # Set in .env file

    # LangSmith Settings (Optional)
    langchain_tracing_v2: str = "true"
    langchain_endpoint: str = "https://api.smith.langchain.com"
    langchain_api_key: str = ""  # Set in .env file (optional)
    langchain_project: str = "PlaceDiscoverAgent"
    
    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
