import logging
from app.config import settings
from .base import MessagingProvider
from .smsmobileapi import SMSMobileAPIProvider

logger = logging.getLogger(__name__)

class MessagingService:
    """Factory for messaging providers"""
    
    _instance = None
    
    @classmethod
    def get_provider(cls) -> MessagingProvider:
        if cls._instance:
            return cls._instance
            
        provider_type = settings.messaging_provider
        
        if provider_type == "smsmobileapi":
            if not settings.smsmobileapi_key:
                logger.warning("SMSMobileAPI key not configured")
            cls._instance = SMSMobileAPIProvider(api_key=settings.smsmobileapi_key)
        else:
            # Fallback or other providers
            logger.warning(f"Unknown messaging provider: {provider_type}")
            # For now, return SMSMobileAPIProvider as default if key exists, else None
            if settings.smsmobileapi_key:
                cls._instance = SMSMobileAPIProvider(api_key=settings.smsmobileapi_key)
            
        return cls._instance

    @classmethod
    def send_message(cls, to: str, message: str) -> bool:
        provider = cls.get_provider()
        if provider:
            return provider.send_message(to, message)
        logger.error("No messaging provider available")
        return False

    @classmethod
    def get_messages(cls) -> list:
        provider = cls.get_provider()
        if provider:
            return provider.get_messages()
        logger.error("No messaging provider available")
        return []


# Helper function for easy import
def get_messaging_service():
    """Get the messaging service provider instance"""
    return MessagingService.get_provider()
