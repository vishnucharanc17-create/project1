from abc import ABC, abstractmethod
from typing import Optional

class MessagingProvider(ABC):
    """Abstract base class for messaging providers"""
    
    @abstractmethod
    def send_message(self, to: str, message: str) -> bool:
        """Send an SMS message"""
        pass

    @abstractmethod
    def get_messages(self) -> list:
        """Get received SMS messages"""
        pass
