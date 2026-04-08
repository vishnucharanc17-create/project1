import logging
import requests
from .base import MessagingProvider

logger = logging.getLogger(__name__)

class SMSMobileAPIProvider(MessagingProvider):
    """Provider for SMSMobileAPI"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key

    def send_message(self, to: str, message: str) -> bool:
        try:
            from smsmobileapi import SMSSender
            
            sms = SMSSender(api_key=self.api_key)
            response = sms.send_message(to=to, message=message)
            
            logger.info(f"FULL SMSMobileAPI RESPONSE: {response}")
            
            # Check success
            if response:
                result = response.get("result", response) # Fallback to response if result key missing
                if isinstance(result, dict):
                    # Check for error code 0 (success)
                    if result.get("error") == 0 or result.get("error") == "0":
                        return True
                    if result.get("success") is True:
                        return True
                    if str(result.get("sent")) == "1":
                        return True
            
            return False
            
        except ImportError:
            logger.error("smsmobileapi package not found. Please run `pip install smsmobileapi`")
            return False
        except Exception as e:
            logger.error(f"Error sending SMS via SMSMobileAPI: {str(e)}")
            return False

    def get_messages(self) -> list:
        try:
            from smsmobileapi import SMSSender
            
            sms = SMSSender(api_key=self.api_key)
            messages = sms.get_received_messages()
            
            # logger.info(f"RAW RECEIVED MESSAGES: {messages}") # Commented out for privacy
            logger.info(f"Fetched {len(messages) if isinstance(messages, list) else 'unknown'} messages")
            logger.info(f"Type of messages: {type(messages)}")
            
            # Structure seems to be: {'result': {'error': None, 'sms': [...]}}
            if isinstance(messages, dict):
                if "result" in messages:
                    result = messages["result"]
                    if isinstance(result, dict) and "sms" in result:
                        return result["sms"]
                
                # Fallback checks
                if "sms" in messages:
                    return messages["sms"]
            
            # If it's already a list, return it
            if isinstance(messages, list):
                return messages
                
            return []
            
        except ImportError:
            logger.error("smsmobileapi package not found.")
            return []
        except Exception as e:
            logger.error(f"Error getting messages via SMSMobileAPI: {str(e)}")
            return []
