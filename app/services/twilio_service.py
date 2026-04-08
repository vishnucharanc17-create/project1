from twilio.rest import Client
from app.config import settings
import logging

logger = logging.getLogger(__name__)

class TwilioService:
    def __init__(self):
        self.client = None
        if settings.twilio_account_sid and settings.twilio_auth_token:
            try:
                self.client = Client(settings.twilio_account_sid, settings.twilio_auth_token)
                logger.info("✅ Twilio Client Initialized")
            except Exception as e:
                logger.error(f"❌ Failed to initialize Twilio Client: {e}")
        else:
            logger.warning("⚠️ Twilio credentials missing. SMS will not be sent.")

    def send_sms(self, to_number: str, message: str) -> bool:
        """
        Sends an SMS using Twilio.
        """
        if not self.client:
            logger.warning("⚠️ Twilio client not initialized. Skipping SMS.")
            return False

        try:
            logger.info(f"📤 Sending SMS to {to_number}...")
            
            # Use Messaging Service if available, else use From number
            kwargs = {
                "to": to_number,
                "body": message
            }
            
            if settings.twilio_messaging_service_sid:
                kwargs["messaging_service_sid"] = settings.twilio_messaging_service_sid
            else:
                kwargs["from_"] = settings.twilio_phone_number

            message = self.client.messages.create(**kwargs)
            
            logger.info(f"✅ SMS Sent! SID: {message.sid}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to send SMS: {e}")
            return False

# Global instance
twilio_service = TwilioService()
