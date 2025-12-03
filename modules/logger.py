import logging
import os
from datetime import datetime

class LoggerManager:
    def __init__(self, log_file="app.log"):
        """
        Initializes the LoggerManager.
        Configures logging to file with timestamp.
        """
        self.log_file = log_file

        # Configure logging
        logging.basicConfig(
            filename=self.log_file,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            force=True # Ensure we override any existing configuration
        )
        self.logger = logging.getLogger("PackyLogger")

    def log_interaction(self, user_input, ai_response):
        """
        Logs the user input and AI response.
        """
        self.logger.info(f"User: {user_input}")
        self.logger.info(f"AI: {ai_response}")

    def log_error(self, error_message):
        """
        Logs an error message.
        """
        self.logger.error(f"Error: {error_message}")
