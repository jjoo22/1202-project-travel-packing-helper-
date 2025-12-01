from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

class ChatHistoryManager:
    def __init__(self):
        """
        Initializes the chat history manager.
        """
        self.history = ChatMessageHistory()

    def add_user_message(self, message: str):
        """
        Adds a user message to the history.
        """
        self.history.add_user_message(message)

    def add_ai_message(self, message: str):
        """
        Adds an AI message to the history.
        """
        self.history.add_ai_message(message)

    def get_messages(self):
        """
        Returns the list of messages in the history.
        """
        return self.history.messages

    def clear(self):
        """
        Clears the chat history.
        """
        self.history.clear()
