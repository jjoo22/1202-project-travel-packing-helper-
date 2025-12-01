from langchain_openai import ChatOpenAI
import os

class LLMManager:
    def __init__(self, model_name="gpt-3.5-turbo", temperature=0.7):
        """
        Initializes the OpenAI LLM.
        """
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=os.getenv("OPENAI_API_KEY") # Ensure OPENAI_API_KEY is set in environment
        )

    def get_llm(self):
        """
        Returns the initialized LLM instance.
        """
        return self.llm
