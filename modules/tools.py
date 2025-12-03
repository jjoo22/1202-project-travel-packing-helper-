from langchain_community.tools import DuckDuckGoSearchRun
from langchain.tools.retriever import create_retriever_tool

class ToolManager:
    def __init__(self, retriever=None):
        """
        Initializes the ToolManager.

        Args:
            retriever: The vector store retriever instance (optional).
        """
        self.retriever = retriever

    def get_tools(self):
        """
        Returns a list of tools available for the agent.
        """
        tools = []

        # 1. Web Search Tool (DuckDuckGo)
        search_tool = DuckDuckGoSearchRun(name="web_search", description="Search the web for current information.")
        tools.append(search_tool)

        # 2. Retriever Tool (Knowledge Base)
        if self.retriever:
            retriever_tool = create_retriever_tool(
                self.retriever,
                "travel_guide_search",
                "Search for travel information, packing tips, and local regulations from the knowledge base."
            )
            tools.append(retriever_tool)

        return tools
