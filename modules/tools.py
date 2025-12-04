from langchain_community.tools import DuckDuckGoSearchRun
from langchain.tools.retriever import create_retriever_tool

class ToolManager:
    def __init__(self, retriever):
        self.retriever = retriever

    def get_tools(self):
        # 1. RAG 검색 도구 (내 지식)
        retriever_tool = create_retriever_tool(
            self.retriever,
            "packy_knowledge_base",
            "여행 짐 싸기에 대한 기본 준비물, 국가별 꿀팁, 항공 규정 등을 검색할 때 사용합니다."
        )

        # 2. 웹 검색 도구 (실시간 정보)
        search_tool = DuckDuckGoSearchRun(
            name="web_search",
            description="현재 날씨, 환율, 최신 여행 정보 등을 인터넷에서 검색할 때 사용합니다."
        )

        return [retriever_tool, search_tool]