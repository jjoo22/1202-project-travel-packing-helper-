from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
from langchain_core.prompts import PromptTemplate
from modules.tools import ToolManager

class PackyAgent:
    def __init__(self, llm, retriever):
        """
        Initializes the PackyAgent as a ReAct Agent.

        Args:
            llm: The language model instance.
            retriever: The vector store retriever.
        """
        self.llm = llm
        self.tool_manager = ToolManager(retriever)
        self.tools = self.tool_manager.get_tools()
        self.agent_executor = self._build_agent()

    def _build_agent(self):
        """
        Builds the ReAct agent executor.
        """
        # Pull the react prompt
        # We can use a custom prompt to enforce Korean

        template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question.

IMPORTANT: The "Final Answer" MUST be in polite Korean (존댓말).

Begin!

Question: {input}
Thought:{agent_scratchpad}"""

        prompt = PromptTemplate.from_template(template)

        agent = create_react_agent(self.llm, self.tools, prompt)

        agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True
        )

        return agent_executor

    def get_response(self, user_input, chat_history=None):
        """
        Generates a response for the user input using the Agent.

        Args:
            user_input (str): The user's query.
            chat_history (list): List of message objects (not directly used by simple ReAct,
                                 but kept for signature compatibility or future memory integration).

        Returns:
            str: The agent's response.
        """
        if not self.agent_executor:
            return "죄송합니다. 에이전트가 올바르게 초기화되지 않았습니다."

        try:
            # We can optionally pass chat_history into the prompt if we modified the prompt to accept it
            # For this simple ReAct implementation, we focus on the current input.
            # To handle conversation history properly with ReAct, we would need to pass it as context
            # or use a Conversational ReAct agent.
            # For now, we will just answer the current question as per the "ReAct Agent" requirement.

            result = self.agent_executor.invoke({"input": user_input})
            return result["output"]
        except Exception as e:
            return f"오류가 발생했습니다: {str(e)}"
