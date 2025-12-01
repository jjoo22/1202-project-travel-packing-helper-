from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

class PackyAgent:
    def __init__(self, llm, retriever):
        """
        Initializes the PackyAgent.

        Args:
            llm: The language model instance.
            retriever: The vector store retriever.
        """
        self.llm = llm
        self.retriever = retriever
        self.chain = self._build_chain()

    def _build_chain(self):
        """
        Builds the RAG chain for the agent.
        """

        # 1. Contextualize question prompt
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )

        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                ("placeholder", "{chat_history}"),
                ("human", "{input}"),
            ]
        )

        # History aware retriever
        if self.retriever:
            history_aware_retriever = create_history_aware_retriever(
                self.llm, self.retriever, contextualize_q_prompt
            )
        else:
            # Fallback if no retriever is available (e.g. no documents yet)
            # Depending on logic, might just return llm directly, but for structure we keep similar flow
            history_aware_retriever = None

        # 2. Answer question prompt
        system_prompt = (
            "You are Packy, a helpful travel packing assistant. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            "{context}"
        )

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("placeholder", "{chat_history}"),
                ("human", "{input}"),
            ]
        )

        question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt)

        if history_aware_retriever:
            rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
        else:
            # If no retriever, we can't use retrieval chain.
            # Ideally we should handle this, but for now we assume retriever exists or fail gracefully.
            # A simple fallback:
            rag_chain = question_answer_chain # Note: this will fail if it expects 'context' and we don't provide it

        return rag_chain

    def get_response(self, user_input, chat_history):
        """
        Generates a response for the user input.

        Args:
            user_input (str): The user's query.
            chat_history (list): List of message objects.

        Returns:
            str: The agent's response.
        """
        if not self.chain:
            return "I am not initialized correctly."

        # If retriever is None, we need to handle context differently or skip retrieval
        if self.retriever is None:
             # Simple invocation without retrieval
             # Using a basic LLM call if no docs are loaded
             messages = chat_history + [("human", user_input)]
             response = self.llm.invoke(messages)
             return response.content

        response = self.chain.invoke({"input": user_input, "chat_history": chat_history})
        return response["answer"]
