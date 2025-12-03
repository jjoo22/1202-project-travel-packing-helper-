import streamlit as st
import os
from modules.llm import LLMManager
from modules.vector_store import VectorStoreManager
from modules.agent import PackyAgent
from modules.history import ChatHistoryManager
from modules.logger import LoggerManager
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page Config
st.set_page_config(page_title="Packy - 여행 짐 싸기 도우미", page_icon="🧳")

st.title("🧳 Packy: 당신의 여행 짐 싸기 도우미")

# Initialize Session State
if "history_manager" not in st.session_state:
    st.session_state.history_manager = ChatHistoryManager()

if "logger_manager" not in st.session_state:
    st.session_state.logger_manager = LoggerManager()

if "vector_store_manager" not in st.session_state:
    st.session_state.vector_store_manager = VectorStoreManager()
    # Attempt to load and index documents on startup
    with st.spinner("지식 베이스를 로드하고 있습니다..."):
        try:
            st.session_state.vector_store_manager.load_and_index()
        except Exception as e:
            st.error(f"데이터 로드 중 오류 발생: {e}")
            st.session_state.logger_manager.log_error(f"Vector Store Load Error: {e}")

if "llm_manager" not in st.session_state:
    st.session_state.llm_manager = LLMManager()

if "agent" not in st.session_state:
    llm = st.session_state.llm_manager.get_llm()
    retriever = st.session_state.vector_store_manager.get_retriever()
    st.session_state.agent = PackyAgent(llm, retriever)

# Display Chat History
for msg in st.session_state.history_manager.get_messages():
    if msg.type == "human":
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif msg.type == "ai":
        with st.chat_message("assistant"):
            st.markdown(msg.content)

# User Input
if prompt := st.chat_input("어디로 여행을 가시나요? 질문을 입력해주세요."):
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Add to history
    st.session_state.history_manager.add_user_message(prompt)

    # Get response
    with st.chat_message("assistant"):
        with st.spinner("Packy가 생각 중입니다..."):
            try:
                chat_history = st.session_state.history_manager.get_messages()
                response = st.session_state.agent.get_response(prompt, chat_history)
                st.markdown(response)

                # Add to history
                st.session_state.history_manager.add_ai_message(response)

                # Log interaction
                st.session_state.logger_manager.log_interaction(prompt, response)

            except Exception as e:
                error_msg = f"죄송합니다. 오류가 발생했습니다: {e}"
                st.error(error_msg)
                st.session_state.logger_manager.log_error(f"Agent Response Error: {e}")

# Sidebar for additional controls (optional)
with st.sidebar:
    st.header("설정")
    if st.button("대화 내용 지우기"):
        st.session_state.history_manager.clear()
        st.rerun()

    st.header("지식 베이스")
    if st.button("데이터 새로고침"):
        with st.spinner("새로고침 중..."):
            try:
                st.session_state.vector_store_manager.load_and_index()
                # Re-initialize agent with new retriever
                retriever = st.session_state.vector_store_manager.get_retriever()
                llm = st.session_state.llm_manager.get_llm()
                st.session_state.agent = PackyAgent(llm, retriever)
                st.success("데이터가 성공적으로 로드되었습니다!")
            except Exception as e:
                st.error(f"데이터 로드 실패: {e}")
                st.session_state.logger_manager.log_error(f"Reload Data Error: {e}")
