import streamlit as st
import os
from modules.llm import LLMManager
from modules.vector_store import VectorStoreManager
from modules.agent import PackyAgent
from modules.history import ChatHistoryManager
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page Config
st.set_page_config(page_title="Packy - Travel Packing Helper", page_icon="🧳")

st.title("🧳 Packy: Your Travel Packing Assistant")

# Initialize Session State
if "history_manager" not in st.session_state:
    st.session_state.history_manager = ChatHistoryManager()

if "vector_store_manager" not in st.session_state:
    st.session_state.vector_store_manager = VectorStoreManager()
    # Attempt to load and index documents on startup
    with st.spinner("Loading knowledge base..."):
        st.session_state.vector_store_manager.load_and_index()

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
if prompt := st.chat_input("Where are you going?"):
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Add to history
    st.session_state.history_manager.add_user_message(prompt)

    # Get response
    with st.chat_message("assistant"):
        with st.spinner("Packy is thinking..."):
            try:
                chat_history = st.session_state.history_manager.get_messages()
                response = st.session_state.agent.get_response(prompt, chat_history)
                st.markdown(response)

                # Add to history
                st.session_state.history_manager.add_ai_message(response)
            except Exception as e:
                st.error(f"An error occurred: {e}")

# Sidebar for additional controls (optional)
with st.sidebar:
    st.header("Settings")
    if st.button("Clear History"):
        st.session_state.history_manager.clear()
        st.rerun()

    st.header("Knowledge Base")
    if st.button("Reload Data"):
        with st.spinner("Reloading..."):
            st.session_state.vector_store_manager.load_and_index()
            # Re-initialize agent with new retriever
            retriever = st.session_state.vector_store_manager.get_retriever()
            llm = st.session_state.llm_manager.get_llm()
            st.session_state.agent = PackyAgent(llm, retriever)
        st.success("Data reloaded!")
