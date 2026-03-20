import streamlit as st
import os
import sys
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Load environment variables
load_dotenv()

# Fix import path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from models.llm import get_chatgroq_model


def get_chat_response(chat_model, messages, system_prompt):
    """Get response from the chat model"""
    try:
        formatted_messages = []

        # Add system prompt if provided
        if system_prompt:
            formatted_messages.append(SystemMessage(content=system_prompt))

        # Add chat history
        for msg in messages:
            if msg["role"] == "user":
                formatted_messages.append(HumanMessage(content=msg["content"]))
            else:
                formatted_messages.append(AIMessage(content=msg["content"]))

        # Get response
        response = chat_model.invoke(formatted_messages)
        return response.content

    except Exception as e:
        return f"❌ Error getting response: {str(e)}"


def instructions_page():
    """Instructions page"""
    st.title("📘 The Chatbot Blueprint")

    st.markdown("""
    ## 🔧 Installation
    ```bash
    pip install -r requirements.txt
    ```

    ## 🔑 API Key Setup
    Create a `.env` file in your root folder:
    ```
    GROQ_API_KEY=your_api_key_here
    ```

    ## 🤖 Supported Models
    - llama3-70b-8192 ✅
    - llama3-8b-8192
    - mixtral-8x7b-32768

    ## 🚀 Usage
    1. Go to Chat page
    2. Start chatting

    ## ⚠️ Troubleshooting
    - API key missing → check `.env`
    - Model error → verify model name
    """)


def chat_page():
    """Main chatbot page"""
    st.title("🤖 AI ChatBot")

    # Check API key
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("🚫 GROQ_API_KEY not found. Please set it in .env file.")
        st.stop()

    # Load model
    chat_model = get_chatgroq_model()

    # Sidebar settings
    st.sidebar.subheader("⚙️ Settings")
    system_prompt = st.sidebar.text_area(
        "System Prompt (optional)",
        placeholder="You are a helpful AI assistant..."
    )

    # Initialize session
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Show chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input box
    prompt = st.chat_input("💬 Type your message here...")

    if prompt:
        # Save user message
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking... 🤔"):
                response = get_chat_response(
                    chat_model,
                    st.session_state.messages,
                    system_prompt
                )
                st.markdown(response)

        # Save response
        st.session_state.messages.append({"role": "assistant", "content": response})


def main():
    st.set_page_config(
        page_title="AI ChatBot",
        page_icon="🤖",
        layout="wide"
    )

    # Sidebar navigation
    with st.sidebar:
        st.title("📂 Navigation")
        page = st.radio("Go to:", ["Chat", "Instructions"])

        st.divider()

        if page == "Chat":
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()

    # Routing
    if page == "Instructions":
        instructions_page()
    elif page == "Chat":
        chat_page()


if __name__ == "__main__":
    main()