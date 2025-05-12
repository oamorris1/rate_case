import streamlit as st
import requests
import json
import uuid
from datetime import datetime

# Configure the app
st.set_page_config(
    page_title="Chat with Documents",
    page_icon="📚",
    layout="wide"
)

# API endpoint configuration
API_URL = "http://localhost:8000"  # Change to FastAPI server URL in azure webapp after we push it

# Initialize session state variables
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "available_models" not in st.session_state:
    # get available models from API
    try:
        response = requests.get(f"{API_URL}/models")
        if response.status_code == 200:
            models_data = response.json()
            st.session_state.available_models = models_data.get("models", ["gpt-4o", "gpt4o-mini"])
            st.session_state.default_model = models_data.get("default", "gpt-4o")
        else:
            # Fallback if API call fails
            st.session_state.available_models = ["gpt-4o", "gpt4o-mini"]
            st.session_state.default_model = "gpt-4o"
    except Exception as e:
        st.error(f"Could not fetch available models: {str(e)}")
        st.session_state.available_models = ["gpt-4o", "gpt4o-mini"]
        st.session_state.default_model = "gpt-4o"

if "selected_model" not in st.session_state:
    st.session_state.selected_model = st.session_state.default_model


def query_api(question, session_id, model):
    """Send a query to the FastAPI backend"""
    try:
        payload = {
            "query": question,
            "session_id": session_id,
            "model": model 
        }
        
        response = requests.post(
            f"{API_URL}/chat",
            json=payload
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error from API: {response.status_code} - {response.text}")
            return {"answer": f"Error: Could not get response from API (Status: {response.status_code})"}
    except Exception as e:
        st.error(f"Exception when calling API: {str(e)}")
        return {"answer": f"Error: {str(e)}"}

# App title and description
st.title("📚 Chat with Documents")
st.subheader("Ask questions about your documents")

# sidebar stuff
with st.sidebar:
    st.header("Settings")
    
    # choose model
    selected_model = st.selectbox(
        "Select Model",
        options=st.session_state.available_models,
        index=st.session_state.available_models.index(st.session_state.selected_model)
    )
    
    if selected_model != st.session_state.selected_model:
        st.session_state.selected_model = selected_model
    
    # Session info
    st.subheader("Session Information")
    st.write(f"Session ID: {st.session_state.session_id}")
    
    # reset conversation
    if st.button("Reset Conversation"):
        st.session_state.chat_history = []
        st.session_state.session_id = str(uuid.uuid4())
        st.success("Conversation has been reset!")

# Display chat messages
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "model" in message and message["role"] == "assistant":
            st.caption(f"Model: {message['model']} | {message.get('timestamp', 'N/A')}")

# Chat input
user_input = st.chat_input("Ask a question about your documents...")

if user_input:
    # Add user message to chat history
    st.session_state.chat_history.append({
        "role": "user",
        "content": user_input
    })
    
    # Display the user message
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Get the response from the API
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = query_api(
                user_input, 
                st.session_state.session_id, 
                st.session_state.selected_model
            )
            
            answer = response.get("answer", "Sorry, I couldn't process your question.")
            st.markdown(answer)
            
            # Add caption with model info
            model_used = response.get("model", st.session_state.selected_model)
            timestamp = response.get("timestamp", datetime.now().isoformat())
            st.caption(f"Model: {model_used} | {timestamp}")
    
    # Add assistant response to chat history
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": response.get("answer", "Sorry, I couldn't process your question."),
        "model": response.get("model", st.session_state.selected_model),
        "timestamp": response.get("timestamp", datetime.now().isoformat())
    })