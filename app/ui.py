import streamlit as st
import requests
from typing import Dict, Any

API_BASE_URL = "http://localhost:8000/chat"

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="wide")
st.title("📚 Retrieval-Augmented Chatbot")

if "chat_turns" not in st.session_state:
    st.session_state.chat_turns = []

# Function to send question to backend and get response
def send_question(question: str, history: list, use_streaming: bool = False, 
                 max_tokens: int = 512, k: int = 3, similarity_threshold: float = 0.7,
                 use_context_limit: bool = True) -> Dict[str, Any]:
    try:
        endpoint = f"{API_BASE_URL}/stream" if use_streaming else f"{API_BASE_URL}"

        payload = {
            "query": question,
            "memory": [turn["content"] for turn in history if "content" in turn],
            "max_tokens": max_tokens,
            "k": k,
            "similarity_threshold": similarity_threshold,
            "use_context_limit": use_context_limit
        }

        response = requests.post(endpoint, json=payload, timeout=60)

        if response.status_code == 200:
            return {"success": True, "data": response.json()}
        else:
            content_type = response.headers.get("content-type", "")
            error_detail = response.json().get("detail", "Unknown error") if content_type.startswith("application/json") else response.text
            return {"success": False, "error": f"HTTP {response.status_code}: {error_detail}"}

    except requests.Timeout:
        return {"success": False, "error": "Request timed out. The server might be processing a complex query."}
    except requests.RequestException as e:
        return {"success": False, "error": f"Network error: {str(e)}"}
    except Exception as e:
        return {"success": False, "error": f"Unexpected error: {str(e)}"}

# Chat UI
with st.form("chat_form"):
    question = st.text_input("You:", "", placeholder="Ask me something based on documents...")
    submitted = st.form_submit_button("Send")

if submitted and question.strip():
    st.session_state.chat_turns.append({"role": "user", "content": question.strip()})
    with st.spinner("Thinking..."):
        result = send_question(
            question.strip(), 
            st.session_state.chat_turns,
            use_streaming=False
        )

    if result["success"]:
        answer = result["data"].get("response", "(No response)")
        st.session_state.chat_turns.append({"role": "assistant", "content": answer})
    else:
        st.error(result["error"])

# Display chat history
for turn in st.session_state.chat_turns:
    if turn["role"] == "user":
        st.markdown(f"**You:** {turn['content']}")
    else:
        st.markdown(f"**Assistant:** {turn['content']}")
