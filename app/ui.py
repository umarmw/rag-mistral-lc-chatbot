import streamlit as st
import requests
import time
from typing import Dict, Any

# Page configuration
st.set_page_config(
    page_title="Chat with Your Documents", 
    page_icon="🧠", 
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        margin-bottom: 2rem;
    }
    .status-indicator {
        padding: 0.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        text-align: center;
    }
    .status-healthy {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    .status-unhealthy {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
    .chat-stats {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #f5c6cb;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Configuration
API_BASE_URL = "http://localhost:8000"
MAX_HISTORY_DISPLAY = 50

def check_server_health() -> Dict[str, Any]:
    """Check if the server is healthy and ready."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            return {"status": "unhealthy", "error": f"HTTP {response.status_code}"}
    except requests.RequestException as e:
        return {"status": "unhealthy", "error": str(e)}

def send_question(question: str, history: list, use_streaming: bool = False, 
                 max_tokens: int = 512, k: int = 3, similarity_threshold: float = 0.7,
                 use_context_limit: bool = True) -> Dict[str, Any]:
    """Send question to the API and get response."""
    try:
        if use_streaming:
            # For now, fall back to regular API for streaming in Streamlit
            # Streamlit doesn't handle SSE well, but we can add this later
            endpoint = f"{API_BASE_URL}/ask"
        else:
            endpoint = f"{API_BASE_URL}/ask"
        
        payload = {
            "question": question,
            "history": history,
            "max_tokens": max_tokens,
            "k": k,
            "similarity_threshold": similarity_threshold,
            "use_context_limit": use_context_limit
        }
        
        response = requests.post(endpoint, json=payload, timeout=30)
        
        if response.status_code == 200:
            return {"success": True, "data": response.json()}
        else:
            error_detail = response.json().get("detail", "Unknown error") if response.headers.get("content-type", "").startswith("application/json") else response.text
            return {"success": False, "error": f"HTTP {response.status_code}: {error_detail}"}
            
    except requests.Timeout:
        return {"success": False, "error": "Request timed out. The server might be processing a complex query."}
    except requests.RequestException as e:
        return {"success": False, "error": f"Network error: {str(e)}"}
    except Exception as e:
        return {"success": False, "error": f"Unexpected error: {str(e)}"}

def clear_cache():
    """Clear the server's retrieval cache."""
    try:
        response = requests.post(f"{API_BASE_URL}/clear-cache", timeout=10)
        return response.status_code == 200
    except:
        return False

def initialize_session_state():
    """Initialize session state variables."""
    if "history" not in st.session_state:
        st.session_state.history = []
    if "chat_turns" not in st.session_state:
        st.session_state.chat_turns = []
    if "total_questions" not in st.session_state:
        st.session_state.total_questions = 0
    if "total_response_time" not in st.session_state:
        st.session_state.total_response_time = 0.0
    if "show_advanced" not in st.session_state:
        st.session_state.show_advanced = False

# Main app
def main():
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.markdown('<div class="main-header">', unsafe_allow_html=True)
    st.title("🧠 Chat with Your Documents")
    st.markdown("Ask questions about your PDF content using advanced RAG technology.")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Sidebar with server status and controls
    with st.sidebar:
        st.header("📊 System Status")
        
        # Health check
        health_status = check_server_health()
        status_class = "status-healthy" if health_status.get("status") == "healthy" else "status-unhealthy"
        status_text = "🟢 Server Online" if health_status.get("status") == "healthy" else "🔴 Server Offline"
        
        st.markdown(f'<div class="{status_class}">{status_text}</div>', unsafe_allow_html=True)
        
        if health_status.get("status") != "healthy":
            st.error(f"Error: {health_status.get('error', 'Unknown error')}")
            st.info("Please make sure the FastAPI server is running on http://localhost:8000")
            return
        
        # Display model info if available
        if "components" in health_status and "vectorstore" in health_status["components"]:
            vectorstore_info = health_status["components"]["vectorstore"]
            if vectorstore_info.get("index_loaded"):
                st.success("✅ Vector Index Loaded")
                with st.expander("Vectorstore Details"):
                    st.write(f"**Total Documents:** {vectorstore_info.get('total_documents', 'N/A')}")
                    st.write(f"**Embedding Model:** {vectorstore_info.get('embedding_model', 'N/A')}")
                    st.write(f"**Embedding Dimension:** {vectorstore_info.get('embedding_dimension', 'N/A')}")
                    st.write(f"**Default K:** {vectorstore_info.get('retrieval_k', 'N/A')}")
                    st.write(f"**Similarity Threshold:** {vectorstore_info.get('similarity_threshold', 'N/A')}")
            else:
                st.error("❌ Vector Index Not Loaded")
        
        if "components" in health_status and "llm" in health_status["components"]:
            model_info = health_status["components"]["llm"]
            if model_info.get("model_loaded"):
                st.success("✅ Model Loaded")
                with st.expander("Model Details"):
                    settings = model_info.get("settings", {})
                    st.write(f"**Context Size:** {settings.get('n_ctx', 'N/A')}")
                    st.write(f"**Max Tokens:** {settings.get('max_tokens', 'N/A')}")
                    st.write(f"**CPU Threads:** {settings.get('n_threads', 'N/A')}")
                    st.write(f"**GPU Layers:** {settings.get('n_gpu_layers', 'N/A')}")
            else:
                st.error("❌ Model Not Loaded")
        
        # Chat statistics
        if st.session_state.total_questions > 0:
            st.header("📈 Chat Statistics")
            st.metric("Total Questions", st.session_state.total_questions)
            avg_time = st.session_state.total_response_time / st.session_state.total_questions
            st.metric("Avg Response Time", f"{avg_time:.2f}s")
        
        # Controls
        st.header("🛠️ Controls")
        
        if st.button("🗑️ Clear Chat History"):
            st.session_state.history = []
            st.session_state.chat_turns = []
            st.session_state.total_questions = 0
            st.session_state.total_response_time = 0.0
            st.success("Chat history cleared!")
            st.rerun()
        
        if st.button("🔍 Test Document Search"):
            test_query = st.text_input("Search query:", placeholder="Enter a search term...")
            if test_query:
                with st.spinner("Searching documents..."):
                    try:
                        response = requests.post(f"{API_BASE_URL}/search", 
                                               params={"query": test_query, "k": 3, "with_scores": True})
                        if response.status_code == 200:
                            search_results = response.json()
                            st.success(f"Found {search_results['count']} documents")
                            
                            for i, result in enumerate(search_results['results'], 1):
                                with st.expander(f"Document {i} (Score: {result.get('similarity_score', 'N/A'):.3f})"):
                                    st.write(result['content'][:500] + "..." if len(result['content']) > 500 else result['content'])
                                    if result.get('metadata'):
                                        st.json(result['metadata'])
                        else:
                            st.error("Search failed")
                    except Exception as e:
                        st.error(f"Search error: {str(e)}")
        
        st.header("📊 System Info")
        if st.button("🔄 Refresh System Info"):
            try:
                response = requests.get(f"{API_BASE_URL}/vectorstore/info")
                if response.status_code == 200:
                    info = response.json()
                    st.json(info)
                else:
                    st.error("Failed to get system info")
            except Exception as e:
                st.error(f"Error: {str(e)}")
        
        # Advanced options
        st.session_state.show_advanced = st.checkbox("Show Advanced Options", st.session_state.show_advanced)
        
        if st.session_state.show_advanced:
            st.header("⚙️ Advanced Options")
            max_tokens = st.slider("Max Response Tokens", 50, 1000, 512)
            retrieval_k = st.slider("Documents to Retrieve (k)", 1, 10, 3)
            similarity_threshold = st.slider("Similarity Threshold", 0.0, 1.0, 0.7, 0.05)
            use_context_limit = st.checkbox("Use Context Length Limiting", True)
            show_processing_time = st.checkbox("Show Processing Time", True)
            show_context_info = st.checkbox("Show Context Info", True)
            show_similarity_scores = st.checkbox("Show Similarity Scores", False)
        else:
            max_tokens = 512
            retrieval_k = 3
            similarity_threshold = 0.7
            use_context_limit = True
            show_processing_time = True
            show_context_info = False
            show_similarity_scores = False
    
    # Main chat interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Display chat history
        if st.session_state.history:
            st.subheader("💬 Conversation")
            
            # Limit displayed history for performance
            display_history = st.session_state.history[-MAX_HISTORY_DISPLAY:]
            if len(st.session_state.history) > MAX_HISTORY_DISPLAY:
                st.info(f"Showing last {MAX_HISTORY_DISPLAY} messages. Total: {len(st.session_state.history)}")
            
            for i, (question, answer_data) in enumerate(display_history):
                with st.chat_message("user"):
                    st.markdown(question)
                
                with st.chat_message("assistant"):
                    if isinstance(answer_data, dict):
                        st.markdown(answer_data.get("answer", "No answer provided"))
                        
                        # Show additional info if enabled
                        if show_processing_time and "processing_time" in answer_data:
                            st.caption(f"⏱️ Response time: {answer_data['processing_time']:.2f}s")
                        
                        if show_context_info and "context_used" in answer_data:
                            context_status = "✅ Used knowledge base" if answer_data["context_used"] else "⚠️ No relevant context found"
                            st.caption(context_status)
                    else:
                        # Handle legacy string responses
                        st.markdown(str(answer_data))
        
        # Chat input
        question = st.chat_input("Ask something about your documents...")
        
        if question:
            # Validate input
            if len(question.strip()) == 0:
                st.error("Please enter a valid question.")
                return
            
            if len(question) > 2000:
                st.error("Question is too long. Please keep it under 2000 characters.")
                return
            
            # Show user message immediately
            with st.chat_message("user"):
                st.markdown(question)
            
            # Show loading spinner and get response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    # Prepare history for API (limit to recent turns to avoid token limits)
                    api_history = st.session_state.chat_turns[-20:] if len(st.session_state.chat_turns) > 20 else st.session_state.chat_turns
                    
                    # Send request
                    result = send_question(
                        question, 
                        api_history,
                        max_tokens=max_tokens,
                        k=retrieval_k,
                        similarity_threshold=similarity_threshold,
                        use_context_limit=use_context_limit
                    )
                    
                    if result["success"]:
                        response_data = result["data"]
                        answer = response_data.get("answer", "No answer provided")
                        
                        # Display answer
                        st.markdown(answer)
                        
                        # Show additional info
                        info_cols = st.columns(4)
                        
                        if show_processing_time and "processing_time" in response_data:
                            with info_cols[0]:
                                st.caption(f"⏱️ {response_data['processing_time']:.2f}s")
                        
                        if show_context_info and "context_used" in response_data:
                            with info_cols[1]:
                                context_status = "✅ KB used" if response_data["context_used"] else "⚠️ No context"
                                st.caption(context_status)
                        
                        if "documents_count" in response_data:
                            with info_cols[2]:
                                st.caption(f"📄 {response_data['documents_count']} docs")
                        
                        if show_similarity_scores and "avg_similarity" in response_data:
                            with info_cols[3]:
                                st.caption(f"🎯 {response_data['avg_similarity']:.2f} sim")
                        
                        # Show context quality if available
                        if show_context_info and "context_quality" in response_data:
                            quality = response_data["context_quality"]
                            quality_color = {"high": "🟢", "moderate": "🟡", "low": "🔴"}.get(quality, "⚪")
                            st.caption(f"{quality_color} Context quality: **{quality}**")
                        
                        if "tokens_estimated" in response_data:
                            with info_cols[2]:
                                st.caption(f"📝 ~{int(response_data['tokens_estimated'])} tokens")
                        
                        # Update session state
                        st.session_state.history.append((question, response_data))
                        st.session_state.chat_turns.append({"role": "user", "content": question})
                        st.session_state.chat_turns.append({"role": "assistant", "content": answer})
                        st.session_state.total_questions += 1
                        st.session_state.total_response_time += response_data.get("processing_time", 0)
                        
                    else:
                        # Handle errors
                        error_msg = result["error"]
                        st.markdown(f'<div class="error-message">❌ <strong>Error:</strong> {error_msg}</div>', unsafe_allow_html=True)
                        
                        # Still add to history for debugging
                        st.session_state.history.append((question, {"answer": f"Error: {error_msg}", "error": True}))
            
            # Rerun to update the interface
            st.rerun()
    
    with col2:
        # Tips and help
        st.subheader("💡 Tips")
        st.markdown("""
        - **Be specific** in your questions
        - **Ask follow-up questions** to dive deeper
        - **Reference previous answers** in your questions
        - Use the **Clear Cache** button if responses seem outdated
        """)
        
        st.subheader("🔧 Troubleshooting")
        st.markdown("""
        - If server is offline, start it with: `python app/server.py`
        - **Timeout errors**: Try shorter, simpler questions
        - **No relevant context**: Your question may not match the document content
        """)

if __name__ == "__main__":
    main()