import logging
import time
from typing import Iterator, Optional
from llama_cpp import Llama
from pydantic_settings import BaseSettings
from functools import lru_cache

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMSettings(BaseSettings):
    model_path: str = "./models/mistral-7b-instruct-v0.2.Q5_K_M.gguf"
    n_ctx: int = 4096
    n_threads: int = 8
    n_gpu_layers: int = 0
    max_tokens: int = 512
    max_dialog_tokens: int = 2048
    
    class Config:
        env_prefix = "LLM_"

settings = LLMSettings()

# Global model instance
llm: Optional[Llama] = None

def initialize_model():
    """Initialize the LLM model. Should be called once at startup."""
    global llm
    try:
        logger.info(f"Loading model from {settings.model_path}")
        start_time = time.time()
        
        llm = Llama(
            model_path=settings.model_path,
            n_ctx=settings.n_ctx,
            n_threads=settings.n_threads,
            n_gpu_layers=settings.n_gpu_layers,
            verbose=False  # Reduce llama.cpp logging
        )
        
        load_time = time.time() - start_time
        logger.info(f"Model loaded successfully in {load_time:.2f} seconds")
        
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise

def estimate_tokens(text: str) -> int:
    """Rough token estimation (1 token ≈ 0.75 words)"""
    # Fix: Ensure we always return an integer
    if not isinstance(text, str):
        text = str(text)
    word_count = len(text.split())
    # Use round() to get proper integer conversion
    token_estimate = round(word_count * 1.3)
    # Ensure we return an int type (not numpy.int64 or similar)
    return int(token_estimate)

def trim_conversation_history(history: list, max_history_tokens: int = 1000) -> list:
    """Trim conversation history to fit within token budget."""
    if not history:
        return []
    
    trimmed_history = []
    token_count = 0
    
    # Process history in reverse to keep most recent conversations
    for turn in reversed(history):
        turn_tokens = estimate_tokens(turn.get('content', ''))
        if token_count + turn_tokens > max_history_tokens:
            break
        trimmed_history.insert(0, turn)
        token_count += turn_tokens
    
    if len(trimmed_history) < len(history):
        logger.info(f"Trimmed history from {len(history)} to {len(trimmed_history)} turns")
    
    return trimmed_history

def build_rag_prompt(context: str, history: list, question: str, documents_metadata: list = None) -> str:
    """Build a well-structured RAG prompt with enhanced context information."""
    # Trim history to prevent context overflow
    max_context_tokens = estimate_tokens(context)
    remaining_tokens = settings.max_dialog_tokens - max_context_tokens - 300  # Increased buffer
    
    trimmed_history = trim_conversation_history(history, max(remaining_tokens, 500))
    
    # Build context section with metadata if available
    context_section = "[CONTEXT]\n"
    if documents_metadata and len(documents_metadata) > 0:
        context_section += f"Found {len(documents_metadata)} relevant documents:\n\n"
        for i, doc_meta in enumerate(documents_metadata, 1):
            similarity = doc_meta.get('similarity_score', 'N/A')
            if isinstance(similarity, float):
                similarity = f"{similarity:.2f}"
            context_section += f"Document {i} (Relevance: {similarity}):\n{doc_meta.get('content', '')}\n\n"
    else:
        context_section += f"{context}\n\n"
    
    # Determine context quality
    context_quality = "high" if documents_metadata and any(
        doc.get('similarity_score', 0) > 0.8 for doc in documents_metadata
    ) else "moderate"
    
    prompt = f"""You are a knowledgeable assistant. Use the provided context to answer questions accurately and comprehensively.

{context_section}

INSTRUCTIONS:
- Base your answer primarily on the context provided above
- If the context has high relevance, provide a detailed answer
- If the context has moderate relevance, acknowledge any limitations
- If the context doesn't contain relevant information, state this clearly
- Cite specific information from the documents when possible
- Current context quality: {context_quality}

[CONVERSATION HISTORY]
"""
    
    # Add conversation history
    for turn in trimmed_history:
        role = "Human" if turn.get("role") == "user" else "Assistant"
        content = turn.get("content", "").strip()
        if content:
            prompt += f"{role}: {content}\n"
    
    prompt += f"Human: {question}\nAssistant: "
    
    # Log prompt length for monitoring
    prompt_tokens = estimate_tokens(prompt)
    logger.info(f"Generated prompt with ~{prompt_tokens} tokens, context quality: {context_quality}")
    
    if prompt_tokens > settings.n_ctx - settings.max_tokens:
        logger.warning(f"Prompt may exceed context window ({prompt_tokens} + {settings.max_tokens} > {settings.n_ctx})")
    
    return prompt

@lru_cache(maxsize=50)
def get_stop_sequences() -> list:
    """Get stop sequences for text generation."""
    return ["</s>", "###", "Human:", "Assistant:", "\n\nHuman:", "\n\nAssistant:"]

def generate_response(prompt: str, max_tokens: Optional[int] = None) -> str:
    """Generate response using the local LLM."""
    if llm is None:
        raise RuntimeError("Model not initialized. Call initialize_model() first.")
    
    max_tokens = max_tokens or settings.max_tokens
    
    try:
        start_time = time.time()
        logger.info("Generating response...")
        
        response = llm(
            prompt,
            max_tokens=max_tokens,
            stop=get_stop_sequences(),
            echo=False,
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            repeat_penalty=1.1
        )
        
        generation_time = time.time() - start_time
        
        if not response or "choices" not in response or not response["choices"]:
            raise ValueError("Empty response from model")
        
        answer = response["choices"][0]["text"].strip()
        
        # Log metrics
        logger.info(f"Response generated in {generation_time:.2f}s, length: {len(answer)} chars")
        
        return answer
        
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        raise RuntimeError(f"Failed to generate response: {str(e)}")

def generate_response_stream(prompt: str, max_tokens: Optional[int] = None) -> Iterator[str]:
    """Generate streaming response using the local LLM."""
    if llm is None:
        raise RuntimeError("Model not initialized. Call initialize_model() first.")
    
    max_tokens = max_tokens or settings.max_tokens
    
    try:
        logger.info("Starting streaming response generation...")
        
        stream = llm(
            prompt,
            max_tokens=max_tokens,
            stop=get_stop_sequences(),
            echo=False,
            stream=True,
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            repeat_penalty=1.1
        )
        
        for chunk in stream:
            if chunk and "choices" in chunk and chunk["choices"]:
                token = chunk["choices"][0].get("text", "")
                if token:
                    yield token
                    
    except Exception as e:
        logger.error(f"Error in streaming response: {str(e)}")
        raise RuntimeError(f"Failed to generate streaming response: {str(e)}")

# Health check function
def health_check() -> dict:
    """Check if the model is loaded and ready."""
    return {
        "model_loaded": llm is not None,
        "settings": {
            "model_path": settings.model_path,
            "n_ctx": settings.n_ctx,
            "max_tokens": settings.max_tokens,
            "n_threads": settings.n_threads,
            "n_gpu_layers": settings.n_gpu_layers
        }
    }