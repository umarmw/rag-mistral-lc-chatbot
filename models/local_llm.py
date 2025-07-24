import logging
import time
from typing import Iterator, Optional, List
from llama_cpp import Llama
from pydantic_settings import BaseSettings

# For local sentence-transformer embeddings
from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------
# LLM Settings and Init
# ---------------------------

class LLMSettings(BaseSettings):
    model_path: str = "./models/mistral-7b-instruct-v0.2.Q5_K_M.gguf"
    n_ctx: int = 4096
    n_threads: int = 8
    n_gpu_layers: int = 0
    max_tokens: int = 512
    max_dialog_tokens: int = 2048
    embedding_model: str = "./models/embeddings/all-MiniLM-L6-v2"  # Local path

    class Config:
        env_prefix = "LLM_"

settings = LLMSettings()

llm: Optional[Llama] = None

def initialize_model():
    global llm
    try:
        logger.info(f"Loading LLM from {settings.model_path}")
        start_time = time.time()
        llm = Llama(
            model_path=settings.model_path,
            n_ctx=settings.n_ctx,
            n_threads=settings.n_threads,
            n_gpu_layers=settings.n_gpu_layers,
        )
        logger.info(f"Model loaded in {time.time() - start_time:.2f}s")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

# ---------------------------
# RAG Prompt Builder
# ---------------------------

def build_rag_prompt(query: str, docs: List[str], memory: Optional[List[str]] = None) -> str:
    context = "\n".join([doc.page_content for doc in docs])
    memory_text = "\n".join(memory[-5:]) if memory else ""
    return f"""You are a helpful assistant. Use the provided context to answer the user's query.

Context:
{context}

Conversation history:
{memory_text}

User: {query}
Assistant:"""

# ---------------------------
# LLM Inference
# ---------------------------

def generate_response(prompt: str) -> str:
    if not llm:
        raise RuntimeError("Model is not initialized.")
    output = llm(prompt, max_tokens=settings.max_tokens)
    return output["choices"][0]["text"].strip()

def generate_response_stream(prompt: str) -> Iterator[str]:
    if not llm:
        raise RuntimeError("Model is not initialized.")
    for chunk in llm(prompt, max_tokens=settings.max_tokens, stream=True):
        yield chunk["choices"][0]["text"]

# ---------------------------
# Embedding Class
# ---------------------------

class LocalEmbedding(Embeddings):
    def __init__(self, model_path: str = settings.embedding_model):
        logger.info(f"Loading sentence-transformer model from: {model_path}")
        self.model = SentenceTransformer(model_path)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, convert_to_tensor=False).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode([text], convert_to_tensor=False)[0].tolist()

# ---------------------------
# Health Check
# ---------------------------

def health_check() -> bool:
    return llm is not None
