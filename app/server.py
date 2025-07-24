import logging
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.rag_chain import (
    search_documents, 
    get_vectorstore_info, 
    health_check as rag_health_check
)
from models.local_llm import (
    initialize_model, 
    generate_response, 
    generate_response_stream,
    build_rag_prompt,
    health_check,
    settings as llm_settings
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG Chatbot API",
    description="A RAG-based chatbot using Mistral 7B and FAISS",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data Models
class ChatRequest(BaseModel):
    query: str
    memory: Optional[List[str]] = []

@app.on_event("startup")
def startup_event():
    initialize_model()
    logger.info("Model initialized.")

@app.get("/health")
def health():
    return {
        "llm": health_check(),
        "retriever": rag_health_check(),
    }

@app.post("/chat")
def chat(request: ChatRequest):
    try:
        docs = search_documents(request.query, memory=request.memory)
        prompt = build_rag_prompt(request.query, docs, request.memory)
        result = generate_response(prompt)
        return {"response": result}
    except Exception as e:
        logger.error(f"Chat failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/stream")
def chat_stream(request: ChatRequest):
    try:
        docs = search_documents(request.query, memory=request.memory)
        prompt = build_rag_prompt(request.query, docs, request.memory)
        stream = generate_response_stream(prompt)
        return StreamingResponse(stream, media_type="text/event-stream")
    except Exception as e:
        logger.error(f"Streaming chat failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/info")
def info():
    return get_vectorstore_info()
