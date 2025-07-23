import logging
import time
from typing import List, Dict, Any
from functools import lru_cache

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, validator

from app.rag_chain import (
    get_retriever, 
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

# Initialize FastAPI app
app = FastAPI(
    title="RAG Chatbot API",
    description="A RAG-based chatbot using Mistral 7B and FAISS",
    version="1.0.0"
)

# Global variables
retriever = None

# CORS configuration
origins = [
    "http://localhost:3000",  # React default
    "http://localhost:8080",  # Vue default
    "http://127.0.0.1:3000",
    "http://127.0.0.1:8080",
    # Add your frontend URLs here
    # "*"  # Use only for development
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Pydantic models
class ConversationTurn(BaseModel):
    role: str
    content: str
    
    @validator('role')
    def validate_role(cls, v):
        if v not in ['user', 'assistant']:
            raise ValueError('Role must be either "user" or "assistant"')
        return v
    
    @validator('content')
    def validate_content(cls, v):
        if not v.strip():
            raise ValueError('Content cannot be empty')
        return v.strip()

class Query(BaseModel):
    question: str
    history: List[ConversationTurn] = []
    max_tokens: int = None
    k: int = 3  # Number of documents to retrieve
    similarity_threshold: float = 0.7  # Similarity threshold for filtering
    use_context_limit: bool = True  # Whether to use context length limiting
    
    @validator('question')
    def validate_question(cls, v):
        if not v.strip():
            raise ValueError('Question cannot be empty')
        if len(v) > 2000:
            raise ValueError('Question too long (max 2000 characters)')
        return v.strip()
    
    @validator('history')
    def validate_history(cls, v):
        if len(v) > 50:  # Limit conversation length
            logger.warning(f"History too long ({len(v)} turns), keeping only recent 50")
            return v[-50:]
        return v
    
    @validator('max_tokens')
    def validate_max_tokens(cls, v):
        if v is not None and (v < 1 or v > 2048):
            raise ValueError('max_tokens must be between 1 and 2048')
        return v
    
    @validator('k')
    def validate_k(cls, v):
        if v < 1 or v > 10:
            raise ValueError('k must be between 1 and 10')
        return v
    
    @validator('similarity_threshold')
    def validate_similarity_threshold(cls, v):
        if v < 0.0 or v > 1.0:
            raise ValueError('similarity_threshold must be between 0.0 and 1.0')
        return v

class ChatResponse(BaseModel):
    answer: str
    processing_time: float
    context_used: bool = True
    tokens_estimated: int = 0
    documents_count: int = 0
    avg_similarity: float = 0.0
    context_quality: str = "unknown"

class ErrorResponse(BaseModel):
    error: str
    details: str = None

# RAG Service Class
class RAGService:
    def __init__(self, retriever):
        self.retriever = retriever
        
    @lru_cache(maxsize=100)
    def get_cached_context(self, question: str, k: int = 3) -> str:
        """Get cached context for frequently asked questions."""
        try:
            retrieved = self.retriever.get_relevant_documents(question, k=k)
            
            # Filter documents by relevance if score is available
            relevant_docs = []
            for doc in retrieved:
                # Some retrievers provide a score attribute
                score = getattr(doc, 'score', None)
                if score is None or score > 0.5:  # Keep if no score or good score
                    relevant_docs.append(doc)
            
            if not relevant_docs:
                logger.warning("No relevant documents found for question")
                return "No relevant context found in the knowledge base."
            
            context = "\n\n".join([doc.page_content for doc in relevant_docs[:3]])  # Limit to top 3
            logger.info(f"Retrieved {len(relevant_docs)} relevant documents")
            
            return context
            
        except Exception as e:
            logger.error(f"Error retrieving context: {str(e)}")
            return "Error retrieving context from knowledge base."
    
    def generate_answer(self, question: str, history: List[Dict], max_tokens: int = None, 
                       k: int = 3, similarity_threshold: float = 0.7, 
                       use_context_limit: bool = True) -> Dict[str, Any]:
        """Generate answer using enhanced RAG pipeline."""
        start_time = time.time()
        
        try:
            # Get enhanced retriever with custom parameters
            retriever = get_retriever(k=k, similarity_threshold=similarity_threshold)
            
            # Get documents with enhanced features
            if use_context_limit:
                docs = retriever.get_relevant_documents_with_context_limit(question)
            else:
                docs = retriever.get_relevant_documents(question, k=k)
            
            context_used = len(docs) > 0
            
            # Extract context and metadata
            documents_metadata = []
            context_parts = []
            
            for doc in docs:
                doc_info = {
                    'content': doc.page_content,
                    'similarity_score': doc.metadata.get('similarity_score', 0.0),
                    'source': doc.metadata.get('source', 'unknown'),
                    'trimmed': doc.metadata.get('trimmed', False)
                }
                documents_metadata.append(doc_info)
                context_parts.append(doc.page_content)
            
            context = "\n\n".join(context_parts) if context_parts else "No relevant context found."
            
            # Calculate context quality metrics
            avg_similarity = 0.0
            if documents_metadata:
                similarities = [doc.get('similarity_score', 0.0) for doc in documents_metadata]
                avg_similarity = sum(similarities) / len(similarities)
            
            context_quality = "high" if avg_similarity > 0.8 else "moderate" if avg_similarity > 0.6 else "low"
            
            # Convert ConversationTurn objects to dicts for prompt building
            history_dicts = []
            for turn in history:
                if isinstance(turn, ConversationTurn):
                    history_dicts.append({"role": turn.role, "content": turn.content})
                else:
                    history_dicts.append(turn)
            
            # Build enhanced prompt with metadata
            prompt = build_rag_prompt(context, history_dicts, question, documents_metadata)
            
            # Generate response
            answer = generate_response(prompt, max_tokens)
            
            processing_time = time.time() - start_time
            
            return {
                "answer": answer,
                "processing_time": processing_time,
                "context_used": context_used,
                "tokens_estimated": len(answer.split()) * 1.3,
                "documents_count": len(docs),
                "avg_similarity": round(avg_similarity, 3),
                "context_quality": context_quality
            }
            
        except Exception as e:
            logger.error(f"Error in enhanced RAG pipeline: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to generate response: {str(e)}")

# Global RAG service
rag_service = None

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize models and services on startup."""
    global retriever, rag_service
    
    try:
        logger.info("Starting up RAG Chatbot API...")
        
        # Initialize LLM
        logger.info("Initializing language model...")
        initialize_model()
        
        # Initialize retriever
        logger.info("Initializing document retriever...")
        retriever = get_retriever()
        
        # Initialize RAG service
        rag_service = RAGService(retriever)
        
        logger.info("Startup completed successfully!")
        
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down RAG Chatbot API...")

# API Routes
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "RAG Chatbot API",
        "version": "1.0.0",
        "endpoints": {
            "ask": "/ask",
            "ask_stream": "/ask-stream", 
            "health": "/health"
        }
    }

@app.get("/health")
async def health():
    """Enhanced health check endpoint."""
    try:
        model_health = health_check()
        rag_health = rag_health_check()
        vectorstore_info = get_vectorstore_info()
        
        overall_status = "healthy" if (
            model_health["model_loaded"] and 
            rag_health["status"] == "healthy"
        ) else "unhealthy"
        
        return {
            "status": overall_status,
            "timestamp": time.time(),
            "components": {
                "llm": model_health,
                "rag": rag_health,
                "vectorstore": vectorstore_info
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {"status": "unhealthy", "error": str(e)}

@app.post("/ask", response_model=ChatResponse)
async def ask_question(query: Query):
    """Enhanced chat endpoint with advanced RAG features."""
    if rag_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        logger.info(f"Processing question: {query.question[:100]}... (k={query.k}, threshold={query.similarity_threshold})")
        
        result = rag_service.generate_answer(
            question=query.question,
            history=query.history,
            max_tokens=query.max_tokens,
            k=query.k,
            similarity_threshold=query.similarity_threshold,
            use_context_limit=query.use_context_limit
        )
        
        return ChatResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in ask endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/ask-stream")
async def ask_question_stream(query: Query):
    """Streaming chat endpoint."""
    if rag_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        logger.info(f"Processing streaming question: {query.question[:100]}...")
        
        # Get context (same as regular endpoint)
        context = rag_service.get_cached_context(query.question)
        
        # Convert history
        history_dicts = []
        for turn in query.history:
            if isinstance(turn, ConversationTurn):
                history_dicts.append({"role": turn.role, "content": turn.content})
            else:
                history_dicts.append(turn)
        
        # Build prompt
        prompt = build_rag_prompt(context, history_dicts, query.question)
        
        def generate():
            try:
                for token in generate_response_stream(prompt, query.max_tokens):
                    yield f"data: {token}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                logger.error(f"Error in streaming: {str(e)}")
                yield f"data: [ERROR] {str(e)}\n\n"
        
        return StreamingResponse(
            generate(),
            media_type="text/plain",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
        )
        
    except Exception as e:
        logger.error(f"Error in streaming endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search")
async def search_documents_endpoint(
    query: str,
    k: int = 3,
    with_scores: bool = True
):
    """Search documents directly without generating chat response."""
    try:
        results = search_documents(query, k=k, with_scores=with_scores)
        return {
            "query": query,
            "results": results,
            "count": len(results)
        }
    except Exception as e:
        logger.error(f"Error in search endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.get("/vectorstore/info")
async def get_vectorstore_info_endpoint():
    """Get information about the loaded vectorstore."""
    try:
        info = get_vectorstore_info()
        return info
    except Exception as e:
        logger.error(f"Error getting vectorstore info: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/clear-cache")
async def clear_cache():
    """Clear the retrieval cache."""
    try:
        if rag_service:
            rag_service.get_cached_context.cache_clear()
        return {"message": "Cache cleared successfully"}
    except Exception as e:
        logger.error(f"Error clearing cache: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to clear cache")

# Error handlers
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    return {"error": "Invalid input", "details": str(exc)}

@app.exception_handler(RuntimeError)
async def runtime_error_handler(request, exc):
    return {"error": "Runtime error", "details": str(exc)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )