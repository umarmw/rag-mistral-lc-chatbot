import os
import logging
from typing import Optional, Dict, Any, List
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic_settings import BaseSettings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGSettings(BaseSettings):
    """Configuration for RAG components."""
    index_dir: str = os.path.join(os.path.dirname(__file__), '../faiss_index')
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    retrieval_k: int = 3
    similarity_threshold: float = 0.7
    max_context_length: int = 4000  # Maximum characters in context
    
    class Config:
        env_prefix = "RAG_"

settings = RAGSettings()

class EnhancedRetriever:
    """Enhanced retriever with additional functionality."""
    
    def __init__(self, vectorstore: FAISS, k: int = 3, similarity_threshold: float = 0.7):
        self.vectorstore = vectorstore
        self.k = k
        self.similarity_threshold = similarity_threshold
        self.base_retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    
    def get_relevant_documents(self, query: str, k: Optional[int] = None) -> List[Document]:
        """Get relevant documents with similarity filtering."""
        k = k or self.k
        
        try:
            # Use similarity search with scores to filter by threshold
            docs_with_scores = self.vectorstore.similarity_search_with_score(query, k=k*2)  # Get more to filter
            
            # Filter by similarity threshold
            filtered_docs = []
            for doc, score in docs_with_scores:
                # Lower scores are better in FAISS (distance-based)
                if score <= (1.0 - self.similarity_threshold):  # Convert similarity to distance
                    # Add metadata about the score
                    doc.metadata['similarity_score'] = 1.0 - score
                    doc.metadata['retrieval_query'] = query
                    filtered_docs.append(doc)
            
            # Take only the top k after filtering
            result_docs = filtered_docs[:k]
            
            logger.info(f"Retrieved {len(result_docs)} relevant documents out of {len(docs_with_scores)} candidates")
            
            return result_docs
            
        except Exception as e:
            logger.error(f"Error during document retrieval: {str(e)}")
            # Fallback to basic retrieval
            return self.base_retriever.get_relevant_documents(query)
    
    def get_relevant_documents_with_context_limit(self, query: str, max_length: int = None) -> List[Document]:
        """Get relevant documents while respecting context length limits."""
        max_length = max_length or settings.max_context_length
        
        docs = self.get_relevant_documents(query)
        
        # Trim documents to fit within context limit
        total_length = 0
        trimmed_docs = []
        
        for doc in docs:
            content = doc.page_content
            content_length = len(content)
            
            if total_length + content_length <= max_length:
                trimmed_docs.append(doc)
                total_length += content_length
            else:
                # Try to fit partial content
                remaining_space = max_length - total_length
                if remaining_space > 100:  # Only if we have meaningful space left
                    # Trim the document content
                    trimmed_content = content[:remaining_space-10] + "..."
                    trimmed_doc = Document(
                        page_content=trimmed_content,
                        metadata={**doc.metadata, 'trimmed': True}
                    )
                    trimmed_docs.append(trimmed_doc)
                break
        
        if len(trimmed_docs) < len(docs):
            logger.info(f"Trimmed documents from {len(docs)} to {len(trimmed_docs)} to fit context limit")
        
        return trimmed_docs

# Global variables
_embeddings: Optional[HuggingFaceEmbeddings] = None
_vectorstore: Optional[FAISS] = None
_retriever: Optional[EnhancedRetriever] = None

def get_embeddings() -> HuggingFaceEmbeddings:
    """Get or create embeddings model."""
    global _embeddings
    
    if _embeddings is None:
        try:
            logger.info(f"Loading embeddings model: {settings.embedding_model}")
            _embeddings = HuggingFaceEmbeddings(
                model_name=settings.embedding_model,
                model_kwargs={'device': 'cpu'},  # Explicit CPU usage
                encode_kwargs={'normalize_embeddings': True}  # Normalize for better similarity
            )
            logger.info("Embeddings model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embeddings model: {str(e)}")
            raise
    
    return _embeddings

def load_vectorstore() -> FAISS:
    """Load FAISS vectorstore."""
    global _vectorstore
    
    if _vectorstore is None:
        try:
            if not os.path.exists(settings.index_dir):
                raise FileNotFoundError(f"FAISS index directory not found: {settings.index_dir}")
            
            logger.info(f"Loading FAISS index from: {settings.index_dir}")
            embeddings = get_embeddings()
            
            _vectorstore = FAISS.load_local(
                settings.index_dir, 
                embeddings, 
                allow_dangerous_deserialization=True
            )
            
            # Get some basic info about the vectorstore
            try:
                total_docs = _vectorstore.index.ntotal
                logger.info(f"FAISS index loaded successfully with {total_docs} documents")
            except:
                logger.info("FAISS index loaded successfully")
                
        except Exception as e:
            logger.error(f"Failed to load FAISS index: {str(e)}")
            raise
    
    return _vectorstore

def get_retriever(k: Optional[int] = None, similarity_threshold: Optional[float] = None) -> EnhancedRetriever:
    """Get or create the enhanced retriever."""
    global _retriever
    
    # Use provided values or defaults
    k = k or settings.retrieval_k
    similarity_threshold = similarity_threshold or settings.similarity_threshold
    
    if _retriever is None:
        try:
            vectorstore = load_vectorstore()
            _retriever = EnhancedRetriever(
                vectorstore=vectorstore,
                k=k,
                similarity_threshold=similarity_threshold
            )
            logger.info(f"Enhanced retriever created with k={k}, threshold={similarity_threshold}")
        except Exception as e:
            logger.error(f"Failed to create retriever: {str(e)}")
            raise
    
    return _retriever

def search_documents(query: str, k: int = 3, with_scores: bool = False) -> List[Dict[str, Any]]:
    """Search documents and return results with metadata."""
    try:
        vectorstore = load_vectorstore()
        
        if with_scores:
            docs_with_scores = vectorstore.similarity_search_with_score(query, k=k)
            results = []
            for doc, score in docs_with_scores:
                results.append({
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'similarity_score': 1.0 - score,  # Convert distance to similarity
                    'distance_score': score
                })
            return results
        else:
            docs = vectorstore.similarity_search(query, k=k)
            return [{'content': doc.page_content, 'metadata': doc.metadata} for doc in docs]
            
    except Exception as e:
        logger.error(f"Error searching documents: {str(e)}")
        return []

def get_vectorstore_info() -> Dict[str, Any]:
    """Get information about the loaded vectorstore."""
    try:
        vectorstore = load_vectorstore()
        embeddings = get_embeddings()
        
        info = {
            'index_dir': settings.index_dir,
            'embedding_model': settings.embedding_model,
            'retrieval_k': settings.retrieval_k,
            'similarity_threshold': settings.similarity_threshold,
            'index_loaded': True
        }
        
        try:
            info['total_documents'] = vectorstore.index.ntotal
            info['embedding_dimension'] = vectorstore.index.d
        except:
            info['total_documents'] = 'unknown'
            info['embedding_dimension'] = 'unknown'
        
        return info
        
    except Exception as e:
        return {
            'index_loaded': False,
            'error': str(e),
            'index_dir': settings.index_dir
        }

def health_check() -> Dict[str, Any]:
    """Health check for RAG components."""
    try:
        # Test embeddings
        embeddings = get_embeddings()
        test_embedding = embeddings.embed_query("test")
        
        # Test vectorstore
        vectorstore = load_vectorstore()
        
        # Test retrieval
        retriever = get_retriever()
        test_docs = retriever.get_relevant_documents("test", k=1)
        
        return {
            'status': 'healthy',
            'embeddings_loaded': True,
            'vectorstore_loaded': True,
            'retriever_working': True,
            'embedding_dimension': len(test_embedding),
            'test_retrieval_count': len(test_docs)
        }
        
    except Exception as e:
        return {
            'status': 'unhealthy',
            'error': str(e),
            'embeddings_loaded': _embeddings is not None,
            'vectorstore_loaded': _vectorstore is not None,
            'retriever_working': False
        }

# Cleanup function for testing/reloading
def reset_components():
    """Reset all global components (useful for testing or reloading)."""
    global _embeddings, _vectorstore, _retriever
    _embeddings = None
    _vectorstore = None
    _retriever = None
    logger.info("RAG components reset")

# Legacy compatibility function
def get_basic_retriever():
    """Legacy function for backward compatibility."""
    return get_retriever().base_retriever