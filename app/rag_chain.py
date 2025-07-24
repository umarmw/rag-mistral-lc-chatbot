import os
import logging
from typing import Optional, Dict, Any, List, Union
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic_settings import BaseSettings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGSettings(BaseSettings):
    index_dir: str = os.path.join(os.path.dirname(__file__), '../faiss_index')
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    retrieval_k: int = 3
    similarity_threshold: float = 0.7
    max_context_length: int = 4000
    normalize_embeddings: bool = True

    class Config:
        env_prefix = "RAG_"

settings = RAGSettings()

# Global instances
_embeddings: Optional[HuggingFaceEmbeddings] = None
_vectorstore: Optional[FAISS] = None
_retriever: Optional["EnhancedRetriever"] = None


def get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        logger.info(f"Loading embeddings model: {settings.embedding_model}")
        _embeddings = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": settings.normalize_embeddings}
        )
    return _embeddings


def load_vectorstore() -> FAISS:
    global _vectorstore
    if _vectorstore is None:
        if not os.path.exists(settings.index_dir):
            raise FileNotFoundError(f"FAISS index directory not found: {settings.index_dir}")
        embeddings = get_embeddings()
        _vectorstore = FAISS.load_local(
            settings.index_dir,
            embeddings,
            allow_dangerous_deserialization=True
        )
        try:
            logger.info(f"Loaded FAISS index with {_vectorstore.index.ntotal} documents")
        except Exception:
            logger.info("FAISS index loaded")
    return _vectorstore


class EnhancedRetriever:
    def __init__(self, vectorstore: FAISS, k: int = 3, similarity_threshold: float = 0.7):
        self.vectorstore = vectorstore
        self.k = k
        self.similarity_threshold = similarity_threshold
        self.base_retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    def _build_contextual_query(self, query: str, memory: Optional[List[str]] = None) -> str:
        if not memory:
            return query
        history = "\n".join(memory)
        return f"{history}\nCurrent user query: {query}"

    def get_relevant_documents(
        self, query: str, memory: Optional[List[str]] = None, k: Optional[int] = None
    ) -> List[Document]:
        k = k or self.k
        contextual_query = self._build_contextual_query(query, memory)

        try:
            docs_with_scores = self.vectorstore.similarity_search_with_score(
                contextual_query, k=k * 2  # overfetch
            )

            # Adjust threshold for FAISS L2 distance (lower is more similar)
            score_threshold = self.similarity_threshold or 2.2  # default upper bound

            filtered = []
            for doc, score in docs_with_scores:
                logger.info(f"Candidate doc score: {score:.4f} | content: {doc.page_content[:100]}")

                if score <= score_threshold:
                    doc.metadata.update({
                        "similarity_score": 1.0 / (1.0 + score),  # optional normalization
                        "retrieval_query": contextual_query
                    })
                    filtered.append(doc)

            final_docs = filtered[:k]
            logger.info(f"Retrieved {len(final_docs)} documents (filtered from {len(docs_with_scores)})")
            return final_docs

        except Exception as e:
            logger.error(f"Retriever error: {e}")
            return self.base_retriever.get_relevant_documents(query)

    def get_relevant_documents_with_context_limit(
            self, query: str, memory: Optional[List[str]] = None, max_length: Optional[int] = None
        ) -> List[Document]:
        max_length = max_length or settings.max_context_length
        docs = self.get_relevant_documents(query, memory=memory)
        total = 0
        trimmed_docs = []

        for doc in docs:
            content = doc.page_content
            if total + len(content) <= max_length:
                trimmed_docs.append(doc)
                total += len(content)
            else:
                remain = max_length - total
                if remain > 100:
                    doc_trimmed = Document(
                        page_content=content[:remain - 10] + "...",
                        metadata={**doc.metadata, "trimmed": True}
                    )
                    trimmed_docs.append(doc_trimmed)
                break

        if len(trimmed_docs) < len(docs):
            logger.info(f"Trimmed {len(docs)} docs to {len(trimmed_docs)} for context length {max_length}")
        return trimmed_docs


def get_retriever(k: Optional[int] = None, similarity_threshold: Optional[float] = None) -> EnhancedRetriever:
    global _retriever
    k = k or settings.retrieval_k
    similarity_threshold = similarity_threshold or settings.similarity_threshold
    if _retriever is None:
        _retriever = EnhancedRetriever(load_vectorstore(), k, similarity_threshold)
    return _retriever


def search_documents(*, query: str, memory: Optional[List[str]] = None, k: int = 3, with_scores: bool = False):
    try:
        retriever = get_retriever()
        docs = retriever.get_relevant_documents(query, memory=memory, k=k)

        results = []
        for doc in docs:
            result = {
                "content": doc.page_content,
                "metadata": doc.metadata
            }
            if with_scores:
                result["similarity_score"] = doc.metadata.get("similarity_score")
            results.append(result)

        logger.info(f"Top result: {results[0]['content'][:200]}") if results else logger.warning("No results found.")
        return results
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return []


def get_vectorstore_info() -> Dict[str, Any]:
    try:
        vs = load_vectorstore()
        info = {
            "index_dir": settings.index_dir,
            "embedding_model": settings.embedding_model,
            "retrieval_k": settings.retrieval_k,
            "similarity_threshold": settings.similarity_threshold,
            "index_loaded": True
        }
        try:
            info["total_documents"] = vs.index.ntotal
            info["embedding_dimension"] = vs.index.d
        except Exception:
            info["total_documents"] = "unknown"
            info["embedding_dimension"] = "unknown"
        return info
    except Exception as e:
        return {"index_loaded": False, "error": str(e), "index_dir": settings.index_dir}


def health_check() -> Dict[str, Any]:
    try:
        embeddings = get_embeddings()
        _ = embeddings.embed_query("test")
        vectorstore = load_vectorstore()
        retriever = get_retriever()
        test_docs = retriever.get_relevant_documents("test", memory=["Hi", "Can you help me?"])
        return {
            "status": "healthy",
            "embeddings_loaded": True,
            "vectorstore_loaded": True,
            "retriever_working": True,
            "embedding_dimension": len(_),
            "test_retrieval_count": len(test_docs)
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "embeddings_loaded": _embeddings is not None,
            "vectorstore_loaded": _vectorstore is not None,
            "retriever_working": False
        }


def reset_components():
    global _embeddings, _vectorstore, _retriever
    _embeddings = None
    _vectorstore = None
    _retriever = None
    logger.info("Reset all RAG components")


def get_basic_retriever():
    return get_retriever().base_retriever
