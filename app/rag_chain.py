import os
import logging
from typing import List, Optional, Dict, Any
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from app.settings import settings

logger = logging.getLogger("app.rag_chain")


# ------------------------
# Embeddings Wrapper
# ------------------------
class LocalEmbedding:
    def __init__(self, model_path: str):
        logger.info(f"Loading embedding model from: {model_path}")
        self.model = SentenceTransformer(model_path)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, convert_to_tensor=False).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode([text], convert_to_tensor=False)[0].tolist()


# ------------------------
# Main RAG Chain
# ------------------------
class RAGChain:
    def __init__(self):
        self.embedding_model = LocalEmbedding(settings.embedding_model_path)
        self.persist_directory = settings.persist_directory
        self.k = settings.k

        self.vectorstore = self._load_vectorstore()

    def _load_vectorstore(self):
        path = os.path.join(self.persist_directory, "index.faiss")
        if os.path.exists(path):
            logger.info("Loading existing FAISS index...")
            return FAISS.load_local(
                self.persist_directory,
                embeddings=self.embedding_model,
                index_name="index",
                allow_dangerous_deserialization=True
            )
        else:
            logger.info("No existing FAISS index found.")
            return None

    def add_documents(self, documents: List[Document]):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap
        )
        docs = splitter.split_documents(documents)

        if not self.vectorstore:
            self.vectorstore = FAISS.from_documents(docs, self.embedding_model)
        else:
            self.vectorstore.add_documents(docs)

        self.vectorstore.save_local(self.persist_directory, index_name="index")

    def get_relevant_documents(self, query: str, memory: Optional[List[str]] = None, k: Optional[int] = None) -> List[Document]:
        k = k or self.k
        contextual_query = self._build_contextual_query(query, memory)
        return self.vectorstore.similarity_search(contextual_query, k=k)

    def _build_contextual_query(self, query: str, memory: Optional[List[str]] = None) -> str:
        if memory:
            history = "\n".join(memory)
            return f"{history}\n\n{query}"
        return query


# ------------------------
# Global RAGChain instance
# ------------------------
_rag_chain = None


def get_rag_chain() -> RAGChain:
    global _rag_chain
    if _rag_chain is None:
        _rag_chain = RAGChain()
    return _rag_chain


# ------------------------
# Public Interface
# ------------------------
def search_documents(query: str, memory: Optional[List[str]] = None, k: int = 3, with_scores: bool = False):
    try:
        retriever = get_rag_chain()
        docs = retriever.get_relevant_documents(query, memory=memory, k=k)

        results = []
        for doc in docs:
            result = {
                "content": doc.page_content,
                "metadata": doc.metadata
            }
            if with_scores:
                result["similarity_score"] = doc.metadata.get("similarity_score", None)
            results.append(result)

        if results:
            logger.info(f"Top result: {results[0]['content'][:200]}")
        else:
            logger.warning("No results found.")
        return results
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return []


def get_vectorstore_info() -> Dict[str, Any]:
    try:
        rag = get_rag_chain()
        vs = rag.vectorstore
        info = {
            "index_dir": settings.persist_directory,
            "embedding_model": settings.embedding_model_path,
            "retrieval_k": settings.k,
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
        return {
            "index_loaded": False,
            "error": str(e),
            "index_dir": settings.persist_directory
        }


def health_check() -> Dict[str, Any]:
    try:
        rag = get_rag_chain()
        test_embed = rag.embedding_model.embed_query("test")
        test_docs = rag.get_relevant_documents("test", memory=["Hi", "Can you help me?"], k=1)

        return {
            "status": "healthy",
            "embeddings_loaded": True,
            "vectorstore_loaded": rag.vectorstore is not None,
            "retriever_working": True,
            "embedding_dimension": len(test_embed),
            "test_retrieval_count": len(test_docs)
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "embeddings_loaded": _rag_chain.embedding_model is not None if _rag_chain else False,
            "vectorstore_loaded": _rag_chain.vectorstore is not None if _rag_chain else False,
            "retriever_working": False
        }
