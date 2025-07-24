import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformersEmbeddings

INDEX_DIR = os.path.join(os.path.dirname(__file__), '../faiss_index')

def get_retriever():
    embeddings = SentenceTransformersEmbeddings(
        model_name="models/all-MiniLM-L6-v2",  # Local path to your downloaded model
        cache_folder=None  # Optional, set if you want caching elsewhere
    )
    db = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
    return db.as_retriever(search_kwargs={"k": 3})
