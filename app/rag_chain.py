from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import os

INDEX_DIR = os.path.join(os.path.dirname(__file__), '../faiss_index')

def get_retriever():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
    return db.as_retriever(search_kwargs={"k": 3})
