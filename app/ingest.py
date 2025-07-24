import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter


DATA_DIR = os.path.join(os.path.dirname(__file__), '../data')
INDEX_DIR = os.path.join(os.path.dirname(__file__), '../faiss_index')

def ingest():
    print("[INFO] Loading documents...")
    all_docs = []
    for file in os.listdir(DATA_DIR):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(DATA_DIR, file))
            docs = loader.load()
            all_docs.extend(docs)

    print(f"[INFO] Loaded {len(all_docs)} documents. Splitting...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(all_docs)

    print("[INFO] Creating embeddings and storing in FAISS...")
    embeddings = SentenceTransformerEmbeddings(model_name="models/all-MiniLM-L6-v2")
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(INDEX_DIR)
    print(f"[INFO] Vector store saved to {INDEX_DIR}")

if __name__ == '__main__':
    ingest()
