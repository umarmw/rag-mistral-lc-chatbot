# RAG Chatbot using Mistral 7B

This is a minimal Retrieval-Augmented Generation (RAG) chatbot using LangChain, FAISS, and Mistral 7B.

## 🔧 Setup

1. Clone the repo:
```bash
git clone <repo-url>
cd rag-mistral-lc-chatbot
```

2. Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Place your PDF files inside the `data/` directory.

5. Run the embedding pipeline:
```bash
python app/ingest.py
```

6. Start the FastAPI backend:
```bash
uvicorn app.server:app --reload
```

7. In another terminal, run the Streamlit UI:
```bash
streamlit run app/ui.py
```

8. Use the chat-style interface to ask questions.

## Docker

1. Build your Docker image:
`docker build -t rag-lc-chatbot .`

2. Run locally (for testing):
`docker run -p 8000:8000 rag-lc-chatbot`

## Custom LLM

1. Local setup
`pip install llama-cpp-python==0.2.38`

2. Download the Mistral GGUF model
Download from TheBloke's repo (choose a q4_K_M or q5_K_M variant for decent performance and low memory use)
`wget https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q5_K_M.gguf -P ./models/`