from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    embedding_model_path: str = "./models/embeddings/all-MiniLM-L6-v2"
    persist_directory: str = "./faiss_index"
    k: int = 5  # Number of top documents to retrieve
    chunk_size: int = 500
    chunk_overlap: int = 50

    class Config:
        env_file = ".env"  # optional


settings = Settings()
