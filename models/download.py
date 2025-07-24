from sentence_transformers import SentenceTransformer

# Downloads and caches the model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Optional: Save it manually to a custom path
model.save('./models/embeddings/all-MiniLM-L6-v2')