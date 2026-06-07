from google import genai
from dotenv import load_dotenv
import os
import numpy as np
import faiss

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()

# -----------------------------
# Gemini Client
# -----------------------------
client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

# -----------------------------
# Documents
# -----------------------------
documents = [
    "Forgot password guide",
    "Kubernetes deployment tutorial",
    "Database optimization techniques",
    "How to recover your account password",
    "Microservices architecture basics"
]

# -----------------------------
# Query
# -----------------------------
query = "How do I reset my password?"

# -----------------------------
# Embedding Function
# -----------------------------
def get_embedding(text):
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text
    )

    return np.array(
        response.embeddings[0].values,
        dtype=np.float32
    )

# -----------------------------
# Generate Document Embeddings
# -----------------------------
document_embeddings = []

for doc in documents:
    embedding = get_embedding(doc)
    document_embeddings.append(embedding)

document_embeddings = np.array(document_embeddings)

# -----------------------------
# Build FAISS Index
# -----------------------------
dimension = document_embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(document_embeddings)

print(f"\nIndexed {index.ntotal} documents")

# -----------------------------
# Query Embedding
# -----------------------------
query_embedding = get_embedding(query)

query_embedding = np.array(
    [query_embedding],
    dtype=np.float32
)

# -----------------------------
# Search
# -----------------------------
k = 3

distances, indices = index.search(
    query_embedding,
    k
)

# -----------------------------
# Results
# -----------------------------
print("\nQuery:")
print(query)

print("\nTop Matches:\n")

for rank, idx in enumerate(indices[0], start=1):
    print(f"{rank}. {documents[idx]}")