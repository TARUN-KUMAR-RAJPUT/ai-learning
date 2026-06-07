from google import genai
from dotenv import load_dotenv
import os
import numpy as np
import faiss

# -----------------------------
# Load Environment
# -----------------------------
load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

# -----------------------------
# Documents
# -----------------------------
documents = [
    "Password reset guide",
    "Account recovery instructions",
    "Authentication troubleshooting handbook",
    "Kubernetes deployment tutorial",
    "Database optimization techniques"
]

# -----------------------------
# User Query
# -----------------------------
user_query = "login issue"

# -----------------------------
# Query Expansion
# -----------------------------
expanded_queries = [
    user_query,
    "password reset",
    "authentication failure",
    "account recovery"
]

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
# Create Document Embeddings
# -----------------------------
document_embeddings = np.array(
    [get_embedding(doc) for doc in documents]
)

# -----------------------------
# Build FAISS Index
# -----------------------------
dimension = document_embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(document_embeddings)

# -----------------------------
# Multi Query Retrieval
# -----------------------------
results = set()

for query in expanded_queries:

    query_embedding = np.array(
        [get_embedding(query)],
        dtype=np.float32
    )

    distances, indices = index.search(
        query_embedding,
        2
    )

    print(f"\nResults for: {query}")

    for idx in indices[0]:
        print(f"- {documents[idx]}")
        results.add(documents[idx])

# -----------------------------
# Final Merged Results
# -----------------------------
print("\n======================")
print("FINAL MERGED RESULTS")
print("======================")

for doc in results:
    print(f"- {doc}")