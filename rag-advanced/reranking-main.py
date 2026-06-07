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
    "Kubernetes deployment tutorial",
    "Container orchestration basics",
    "Database optimization techniques",
    "ERR_CONN_1045 database connection error"
]

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
# Create Embeddings
# -----------------------------
embeddings = np.array(
    [get_embedding(doc) for doc in documents]
)

# -----------------------------
# Build FAISS Index
# -----------------------------
dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)

# -----------------------------
# Query Search
# -----------------------------
query_embedding = np.array(
    [get_embedding(query)],
    dtype=np.float32
)

# Retrieve Top 5
distances, indices = index.search(
    query_embedding,
    5
)

# -----------------------------
# First Stage Retrieval
# -----------------------------
candidates = []

for idx in indices[0]:
    candidates.append(documents[idx])

print("\n=== BEFORE RE-RANKING ===")

for i, doc in enumerate(candidates, start=1):
    print(f"{i}. {doc}")

# -----------------------------
# Re-ranking
# -----------------------------
query_words = query.lower().split()

reranked = []

for doc in candidates:

    score = 0

    doc_lower = doc.lower()

    for word in query_words:

        if word in doc_lower:
            score += 1

    reranked.append((doc, score))

reranked.sort(
    key=lambda x: x[1],
    reverse=True
)

# -----------------------------
# Final Results
# -----------------------------
print("\n=== AFTER RE-RANKING ===")

for i, (doc, score) in enumerate(reranked[:3], start=1):
    print(f"{i}. {doc} (score={score})")