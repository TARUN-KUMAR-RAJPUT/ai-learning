from google import genai
from dotenv import load_dotenv
import os
import numpy as np

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
# User Query
# -----------------------------
query = "How do I reset my password?"

# -----------------------------
# Get Embedding
# -----------------------------
def get_embedding(text):
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text
    )

    return np.array(response.embeddings[0].values)

# -----------------------------
# Cosine Similarity
# -----------------------------
def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (
        np.linalg.norm(v1) * np.linalg.norm(v2)
    )

# -----------------------------
# Query Embedding
# -----------------------------
query_embedding = get_embedding(query)

# -----------------------------
# Compare Against Documents
# -----------------------------
scores = []

for doc in documents:

    doc_embedding = get_embedding(doc)

    similarity = cosine_similarity(
        query_embedding,
        doc_embedding
    )

    scores.append((doc, similarity))

# -----------------------------
# Sort Results
# -----------------------------
scores.sort(
    key=lambda x: x[1],
    reverse=True
)

# -----------------------------
# Print Results
# -----------------------------
print("\nQuery:")
print(query)

print("\nMost Similar Documents:\n")

for doc, score in scores:
    print(f"{score:.4f}  ->  {doc}")