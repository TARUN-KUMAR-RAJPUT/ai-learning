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
# Keyword Search
# -----------------------------
def keyword_search(query, docs):

    query_words = query.lower().split()

    results = []

    for doc in docs:

        doc_lower = doc.lower()

        score = 0

        for word in query_words:
            if word in doc_lower:
                score += 1

        if score > 0:
            results.append((doc, score))

    results.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return results

# -----------------------------
# Vector Search
# -----------------------------
embeddings = []

for doc in documents:
    embeddings.append(
        get_embedding(doc)
    )

embeddings = np.array(embeddings)

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)

query_embedding = np.array(
    [get_embedding(query)],
    dtype=np.float32
)

distances, indices = index.search(
    query_embedding,
    3
)

vector_results = []

for idx in indices[0]:
    vector_results.append(
        documents[idx]
    )

# -----------------------------
# Hybrid Search
# -----------------------------
keyword_results = keyword_search(
    query,
    documents
)

hybrid_results = set()

for doc, _ in keyword_results:
    hybrid_results.add(doc)

for doc in vector_results:
    hybrid_results.add(doc)

# -----------------------------
# Output
# -----------------------------
print("\nQuery:")
print(query)

print("\nKeyword Results:")
for doc, score in keyword_results:
    print(f"- {doc}")

print("\nVector Results:")
for doc in vector_results:
    print(f"- {doc}")

print("\nHybrid Results:")
for doc in hybrid_results:
    print(f"- {doc}")