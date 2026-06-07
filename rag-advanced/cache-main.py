from google import genai
from dotenv import load_dotenv
import os

# -----------------------------
# Load Environment
# -----------------------------
load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

# -----------------------------
# Simple Cache
# -----------------------------
cache = {}

# -----------------------------
# Function
# -----------------------------
def ask_llm(question):

    # Cache Hit
    if question in cache:
        print("\n✅ CACHE HIT")
        return cache[question]

    # Cache Miss
    print("\n❌ CACHE MISS")
    print("Calling Gemini...")

    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=question
    )

    answer = response.text

    # Save to cache
    cache[question] = answer

    return answer


# -----------------------------
# Question
# -----------------------------
question = "What is Kubernetes?"

# -----------------------------
# First Call
# -----------------------------
print("\nFIRST REQUEST")

answer = ask_llm(question)

print("\nAnswer:")
print(answer)

# -----------------------------
# Second Call
# -----------------------------
print("\nSECOND REQUEST")

answer = ask_llm(question)

print("\nAnswer:")
print(answer)