from google import genai
import os
from dotenv import load_dotenv
import json

# -----------------------------
# 1. Load environment variables
# -----------------------------
load_dotenv()

# -----------------------------
# 2. Initialize Gemini client
# -----------------------------
client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

# -----------------------------
# 3. Define tool
# -----------------------------
def add_numbers(a, b):
    return a + b

# -----------------------------
# 4. User Query
# -----------------------------
user_query = "What is 5 + 7?"

# -----------------------------
# 5. Prompt
# -----------------------------
prompt = f"""
You are an AI assistant.

If the user asks a math question, return ONLY valid JSON.

Format:

{{
  "tool": "add_numbers",
  "arguments": {{
    "a": number,
    "b": number
  }}
}}

User query: {user_query}
"""

# -----------------------------
# 6. Call Gemini
# -----------------------------
response = client.models.generate_content(
    model="gemini-2.5-flash-lite",
    contents=prompt
)

# -----------------------------
# 7. Print raw output
# -----------------------------
print("\nLLM Output:")
print(response.text)

# -----------------------------
# 8. Clean markdown wrappers
# -----------------------------
clean_text = response.text.strip()

clean_text = clean_text.replace("```json", "")
clean_text = clean_text.replace("```", "")
clean_text = clean_text.strip()

# -----------------------------
# 9. Parse JSON
# -----------------------------
try:
    data = json.loads(clean_text)
except Exception as e:
    print("\n❌ JSON parsing failed")
    print("Error:", e)
    print("\nRaw response:")
    print(response.text)
    exit()

# -----------------------------
# 10. Execute tool
# -----------------------------
if data["tool"] == "add_numbers":

    a = data["arguments"]["a"]
    b = data["arguments"]["b"]

    result = add_numbers(a, b)

    print("\n✅ Tool Result:", result)

else:
    print("\n⚠️ Unknown tool")