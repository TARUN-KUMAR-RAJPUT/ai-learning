from google import genai
from dotenv import load_dotenv
import os
import json
from datetime import datetime

# -----------------------------
# 1. Load environment variables
# -----------------------------
load_dotenv()

# -----------------------------
# 2. Gemini Client
# -----------------------------
client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

# -----------------------------
# 3. Tools
# -----------------------------
def add_numbers(a, b):
    return a + b

def multiply_numbers(a, b):
    return a * b

def current_time():
    return datetime.now().strftime("%H:%M:%S")

# -----------------------------
# 4. Tool Registry
# -----------------------------
TOOLS = {
    "add_numbers": add_numbers,
    "multiply_numbers": multiply_numbers,
    "current_time": current_time
}

# -----------------------------
# 5. User Query
# -----------------------------
user_query = input("Ask something: ")

# Examples:
# What is 5 + 7?
# What is 6 * 8?
# What time is it?

# -----------------------------
# 6. Prompt
# -----------------------------
prompt = f"""
You are an AI assistant.

Available tools:

1. add_numbers(a, b)
   Use for addition.

2. multiply_numbers(a, b)
   Use for multiplication.

3. current_time()
   Use when user asks for current time.

Return ONLY valid JSON.

Examples:

{{
  "tool": "add_numbers",
  "arguments": {{
    "a": 5,
    "b": 7
  }}
}}

{{
  "tool": "multiply_numbers",
  "arguments": {{
    "a": 6,
    "b": 8
  }}
}}

{{
  "tool": "current_time",
  "arguments": {{}}
}}

User Query:
{user_query}
"""

# -----------------------------
# 7. Call Gemini
# -----------------------------
response = client.models.generate_content(
    model="gemini-2.5-flash-lite",
    contents=prompt
)

# -----------------------------
# 8. Print Raw Output
# -----------------------------
print("\nLLM Output:")
print(response.text)

# -----------------------------
# 9. Clean Markdown
# -----------------------------
clean_text = response.text.strip()
clean_text = clean_text.replace("```json", "")
clean_text = clean_text.replace("```", "")
clean_text = clean_text.strip()

# -----------------------------
# 10. Parse JSON
# -----------------------------
try:
    data = json.loads(clean_text)
except Exception as e:
    print("\n❌ JSON parsing failed")
    print(e)
    exit()

# -----------------------------
# 11. Dynamic Execution
# -----------------------------
tool_name = data["tool"]
arguments = data.get("arguments", {})

if tool_name not in TOOLS:
    print(f"\n❌ Unknown tool: {tool_name}")
    exit()

result = TOOLS[tool_name](**arguments)

# -----------------------------
# 12. Result
# -----------------------------
print("\n✅ Tool Selected:", tool_name)
print("✅ Result:", result)