from google import genai
from dotenv import load_dotenv
import os
import json

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
# Sample Logs
# -----------------------------
logs = """
ERROR: Database connection timeout
ERROR: Retry limit exceeded
WARN: Slow query detected
INFO: Health check passed
ERROR: Service unavailable
"""

# =====================================================
# STEP 1 - EXTRACT ERRORS
# =====================================================

prompt_extract = f"""
You are a log analysis assistant.

Extract only ERROR messages.

Return JSON ONLY.

Format:

{{
  "errors": [
    "error1",
    "error2"
  ]
}}

Logs:

{logs}
"""

response1 = client.models.generate_content(
    model="gemini-2.5-flash-lite",
    contents=prompt_extract
)

print("\n========== STEP 1 : EXTRACT ERRORS ==========\n")
print(response1.text)

# Clean markdown
clean1 = response1.text.replace("```json", "").replace("```", "").strip()

errors_json = json.loads(clean1)

# =====================================================
# STEP 2 - ANALYZE ERRORS
# =====================================================

prompt_analyze = f"""
You are a backend production engineer.

Analyze these errors:

{json.dumps(errors_json)}

Return JSON ONLY.

Format:

{{
  "root_cause": "..."
}}
"""

response2 = client.models.generate_content(
    model="gemini-2.5-flash-lite",
    contents=prompt_analyze
)

print("\n========== STEP 2 : ROOT CAUSE ==========\n")
print(response2.text)

clean2 = response2.text.replace("```json", "").replace("```", "").strip()

analysis_json = json.loads(clean2)

# =====================================================
# STEP 3 - CREATE FINAL REPORT
# =====================================================

prompt_report = f"""
Create a final incident report.

Errors:

{json.dumps(errors_json)}

Analysis:

{json.dumps(analysis_json)}

Return JSON ONLY.

Format:

{{
  "summary": "...",
  "severity": "LOW/MEDIUM/HIGH"
}}
"""

response3 = client.models.generate_content(
    model="gemini-2.5-flash-lite",
    contents=prompt_report
)

print("\n========== STEP 3 : FINAL REPORT ==========\n")
print(response3.text)

clean3 = response3.text.replace("```json", "").replace("```", "").strip()

report_json = json.loads(clean3)

# =====================================================
# FINAL OUTPUT
# =====================================================

print("\n========== FINAL RESULT ==========\n")

print(json.dumps(report_json, indent=2))