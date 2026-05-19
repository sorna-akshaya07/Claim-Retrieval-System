from groq import Groq
from dotenv import load_dotenv
import os
import re
import json
import time

# 🔹 Load .env file
load_dotenv()

# 🔹 Get API Key
groq_key = os.getenv("GROQ_API_KEY")

# 🔹 Groq Client
client = Groq(api_key=groq_key)


# 🔹 Remove <think> blocks
def clean_output(text):

    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    return text.strip()


# 🔹 Generate subclaims
def generate_subclaims(claim):

    prompt = f"""
TASK:
Decompose the given claim into clear, logical sub-claims and build a reasoning chain.

RULES:

SUB-CLAIM RULES:
1. Break the claim into clear, logical sub-claims
2. Each sub-claim must contain ONLY one idea
3. Sub-claims need NOT match exact wording
4. Do NOT add new or external information
5. Do NOT assume anything not in the claim
6. Each sub-claim must be independently fact-checkable
7. Ensure decomposition is meaningful

REASONING RULES:
8. Arrange sub-claims in logical order
9. Each sub-claim must support or lead to the next
10. Build a reasoning chain using sub-claim numbers
11. Use format: 1 → 2 → 3 → Final Claim
12. Do NOT write "Step 1", "Step 2"
13. Use only numbers referring to sub-claims

IMPORTANT:
- DO NOT output thinking process
- DO NOT use <think>
- Return ONLY final formatted answer

OUTPUT FORMAT:

Sub-claims:
1. ...
2. ...
3. ...

Reasoning Chain:
1 → 2 → 3 → Final Claim

CLAIM:
{claim}
"""

    response = client.chat.completions.create(
        model="qwen/qwen3-32b",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a strict claim decomposition engine. "
                    "Never reveal internal reasoning or chain-of-thought. "
                    "Return only the final answer."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2,
        max_tokens=300
    )

    # 🔹 Raw output
    raw_output = response.choices[0].message.content

    # 🔹 Remove everything before </think>
    if "</think>" in raw_output:
        raw_output = raw_output.split("</think>")[-1].strip()

    # 🔹 Clean output
    cleaned_output = clean_output(raw_output)

    return cleaned_output


# 🔹 File paths
input_file = "test.jsonl"
output_file = "output1.txt"

# 🔹 Range
start = 100
end = 105

current_index = 0


# 🔹 Main loop
with open(input_file, "r", encoding="utf-8") as f, \
     open(output_file, "a", encoding="utf-8") as out:

    for line in f:

        # Skip until start
        if current_index < start:
            current_index += 1
            continue

        # Stop at end
        if current_index >= end:
            break

        data = json.loads(line)

        # 🔹 Fetch claim_id + claim only
        claim_id = data.get("claim_id", f"claim_{current_index}")
        claim = data.get("claim", "").strip()

        if not claim:
            current_index += 1
            continue

        print(f"Processing {claim_id}")

        try:
            result = generate_subclaims(claim)

        except Exception as e:
            print("Error:", e)
            current_index += 1
            continue

        # 🔹 Save output
        out.write(f"claim_id: {claim_id}\n")
        out.write(f"claim: {claim}\n\n")
        out.write(result + "\n")
        out.write("=" * 60 + "\n\n")

        current_index += 1

        # 🔹 Avoid rate limit
        time.sleep(1)

print("✅ Claims processed successfully!")