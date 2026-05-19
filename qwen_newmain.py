from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import time

# =========================
# Load Qwen2.5-3B
# =========================

model_name = "Qwen/Qwen2.5-3B-Instruct"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("Loading model...")

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

print("✅ Model loaded successfully!")

# =========================
# Function to generate subclaims
# =========================

def generate_subclaims(claim):

    prompt = f"""
TASK:
Break the claim into atomic sub-claims and create a reasoning chain.

STRICT RULES:

1. Output ONLY:
   - Sub-claims
   - Reasoning Chain

2. Do NOT write:
   - Explanation
   - Notes
   - Summary
   - OUTPUT
   - Final Claim

3. Do NOT repeat the original claim.

4. Each sub-claim must contain only ONE fact.

5. Do NOT invent information not explicitly present in the claim.

6. Keep sub-claims short and fact-checkable.

7. Arrange sub-claims logically.

8. Use this EXACT format:

Sub-claims:
1. ...
2. ...
3. ...

Reasoning Chain:
1 → 2 → 3

IMPORTANT:
Return ONLY the formatted answer.
Do not generate extra text.

CLAIM:
{claim}
"""

    messages = [
        {
            "role": "system",
            "content": (
                "You are a strict claim decomposition engine. "
                "Return ONLY sub-claims and reasoning chain. "
                "Never output explanations, summaries, notes, "
                "OUTPUT labels, or the original claim."
            )
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        text,
        return_tensors="pt"
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0,
        do_sample=False
    )

    result = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    # Remove prompt part
    result = result.split("assistant")[-1].strip()

    # 🔹 Cleanup
    if "Explanation:" in result:
        result = result.split("Explanation:")[0].strip()

    if "OUTPUT:" in result:
        result = result.replace("OUTPUT:", "").strip()

    return result


# =========================
# File paths
# =========================

input_file = "test.jsonl"
output_file = "output_newqwen.txt"

# =========================
# Range
# =========================

start = 0
end = 5

current_index = 0

# =========================
# Main loop
# =========================

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

        claim = data.get("claim", "").strip()

        if not claim:
            current_index += 1
            continue

        print(f"Processing claim_{current_index}")

        try:
            result = generate_subclaims(claim)

        except Exception as e:
            print("Error:", e)
            current_index += 1
            continue

        # Save output
        out.write(f"claim_{current_index}:\n")
        out.write(claim + "\n\n")
        out.write(result + "\n")
        out.write("=" * 60 + "\n\n")

        current_index += 1

        time.sleep(1)

print("✅ Claims processed successfully!")