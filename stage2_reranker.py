# ============================================================
# STAGE 2 → RERANKING + LABELING
# ============================================================

# pip install sentence-transformers transformers torch

print("STAGE 2 STARTED")

# ============================================================
# IMPORTS
# ============================================================

import re
import torch

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

# ============================================================
# LOAD RERANK MODEL
# ============================================================

print("Loading Reranker Model...")

tokenizer = AutoTokenizer.from_pretrained(
    "BAAI/bge-reranker-base"
)

model = AutoModelForSequenceClassification.from_pretrained(
    "BAAI/bge-reranker-base"
)

model.eval()

print("Reranker Loaded Successfully")

# ============================================================
# READ STAGE1 OUTPUT
# ============================================================

input_file = "stage1_output.txt"

with open(input_file, "r", encoding="utf-8") as f:

    content = f.read()

# ============================================================
# SPLIT CLAIM BLOCKS
# ============================================================

claim_blocks = content.split(
    "=" * 100
)

# ============================================================
# SCORE FUNCTION
# ============================================================

def rerank_score(query, passage):

    inputs = tokenizer(
        query,
        passage,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )

    with torch.no_grad():

        logits = model(**inputs).logits

        score = torch.sigmoid(logits).item()

    return score

# ============================================================
# LABEL FUNCTION
# ============================================================

def get_label(score):

    if score >= 0.85:
        return "Highly Relevant"

    elif score >= 0.55:
        return "Relevant"

    else:
        return "Irrelevant"

# ============================================================
# OUTPUT FILE
# ============================================================

output_file = "stage2_output.txt"

# ============================================================
# MAIN PIPELINE
# ============================================================

with open(output_file, "w", encoding="utf-8") as out:

    for block in claim_blocks:

        block = block.strip()

        if not block:
            continue

        # ====================================================
        # CLAIM
        # ====================================================

        claim_id_match = re.search(
            r'CLAIM ID:\s*(.*)',
            block
        )

        claim_match = re.search(
            r'CLAIM:\s*(.*)',
            block
        )

        if not claim_match:
            continue

        claim_id = claim_id_match.group(1).strip()

        claim_text = claim_match.group(1).strip()

        out.write("\n")
        out.write("=" * 100)
        out.write("\n")

        out.write(f"\nCLAIM ID: {claim_id}\n")
        out.write(f"CLAIM: {claim_text}\n")

        # ====================================================
        # SUBCLAIMS
        # ====================================================

        subclaim_sections = re.split(
            r'-{80}',
            block
        )

        for section in subclaim_sections:

            subclaim_match = re.search(
                r'SUBCLAIM\s+\d+:\s*(.*)',
                section
            )

            if not subclaim_match:
                continue

            subclaim = subclaim_match.group(1).strip()

            out.write("\n")
            out.write("-" * 80)
            out.write("\n")

            out.write(f"\nSUBCLAIM: {subclaim}\n")

            # =================================================
            # EXTRACT PASSAGES
            # =================================================

            passages = re.findall(
                r'PASSAGE\s+\d+:\n(.*?)\n-+',
                section,
                re.DOTALL
            )

            scored_passages = []

            # =================================================
            # SCORE EACH PASSAGE
            # =================================================

            for passage in passages[:10]:

                passage = passage.strip()

                if len(passage) < 20:
                    continue

                score = rerank_score(
                    subclaim,
                    passage
                )

                label = get_label(score)

                scored_passages.append({

                    "passage": passage,

                    "score": score,

                    "label": label
                })

            # =================================================
            # SORT BY SCORE
            # =================================================

            scored_passages = sorted(
                scored_passages,
                key=lambda x: x["score"],
                reverse=True
            )

            # =================================================
            # WRITE OUTPUT
            # =================================================

            for idx, item in enumerate(scored_passages, start=1):

                out.write(f"\nPASSAGE {idx}:\n")

                out.write(item["passage"] + "\n\n")

                out.write(
                    f"RERANK SCORE: {item['score']:.4f}\n"
                )

                out.write(
                    f"LABEL: {item['label']}\n"
                )

                out.write("-" * 60 + "\n")

# ============================================================
# FINISHED
# ============================================================

print("\nSTAGE 2 COMPLETED SUCCESSFULLY")

print("Saved File:", output_file)