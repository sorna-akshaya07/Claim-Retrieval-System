# ============================================================
# STAGE 3 → RERANKING + TOP 10 (FIXED VERSION)
# ============================================================

print("STAGE 3 STARTED")

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
# READ STAGE2 OUTPUT
# ============================================================

input_file = "stage2_output.txt"

with open(input_file, "r", encoding="utf-8") as f:
    content = f.read()

# ============================================================
# SPLIT CLAIM BLOCKS (FIXED)
# ============================================================

claim_blocks = re.split(
    r'\n={80,}\n',
    content
)

print("TOTAL CLAIM BLOCKS:", len(claim_blocks))

# ============================================================
# RERANK SCORE FUNCTION
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

        # ✅ RAW LOGIT SCORE
        score = logits.item()

    return score

# ============================================================
# PASSAGE TYPE FUNCTION
# ============================================================

def get_passage_type(score):

    if score >= 5:
        return "Highly Relevant"

    elif score >= 2:
        return "Relevant"

    else:
        return "Irrelevant"

# ============================================================
# OUTPUT FILE
# ============================================================

output_file = "stage3_output.txt"

# ============================================================
# MAIN PIPELINE
# ============================================================

with open(output_file, "w", encoding="utf-8") as out:

    # ========================================================
    # LOOP CLAIM BLOCKS
    # ========================================================

    for block_idx, block in enumerate(claim_blocks):

        block = block.strip()

        if not block:
            continue

        print("\n")
        print("=" * 80)
        print(f"PROCESSING CLAIM BLOCK {block_idx}")
        print("=" * 80)

        # ====================================================
        # EXTRACT CLAIM
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
            print("CLAIM NOT FOUND")
            continue

        claim_id = claim_id_match.group(1).strip()
        claim_text = claim_match.group(1).strip()

        print("CLAIM ID:", claim_id)

        # ====================================================
        # WRITE CLAIM
        # ====================================================

        out.write("\n")
        out.write("=" * 100)
        out.write("\n\n")

        out.write(f"CLAIM ID: {claim_id}\n")
        out.write(f"CLAIM: {claim_text}\n")

        # ====================================================
        # SPLIT SUBCLAIMS (FIXED)
        # ====================================================

        subclaim_sections = re.split(
            r'\n-{80}\n',
            block
        )

        print("TOTAL SUBCLAIM SECTIONS:", len(subclaim_sections))

        # ====================================================
        # LOOP SUBCLAIMS
        # ====================================================

        for section_idx, section in enumerate(subclaim_sections):

            section = section.strip()

            if not section:
                continue

            # =================================================
            # EXTRACT SUBCLAIM
            # =================================================

            subclaim_match = re.search(
                r'SUBCLAIM\s*\d*:\s*(.*)',
                section
            )

            if not subclaim_match:
                continue

            subclaim = subclaim_match.group(1).strip()

            print("\n")
            print("-" * 80)
            print(f"SUBCLAIM {section_idx}")
            print(subclaim)
            print("-" * 80)

            # =================================================
            # WRITE SUBCLAIM
            # =================================================

            out.write("\n")
            out.write("-" * 80)
            out.write("\n\n")

            out.write(f"SUBCLAIM: {subclaim}\n")

            # =================================================
            # EXTRACT PASSAGES (FIXED REGEX)
            # =================================================

            matches = re.findall(

                r'PASSAGE\s+\d+:\s*\n'
                r'(.*?)'
                r'RETRIEVAL SCORE:\s*([0-9.]+)\s*\n'
                r'RERANK SCORE:\s*([0-9.]+)\s*\n'
                r'PASSAGE_TYPE:\s*(.*?)\s*\n'
                r'-{20,}',

                section,
                re.DOTALL
            )

            print("TOTAL PASSAGES FOUND:", len(matches))

            scored_passages = []

            # =================================================
            # RERANK PASSAGES
            # =================================================

            for idx, match in enumerate(matches):

                passage = match[0].strip()
                retrieval_score = match[1]

                # =============================================
                # SKIP SHORT PASSAGES
                # =============================================

                if len(passage) < 20:
                    continue

                print(f"Scoring Passage {idx + 1}")

                # =============================================
                # COMPUTE RERANK SCORE
                # =============================================

                score = rerank_score(
                    subclaim,
                    passage
                )

                passage_type = get_passage_type(score)

                scored_passages.append({

                    "passage": passage,

                    "retrieval_score":
                        float(retrieval_score),

                    "rerank_score":
                        score,

                    "passage_type":
                        passage_type
                })

            # =================================================
            # SORT BY RERANK SCORE
            # =================================================

            scored_passages = sorted(

                scored_passages,

                key=lambda x:
                    x["rerank_score"],

                reverse=True
            )

            # =================================================
            # TOP 10
            # =================================================

            scored_passages = scored_passages[:10]

            # =================================================
            # HANDLE EMPTY
            # =================================================

            if not scored_passages:

                out.write("\nNO PASSAGES FOUND\n")

                print("NO PASSAGES FOUND")

                continue

            # =================================================
            # WRITE OUTPUT
            # =================================================

            for idx, item in enumerate(scored_passages, start=1):

                out.write(f"\nPASSAGE {idx}:\n\n")

                out.write(
                    item["passage"] + "\n\n"
                )

                out.write(
                    f"RETRIEVAL SCORE: "
                    f"{item['retrieval_score']:.4f}\n"
                )

                out.write(
                    f"RERANK SCORE: "
                    f"{item['rerank_score']:.4f}\n"
                )

                

                out.write(
                    "-" * 60 + "\n"
                )

# ============================================================
# FINISHED
# ============================================================

print("\n")
print("=" * 100)
print("STAGE 3 COMPLETED SUCCESSFULLY")
print("=" * 100)

print("Saved File:", output_file)
