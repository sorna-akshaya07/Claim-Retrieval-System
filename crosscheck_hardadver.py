# ============================================================
# ELIMINATE → HARD NEGATIVE + ADVERSARIAL REMOVE
# FINAL IMPROVED VERSION
# ============================================================

print("ELIMINATE STARTED")

# ============================================================
# IMPORTS
# ============================================================

import re
import torch

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import pipeline

# ============================================================
# LOAD RERANK MODEL
# ============================================================

print("Loading Reranker Model...")

rerank_tokenizer = AutoTokenizer.from_pretrained(
    "BAAI/bge-reranker-base"
)

rerank_model = AutoModelForSequenceClassification.from_pretrained(
    "BAAI/bge-reranker-base"
)

rerank_model.eval()

print("Reranker Loaded Successfully")

# ============================================================
# LOAD NLI MODEL
# ============================================================

print("Loading NLI Model...")

nli_pipe = pipeline(
    "text-classification",
    model="cross-encoder/nli-deberta-v3-small",
    device=0 if torch.cuda.is_available() else -1
)

print("NLI Model Loaded Successfully")

# ============================================================
# READ STAGE2 OUTPUT
# ============================================================

input_file = "stage2_output.txt"

with open(input_file, "r", encoding="utf-8") as f:
    content = f.read()

# ============================================================
# SPLIT CLAIM BLOCKS
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

    inputs = rerank_tokenizer(
        query,
        passage,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )

    with torch.no_grad():

        logits = rerank_model(**inputs).logits

        score = logits.item()

    return score

# ============================================================
# PASSAGE TYPE FUNCTION
# ============================================================

def get_passage_type(score):

    if score >= 6:
        return "Highly Relevant"

    elif score >= 3:
        return "Relevant"

    elif score >= 1:
        return "Weakly Relevant"

    else:
        return "Irrelevant"

# ============================================================
# NLI LABEL FUNCTION
# ============================================================

def get_nli_label(subclaim, passage):

    input_text = f"{passage} [SEP] {subclaim}"

    result = nli_pipe(
        input_text,
        truncation=True,
        max_length=512
    )

    label = result[0]["label"].upper()

    if "ENTAIL" in label:
        return "ENTAILMENT"

    elif "CONTRA" in label:
        return "CONTRADICTION"

    else:
        return "NEUTRAL"

# ============================================================
# OUTPUT FILE
# ============================================================

output_file = "hard_adver_output.txt"

# ============================================================
# MAIN PIPELINE
# ============================================================

with open(output_file, "w", encoding="utf-8") as out:

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
            continue

        claim_id = (
            claim_id_match.group(1).strip()
            if claim_id_match else "UNKNOWN"
        )

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
        # SPLIT SUBCLAIMS
        # ====================================================

        subclaim_sections = re.split(
            r'\n-{80}\n',
            block
        )

        print("TOTAL SUBCLAIM SECTIONS:",
              len(subclaim_sections))

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
                r'SUBCLAIM\s*\d*\s*:?\s*(.*)',
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
            # EXTRACT PASSAGES
            # =================================================

            matches = re.findall(

                r'PASSAGE\s+\d+:\s*\n'
                r'(.*?)'
                r'RETRIEVAL SCORE:\s*([0-9.\-]+)\s*\n'
                r'RERANK SCORE:\s*([0-9.\-]+)\s*\n'
                r'PASSAGE_TYPE:\s*(.*?)\s*\n'
                r'-{20,}',

                section,
                re.DOTALL
            )

            print("TOTAL PASSAGES FOUND:",
                  len(matches))

            scored_passages = []
            fallback_passages = []
            emergency_passages = []  # Min 3 guarantee

            # =================================================
            # PROCESS PASSAGES
            # =================================================

            for idx, match in enumerate(matches):

                passage = match[0].strip()
                retrieval_score = match[1]

                if len(passage) < 20:
                    continue

                print(f"\nChecking Passage {idx + 1}")

                # =============================================
                # RERANK SCORE
                # =============================================

                score = rerank_score(
                    subclaim,
                    passage
                )

                passage_type = get_passage_type(score)

                print("RERANK SCORE:",
                      round(score, 4))

                print("PASSAGE TYPE:",
                      passage_type)

                # =============================================
                # EMERGENCY SAVE — before any filter
                # =============================================

                emergency_passages.append({
                    "passage":         passage,
                    "retrieval_score": float(retrieval_score),
                    "rerank_score":    score,
                    "passage_type":    passage_type,
                    "nli_label":       "NEUTRAL"
                })

                # =============================================
                # REMOVE VERY LOW SCORES
                # =============================================

                if score < 1:

                    print(
                        f"REMOVED (Low Score): "
                        f"{score:.4f}"
                    )

                    continue

                # =============================================
                # NLI CHECK
                # =============================================

                nli_label = get_nli_label(
                    subclaim,
                    passage
                )

                print("NLI LABEL:",
                      nli_label)

                # =============================================
                # REMOVE CONTRADICTION
                # =============================================

                if nli_label == "CONTRADICTION":

                    print(
                        "REMOVED "
                        "(Contradiction)"
                    )

                    continue

                # =============================================
                # SAVE FOR FALLBACK
                # =============================================

                fallback_passages.append({

                    "passage":
                        passage,

                    "retrieval_score":
                        float(retrieval_score),

                    "rerank_score":
                        score,

                    "passage_type":
                        passage_type,

                    "nli_label":
                        nli_label
                })

                # =============================================
                # STRICT FILTER
                # =============================================

                if (
                    score >= 2
                    and passage_type != "Irrelevant"
                ):

                    scored_passages.append({

                        "passage":
                            passage,

                        "retrieval_score":
                            float(retrieval_score),

                        "rerank_score":
                            score,

                        "passage_type":
                            passage_type,

                        "nli_label":
                            nli_label
                    })

            # =================================================
            # SORT PASSAGES
            # =================================================

            scored_passages = sorted(
                scored_passages,
                key=lambda x: x["rerank_score"],
                reverse=True
            )

            fallback_passages = sorted(
                fallback_passages,
                key=lambda x: x["rerank_score"],
                reverse=True
            )

            # =================================================
            # FINAL PASSAGES — MIN 3 GUARANTEED
            # =================================================

            if len(scored_passages) >= 3:
                final_passages = scored_passages[:10]
                print("STRICT PASSAGES SELECTED")

            elif len(scored_passages) > 0:
                # scored-ல் 1-2 — fallback-ல் இருந்து fill
                final_passages = scored_passages[:]
                for p in fallback_passages:
                    if p not in final_passages:
                        final_passages.append(p)
                    if len(final_passages) >= 3:
                        break
                final_passages = sorted(
                    final_passages,
                    key=lambda x: x["rerank_score"],
                    reverse=True
                )[:10]
                print("SCORED + FALLBACK FILL")

            elif len(fallback_passages) >= 3:
                final_passages = fallback_passages[:10]
                print("FALLBACK PASSAGES SELECTED")

            else:
                # Last resort — emergency top 3
                emergency_sorted = sorted(
                    emergency_passages,
                    key=lambda x: x["rerank_score"],
                    reverse=True
                )
                final_passages = emergency_sorted[:3]
                print("EMERGENCY TOP 3 SELECTED")

            # =================================================
            # HANDLE EMPTY (safety net — வராது)
            # =================================================

            if not final_passages:

                out.write(
                    "\nNO RELEVANT PASSAGES FOUND\n"
                )

                print(
                    "NO RELEVANT PASSAGES FOUND"
                )

                continue

            # =================================================
            # WRITE OUTPUT
            # =================================================

            for idx, item in enumerate(
                final_passages,
                start=1
            ):

                out.write(
                    f"\nPASSAGE {idx}:\n\n"
                )

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
                    f"PASSAGE_TYPE: "
                    f"{item['passage_type']}\n"
                )

                out.write(
                    "-" * 60 + "\n"
                )

# ============================================================
# FINISHED
# ============================================================

print("\n")
print("=" * 100)
print("ELIMINATE COMPLETED SUCCESSFULLY")
print("=" * 100)

print("Saved File:", output_file)