# ============================================================
# STAGE 4 → CLAIM VERIFICATION
# LIGHTWEIGHT NLI VERIFICATION
# ============================================================

print("VERIFICATION STARTED")

import re
import torch
from transformers import pipeline

print("Loading Verification Model...")

nli_pipe = pipeline(
    "text-classification",
    model="cross-encoder/nli-deberta-v3-small",
    device=0 if torch.cuda.is_available() else -1
)

print("Verification Model Loaded")

input_file = "hard_adver_output.txt"

with open(input_file, "r", encoding="utf-8") as f:
    content = f.read()

claim_blocks = re.split(r'\n={80,}\n', content)

print("TOTAL CLAIMS:", len(claim_blocks))

output_file = "verification_output.txt"

def get_nli_label(claim, passage):
    input_text = f"{passage} [SEP] {claim}"
    result = nli_pipe(input_text, truncation=True, max_length=512)
    label = result[0]["label"].upper()
    if "ENTAIL" in label:
        return "SUPPORTED"
    elif "CONTRA" in label:
        return "REFUTED"
    else:
        return "NEI"

with open(output_file, "w", encoding="utf-8") as out:

    for block_idx, block in enumerate(claim_blocks):

        block = block.strip()
        if not block:
            continue

        print("\n")
        print("=" * 80)
        print(f"PROCESSING CLAIM {block_idx}")
        print("=" * 80)

        claim_id_match = re.search(r'CLAIM ID:\s*(.*)', block)
        claim_id = claim_id_match.group(1).strip() if claim_id_match else "UNKNOWN"

        claim_match = re.search(r'CLAIM:\s*(.*)', block)
        if not claim_match:
            continue

        claim = claim_match.group(1).strip()
        print("CLAIM:", claim)

        out.write("\n")
        out.write("=" * 100)
        out.write("\n\n")
        out.write(f"CLAIM ID: {claim_id}\n")
        out.write(f"CLAIM: {claim}\n")

        subclaim_sections = re.split(r'\n-{80}\n', block)
        subclaim_verdicts = []
        subclaim_num = 0

        for section_idx, section in enumerate(subclaim_sections):

            section = section.strip()
            if not section:
                continue

            subclaim_match = re.search(r'SUBCLAIM:\s*(.*)', section)
            if not subclaim_match:
                continue

            subclaim = subclaim_match.group(1).strip()

            if subclaim.lower().rstrip(":").strip() == "sub-claims":
                continue

            subclaim_num += 1

            print("\n")
            print("-" * 80)
            print(f"SUBCLAIM {subclaim_num}: {subclaim}")
            print("-" * 80)

            out.write("\n")
            out.write("-" * 80)
            out.write("\n\n")
            out.write(f"SUBCLAIM {subclaim_num}: {subclaim}\n\n")

            # FIX: flexible regex
            passages = re.findall(
                r'PASSAGE\s+\d+:\s*\n+(.*?)\n+RETRIEVAL SCORE:',
                section,
                re.DOTALL
            )

            print("TOTAL PASSAGES:", len(passages))

            support_count = 0
            refute_count  = 0
            nei_count     = 0

            for idx, passage in enumerate(passages):

                passage = passage.strip()
                if len(passage) < 20:
                    continue

                verdict = get_nli_label(subclaim, passage)

                print(f"PASSAGE {idx+1} VERDICT: {verdict}")

                out.write(f"PASSAGE {idx+1}:\n\n")
                out.write(passage + "\n\n")
                out.write(f"NLI VERDICT: {verdict}\n")
                out.write("-" * 60 + "\n")

                if verdict == "SUPPORTED":
                    support_count += 1
                elif verdict == "REFUTED":
                    refute_count += 1
                else:
                    nei_count += 1

            # MAJORITY VOTE
            total = support_count + refute_count + nei_count

            if total == 0:
                final_subclaim_verdict = "NOT ENOUGH INFO"

            elif refute_count >= 2:
                final_subclaim_verdict = "REFUTED"

            elif support_count >= 2 and refute_count == 0:
                final_subclaim_verdict = "SUPPORTED"

            elif support_count >= 2 and refute_count == 1:
                final_subclaim_verdict = "SUPPORTED"

            elif support_count == refute_count and support_count >= 1:
                final_subclaim_verdict = "REFUTED"

            elif refute_count >= 1 and support_count == 0:
                final_subclaim_verdict = "REFUTED"

            else:
                final_subclaim_verdict = "NOT ENOUGH INFO"

            subclaim_verdicts.append(final_subclaim_verdict)

            out.write(f"\nVERDICT: {final_subclaim_verdict}\n")
            print("FINAL SUBCLAIM VERDICT:", final_subclaim_verdict)

        # FINAL CLAIM VERDICT
        supported_count = subclaim_verdicts.count("SUPPORTED")
        refuted_count   = subclaim_verdicts.count("REFUTED")
        total_subclaims = len(subclaim_verdicts)

        if total_subclaims == 0:
            final_claim_verdict = "NOT ENOUGH INFO"
        elif refuted_count >= 1:
            final_claim_verdict = "REFUTED"
        elif supported_count == total_subclaims:
            final_claim_verdict = "SUPPORTED"
        else:
            final_claim_verdict = "NOT ENOUGH INFO"

        out.write("\n")
        out.write("=" * 60)
        out.write("\n\n")
        out.write(f"FINAL CLAIM VERDICT: {final_claim_verdict}\n\n")

        print("\nFINAL CLAIM VERDICT:", final_claim_verdict)

print("\n")
print("=" * 100)
print("VERIFICATION COMPLETED")
print("=" * 100)
print("Saved File:", output_file)