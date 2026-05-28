# ============================================================
# PASSAGE TYPE COUNTER
# ============================================================

import re

# ============================================================
# READ STAGE2 OUTPUT
# ============================================================

input_file = "stage2_output.txt"

with open(input_file, "r", encoding="utf-8") as f:
    content = f.read()

# ============================================================
# SPLIT CLAIM BLOCKS
# ============================================================

claim_blocks = re.split(r'=+', content)

# ============================================================
# MAIN PIPELINE
# ============================================================

for block in claim_blocks:

    block = block.strip()

    if not block:
        continue

    # ========================================================
    # CLAIM
    # ========================================================

    claim_id_match = re.search(r'CLAIM ID:\s*(.*)', block)
    claim_match    = re.search(r'CLAIM:\s*(.*)',    block)

    if not claim_match:
        continue

    claim_id   = claim_id_match.group(1).strip()
    claim_text = claim_match.group(1).strip()

    print()
    print("=" * 80)
    print(f"CLAIM ID : {claim_id}")
    print(f"CLAIM    : {claim_text[:80]}...")

    # ========================================================
    # SPLIT SUBCLAIMS
    # ========================================================

    subclaim_sections = re.split(r'-{80}', block)

    claim_highly     = 0
    claim_relevant   = 0
    claim_irrelevant = 0

    for section in subclaim_sections:

        subclaim_match = re.search(r'SUBCLAIM.*?:\s*(.*)', section)

        if not subclaim_match:
            continue

        subclaim = subclaim_match.group(1).strip()

        # ====================================================
        # COUNT PASSAGE TYPES
        # ====================================================

        all_types = re.findall(r'PASSAGE_TYPE:\s*(.*)', section)

        highly_relevant  = 0
        relevant         = 0
        irrelevant       = 0

        for t in all_types:

            t = t.strip()

            if t == "Highly Relevant":
                highly_relevant += 1

            elif t == "Relevant":
                relevant += 1

            elif t == "Irrelevant":
                irrelevant += 1

        total = highly_relevant + relevant + irrelevant

        claim_highly     += highly_relevant
        claim_relevant   += relevant
        claim_irrelevant += irrelevant

        # ====================================================
        # PRINT SUBCLAIM COUNT
        # ====================================================

        print()
        print(f"  SUBCLAIM : {subclaim[:70]}...")
        print(f"  Highly Relevant : {highly_relevant}")
        print(f"  Relevant        : {relevant}")
        print(f"  Irrelevant      : {irrelevant}")
        print(f"  Total           : {total}")
        print(f"  {'-' * 50}")

    # ========================================================
    # CLAIM LEVEL TOTAL
    # ========================================================

    claim_total = claim_highly + claim_relevant + claim_irrelevant

    print()
    print(f"  CLAIM TOTAL SUMMARY:")
    print(f"  Highly Relevant : {claim_highly}")
    print(f"  Relevant        : {claim_relevant}")
    print(f"  Irrelevant      : {claim_irrelevant}")
    print(f"  Total           : {claim_total}")
    print("=" * 80)

# ============================================================
# FINISHED
# ============================================================

print()
print("COMPLETED")