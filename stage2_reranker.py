# ============================================================
# STAGE 2 → RERANKING (FIXED VERSION)
# ============================================================
 
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
 
    elif score >= 0.50:
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
            r'^SUBCLAIM\s+\d+:\s*(.*)',
            section,
            re.MULTILINE
            )
 
            if not subclaim_match:
                continue
 
            subclaim = subclaim_match.group(1).strip()
 
            if subclaim.lower().rstrip(':').strip() == 'sub-claims':
                continue
 
            out.write("\n")
            out.write("-" * 80)
            out.write("\n")
            subclaim_num = re.search(r'\d+', subclaim_match.group(0))
            num = subclaim_num.group(0) if subclaim_num else ""
            out.write(f"\nSUBCLAIM {num}: {subclaim}\n")
 
            # =================================================
            # FIX: EXTRACT PASSAGES WITH RETRIEVAL SCORE
            # =================================================
 
            passage_matches = re.findall(
                r'PASSAGE\s+\d+:\nRetrieval Score:\s*([\d.]+)\n\n(.*?)\n-{60}',
                section,
                re.DOTALL
            )
 
            scored_passages = []
 
            # =================================================
            # SCORE EACH PASSAGE (TOP 20 ONLY)
            # =================================================
 
            for retrieval_score_str, passage_text in passage_matches[:20]:
 
                passage_text = passage_text.strip()
 
                if len(passage_text) < 20:
                    continue
 
                retrieval_score = float(retrieval_score_str)
 
                rerank = rerank_score(f"{claim_text} {subclaim}", passage_text)
 
                label = get_label(rerank)
 
                scored_passages.append({
                    "passage":          passage_text,
                    "retrieval_score":  retrieval_score,
                    "rerank_score":     rerank,
                    "passage_type":     label
                })
 
                print(
                    f"  Subclaim: {subclaim[:50]}... | "
                    f"Retrieval: {retrieval_score:.4f} | "
                    f"Rerank: {rerank:.4f} | "
                    f"Label: {label}"
                )
 
            # =================================================
            # SORT BY RERANK SCORE
            # =================================================
 
            scored_passages = sorted(
                scored_passages,
                key=lambda x: x["rerank_score"],
                reverse=True
            )
 
            # =================================================
            # WRITE OUTPUT
            # =================================================
 
            for idx, item in enumerate(scored_passages, start=1):
 
                out.write(f"\nPASSAGE {idx}:\n")
                out.write(item["passage"] + "\n\n")
                out.write(f"RETRIEVAL SCORE: {item['retrieval_score']:.4f}\n")
                out.write(f"RERANK SCORE:    {item['rerank_score']:.4f}\n")
                out.write(f"PASSAGE_TYPE:    {item['passage_type']}\n")
                out.write("-" * 60 + "\n")
 
# ============================================================
# FINISHED
# ============================================================
 
print("\nSTAGE 2 COMPLETED SUCCESSFULLY")
print("Saved File:", output_file)