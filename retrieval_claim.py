# ============================================================
# INSTALL REQUIRED LIBRARIES
# ============================================================
# pip install wikipedia-api sentence-transformers faiss-cpu torch

print("SUBCLAIM RETRIEVAL PIPELINE RUNNING...")

# ============================================================
# IMPORTS
# ============================================================

import wikipediaapi
from sentence_transformers import SentenceTransformer

import faiss
import numpy as np

import re

# ============================================================
# LOAD BGE MODEL
# ============================================================

print("Loading BGE Retriever Model...")

retriever = SentenceTransformer(
    "BAAI/bge-small-en-v1.5"
)

print("Retriever Loaded Successfully")

# ============================================================
# WIKIPEDIA API
# ============================================================

wiki = wikipediaapi.Wikipedia(
    language="en",
    user_agent="RetrievalPipeline/1.0"
)

# ============================================================
# READ OUTPUT1.TXT
# ============================================================

input_file = "output1.txt"

with open(input_file, "r", encoding="utf-8") as f:

    content = f.read()

# ============================================================
# SPLIT CLAIM BLOCKS
# ============================================================

claim_blocks = content.split(
    "============================================================"
)

claims_data = []

for block in claim_blocks:

    block = block.strip()

    if not block:
        continue

    # --------------------------------------------------------

    claim_id_match = re.search(
        r'claim_id:\s*(.*)',
        block
    )

    claim_match = re.search(
        r'claim:\s*(.*)',
        block
    )

    if not claim_match:
        continue

    claim_id = claim_id_match.group(1).strip()

    claim_text = claim_match.group(1).strip()

    # --------------------------------------------------------

    subclaims = re.findall(
        r'\d+\.\s*(.*)',
        block
    )

    claims_data.append({

        "claim_id": claim_id,

        "claim": claim_text,

        "subclaims": subclaims
    })

# ============================================================
# LIMIT CLAIMS
# ============================================================

claims_data = claims_data[50:101]

print("Total Claims Loaded:", len(claims_data))

# ============================================================
# RELEVANCE FUNCTION
# ============================================================

def get_relevance(score):

    if score >= 0.75:
        return "Highly Relevant"

    elif score >= 0.45:
        return "Relevant"

    else:
        return "Irrelevant"

# ============================================================
# LOAD WIKIPEDIA PASSAGES
# ============================================================

def load_wikipedia_passages():

    titles = [

        "Fergie",
        "Polow da Don",
        "Life Goes On",
        "M.I.L.F.$",

        "Stephen King",
        "Stanley Kubrick",

        "Barack Obama",
        "Canada",
        "Hawaii",

        "Mike Tyson",
        "Bobby Stewart"
    ]

    passages = []

    for title in titles:

        try:

            print("Loading:", title)

            page = wiki.page(title)

            if page.exists():

                text = page.text[:8000]

                paragraphs = text.split("\n")

                for para in paragraphs:

                    para = para.strip()

                    if len(para) > 50:

                        passages.append(para)

        except Exception as e:

            print("Error:", e)

    return passages

# ============================================================
# LOAD PASSAGES
# ============================================================

print("Loading Wikipedia Passages...")

all_passages = load_wikipedia_passages()

print("Total Passages Loaded:", len(all_passages))

# ============================================================
# CREATE EMBEDDINGS
# ============================================================

print("Creating Embeddings...")

passage_embeddings = retriever.encode(

    all_passages,

    convert_to_numpy=True,

    normalize_embeddings=True
)

print("Embeddings Created Successfully")

# ============================================================
# BUILD FAISS INDEX
# ============================================================

dimension = passage_embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)

index.add(passage_embeddings)

print("FAISS Index Created Successfully")

# ============================================================
# RETRIEVE PASSAGES
# ============================================================

def retrieve_passages(subclaim, top_k=10):

    query_embedding = retriever.encode(

        [subclaim],

        convert_to_numpy=True,

        normalize_embeddings=True
    )

    scores, indices = index.search(
        query_embedding,
        top_k
    )

    results = []

    for score, idx in zip(scores[0], indices[0]):

        results.append({

            "passage": all_passages[idx],

            "score": float(score),

            "relevance": get_relevance(float(score))
        })

    return results

# ============================================================
# OUTPUT FILE
# ============================================================

output_file = "output_claims.txt"

# ============================================================
# MAIN PIPELINE
# ============================================================

with open(output_file, "a", encoding="utf-8") as out:

    for item in claims_data:

        out.write("\n")
        out.write("=" * 100)
        out.write("\n")

        # ====================================================
        # CLAIM DETAILS
        # ====================================================

        out.write(
            f"\nCLAIM ID: {item['claim_id']}\n"
        )

        out.write(
            f"CLAIM: {item['claim']}\n"
        )

        # ====================================================
        # SUBCLAIMS
        # ====================================================

        out.write("\nSUBCLAIMS:\n")

        for idx, sc in enumerate(item["subclaims"], start=1):

            out.write(f"{idx}. {sc}\n")

        # ====================================================
        # RETRIEVAL
        # ====================================================

        for idx, sc in enumerate(item["subclaims"], start=1):

            out.write("\n")
            out.write("-" * 80)
            out.write("\n")

            out.write(
                f"\nSUBCLAIM {idx}: {sc}\n"
            )

            retrieved = retrieve_passages(
                sc,
                top_k=10
            )

            # ------------------------------------------------

            for p_idx, passage_item in enumerate(retrieved, start=1):

                out.write(
                    f"\nPASSAGE {p_idx}:\n"
                )

                out.write(
                    passage_item["passage"] + "\n"
                )

                out.write(
                    "\nSIMILARITY SCORE: "
                    + str(round(
                        passage_item["score"], 4
                    ))
                    + "\n"
                )

                out.write(
                    "RELEVANCE: "
                    + passage_item["relevance"]
                    + "\n"
                )

                out.write(
                    "-" * 60 + "\n"
                )

# ============================================================
# FINISHED
# ============================================================

print("\nALL RESULTS SAVED SUCCESSFULLY")
print("Output File:", output_file)