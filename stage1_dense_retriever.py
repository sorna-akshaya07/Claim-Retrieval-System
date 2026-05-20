# ============================================================
# STAGE 1 → DENSE RETRIEVAL (IMPROVED VERSION)
# ============================================================

# pip install wikipedia-api sentence-transformers faiss-cpu torch

print("STAGE 1 RETRIEVAL STARTED")

# ============================================================
# IMPORTS
# ============================================================

import wikipediaapi
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import re

# ============================================================
# LOAD BGE RETRIEVER MODEL
# ============================================================

print("Loading Retriever Model...")

retriever = SentenceTransformer(
    "BAAI/bge-small-en-v1.5"
)

print("Retriever Loaded Successfully")

# ============================================================
# WIKIPEDIA API
# ============================================================

wiki = wikipediaapi.Wikipedia(
    language="en",
    user_agent="Stage1Retriever/1.0"
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

claims_data = claims_data[:31]

print("Total Claims Loaded:", len(claims_data))

# ============================================================
# MANUAL WIKIPEDIA TITLES
# ============================================================

titles = [

    "Fergie",
    "Polow da Don",
    "M.I.L.F.$",
    "Life Goes On (Fergie song)",

    "Stephen King",
    "Stanley Kubrick",
    "The Shining (film)",
    "Where Blood and Fire Bring Rest",

    "Greater Swiss Mountain Dog",
    "Harrier (dog breed)",

    "No One's Gonna Love You",
    "Zombieland",
    "Ruben Fleischer",

    "Before I Go to Sleep (film)",
    "Nicole Kidman",

    "Marko Dmitrović",
    "La Liga",

    "Eatza Pizza",
    "Your Pie",
    "Colorado",
    "Georgia",

    "Carnegie Hall Tower",
    "Staten Island",

    "Paddy Breathnach",
    "Rome (TV series)",
    "Marcus Agrippa",

    "Gregg Rolie",
    "Rob Tyner",

    "Clydebuilt",
    "River Clyde",

    "Jack McFarland",
    "Tony Awards",
    "Sean Hayes",

    "Martin O'Neill",
    "Aston Villa F.C.",

    "2014 All-Pac-12 Conference football team",
    "Mark Helfrich",
    "Oregon Ducks football",

    "Nimavar school",
    "Nimavar Bazaar",
    "Iran",

    "Elzie Crisler Segar",
    "Captain Strong",

    "Counter-terrorism raids in Australia 2014",
    "Supermax prison Australia",

    "Kentucky Department of Corrections",
    "Kentucky River",

    "Stephanie March",
    "Innocence (film)",
    "Law & Order: Special Victims Unit",

    "José Echegaray",
    "Graham Swift",

    "Sian Gibson",
    "Peter Kay's Car Share",

    "The Handmaid's Tale (TV series)",
    "Margaret Atwood",

    "Potomac River",
    "Ashley Estates",
    "Brett Kavanaugh",

    "Giovanni Furno",
    "Swan of Catania",

    "Jimmy Garcia",
    "World Boxing Organization",

    "Vanessa Paradis",
    "Divinidylle Tour",

    "Superdrag",
    "Collective Soul",

    "Burlington Mall",
    "Square One Mall",
    "Middlesex County",

    "Pablo Ozuna",
    "Marcell Ozuna",

    "People v. Turner",
    "Aaron Persky",

    "Battle of Saipan",
    "Battle of Manila",

    "Fizz (cocktail)",
    "Painkiller (cocktail)",

    "Battle of Britain",
    "Battle of Malta",
    "World War II",

    "Poor Old Lu",
    "Shudder to Think",

    "Georg Cantor",
    "Vladimir Arnold",

    "DJ Maphorisa",
    "One Dance",
    "Drake",

    "The Wanted",
    "Word of Mouth (The Wanted album)",
    "Rihanna",

    "Reimar Horten",
    "Gotha Go 229",

    "Qutebrowser",
    "DuckDuckGo",

    "Carrie Coon",
    "Gone Girl (film)",
    "Missouri",

    "Jack Kevorkian",
    "Christy Canyon",

    "Underfist: Halloween Bash",
    "Cartoon Network",
    "Betty Cohen",

    "Florence and the Machine"
]

# ============================================================
# LOAD WIKIPEDIA PASSAGES
# ============================================================

def load_wikipedia_passages():

    passages = []

    loaded_titles = set()

    for title in titles:

        if title in loaded_titles:
            continue

        loaded_titles.add(title)

        try:

            print("Loading Wikipedia Page:", title)

            page = wiki.page(title)

            if page.exists():

                text = page.text[:15000]

                paragraphs = text.split("\n")

                for para in paragraphs:

                    para = para.strip()

                    if len(para) > 80:

                        passages.append(para)

        except Exception as e:

            print("Error:", e)

    return passages

# ============================================================
# LOAD ALL PASSAGES
# ============================================================

print("\nLoading Wikipedia Passages...")

all_passages = load_wikipedia_passages()

# ============================================================
# REMOVE DUPLICATES
# ============================================================

all_passages = list(set(all_passages))

print("\nTotal Unique Passages Loaded:", len(all_passages))

# ============================================================
# CREATE EMBEDDINGS
# ============================================================

print("\nCreating Passage Embeddings...")

passage_embeddings = retriever.encode(

    all_passages,

    batch_size=32,

    convert_to_numpy=True,

    normalize_embeddings=True,

    show_progress_bar=True
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
# RETRIEVE TOP 50 UNIQUE PASSAGES
# ============================================================

def retrieve_top50(subclaim, top_k=50):

    query = f"Retrieve evidence for: {subclaim}"

    query_embedding = retriever.encode(

        [query],

        convert_to_numpy=True,

        normalize_embeddings=True
    )

    scores, indices = index.search(

        query_embedding,

        top_k
    )

    retrieved = []

    seen = set()

    for score, idx in zip(scores[0], indices[0]):

        passage = all_passages[idx].strip()

        if passage not in seen:

            seen.add(passage)

            retrieved.append({

                "score": float(score),

                "passage": passage
            })

    return retrieved

# ============================================================
# OUTPUT FILE
# ============================================================

output_file = "stage1_output.txt"

# ============================================================
# MAIN PIPELINE
# ============================================================

with open(output_file, "w", encoding="utf-8") as out:

    for item in claims_data:

        out.write("\n")

        out.write("=" * 100)

        out.write("\n")

        out.write(
            f"\nCLAIM ID: {item['claim_id']}\n"
        )

        out.write(
            f"CLAIM: {item['claim']}\n"
        )

        for idx, sc in enumerate(item["subclaims"], start=1):

            out.write("\n")

            out.write("-" * 80)

            out.write("\n")

            out.write(
                f"\nSUBCLAIM {idx}: {sc}\n"
            )

            retrieved = retrieve_top50(
                sc,
                top_k=50
            )

            for p_idx, passage_item in enumerate(retrieved, start=1):

                out.write(
                    f"\nPASSAGE {p_idx}:\n"
                )

                

                out.write(
                    passage_item["passage"] + "\n"
                )

                out.write(
                    "-" * 60 + "\n"
                )

# ============================================================
# FINISHED
# ============================================================

print("\nSTAGE 1 COMPLETED SUCCESSFULLY")

print("Saved File:", output_file)