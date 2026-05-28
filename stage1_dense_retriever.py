# ============================================================
# STAGE 1 → DENSE RETRIEVAL (FIXED VERSION)
# ============================================================

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

    claim_id = claim_id_match.group(1).strip() if claim_id_match else "unknown"
    claim_text = claim_match.group(1).strip()
    subclaims = re.findall(r'\d+\.\s*(.*)', block)

    claims_data.append({
        "claim_id":  claim_id,
        "claim":     claim_text,
        "subclaims": subclaims
    })

# ============================================================
# LIMIT TO 31 CLAIMS (claim_0 to claim_30)
# ============================================================

claims_data = claims_data[:31]

print("Total Claims Loaded:", len(claims_data))

# ============================================================
# MANUAL WIKIPEDIA TITLES (claim_0 to claim_30 coverage)
# ============================================================

titles = [
    # Claim 0 — Fergie, M.I.L.F.$, Polow da Don
    "Fergie (singer)",
    "M.I.L.F.$",
    "Polow da Don",
    "Life Goes On (Fergie song)",

    # Claim 1 — Stephen King, Kubrick, Where Blood and Fire
    "Stephen King",
    "The Shining (novel)",
    "The Shining (film)",
    "Stanley Kubrick",
    "Where Blood and Fire Bring Rest",
    "Plague Bearer (band)","Stanley Kubrick filmography",
    "Stephen King bibliography",
    "The Shining",
    "Godspeed You! Black Emperor",
    "F# A# ∞",
    "Lift Your Skinny Fists Like Antennas to Heaven",


    # Claim 2 — Dog breeds
    "Greater Swiss Mountain Dog",
    "Harrier (dog)",

    # Claim 3 — No One's Gonna Love You, Zombieland
    "No One's Gonna Love You",
    "Zombieland",
    "Ruben Fleischer",
    "Band of Horses",

    # Claim 4 — Before I Go to Sleep, Nicole Kidman
    "Before I Go to Sleep (film)",
    "Nicole Kidman",   
    "Nicole Kidman filmography",
    "Nicole Kidman discography",
    "S.J. Watson",

    # Claim 5 — Marko Dmitrović, La Liga
    "Marko Dmitrović",
    "La Liga",
    "Sevilla FC",

    # Claim 6 — Eatza Pizza, Your Pie
    "Eatza Pizza",
    "Your Pie",

    # Claim 7 — Carnegie Hall Tower, Staten Island
    "Carnegie Hall Tower",
    "Staten Island",
    "Marko Dmitrović", "La Liga",
    "Eatza Pizza", "Your Pie",

    # Claim 8 — Paddy Breathnach, Rome, Marcus Agrippa
    "Rome (TV series)",
    "Allen Leech",
    "Paddy Breathnach filmography",
    "Man About Dog",
    "Viva (film)",

    # Claim 9 — Gregg Rolie, Rob Tyner
    "Gregg Rolie",
    "Santana (band)",
    "Journey (band)",
    "Rob Tyner",
    "MC5",

    # Claim 10 — Clydebuilt, River Clyde
    "Scottish Maritime Museum",
    "Braehead",
    "River Clyde",
    "Renfrewshire",


    # Claim 11 — Jack McFarland, Sean Hayes, Tony Awards
    "Jack McFarland (Will & Grace)",
    "Sean Hayes",
    "64th Tony Awards",
    "Will & Grace",

    # Claim 12 — Martin O'Neill, Aston Villa
    "Martin O'Neill",
    "Aston Villa F.C.",
    "David O'Leary",

    # Claim 13 — Oregon Ducks, Mark Helfrich
    "2014 Oregon Ducks football team",
    "Mark Helfrich",

    # Claim 14 — Nimavar school, Iran
    "Nimavar",
    "Semnan Province",
    "Bazaar",
    "Iran",



    # Claim 15 — Elzie Crisler Segar, Popeye, Captain Strong
    "Elzie Crisler Segar",
    "Popeye",
    "Captain Strong",

   # Claim 16 — 2014 Sydney raids, Goulburn prison
    "2014 Endeavour Hills stabbings",
    "Operation Appleby",
    "Goulburn Correctional Centre",
    "Supermax (Australia)",
    

    # Claim 17 — Kentucky Dept of Corrections
    "Kentucky Department of Corrections",
    "Frankfort Kentucky",
    "Kentucky River",
    "Capital Plaza Tower Frankfort",  # ← actual building name
    

     # Claim 18 — Stephanie March, Innocence
    "Stephanie March",
    "Innocence (2013 film)",
    "Law & Order: Special Victims Unit",
    "Alexandra Cabot",

    

    # Claim 19 — José Echegaray, Graham Swift
    "José Echegaray",
    "Graham Swift",
    "Nobel Prize Literature 1904",

    # Claim 20 — Peter Kay's Car Share, Sian Gibson
    "Peter Kay's Car Share",
    "Sian Gibson",

    # Claim 21 — Handmaid's Tale, Margaret Atwood
    "The Handmaid's Tale (TV series)",
    "Margaret Atwood",

    # Claim 22 — Kavanaugh, Ashley Estates, Potomac
    "Brett Kavanaugh",
    "Ashley Estes Kavanaugh",
    "Chevy Chase Club",
    "Chevy Chase Maryland",
    "Potomac River",
 
    # Claim 23 — Giovanni Furno, Swan of Catania
    "Giovanni Furno",
    "Vincenzo Bellini",
    "Swan of Catania",
    "Conservatorio di Musica San Pietro a Majella",

 
    # Claim 24 — Jimmy Garcia boxing
    "Gabriel Ruelas",          
    "Genaro Hernández",       
    "WBO lightweight championship",
    "Jesse James Leija",
    "Super featherweight boxing", 
    "1995 in boxing",           
    "List of WBO champions",

    # Claim 25 — Alizée
    "Alizée",
    "Divinidylle",
    "Divinidylle Tour",
    "Alizée discography",


    # Claim 26 — Superdrag, Collective Soul
    "Superdrag",
    "Collective Soul",

     # Claim 27 — Burlington Mall, Square One Mall
    "Burlington Mall",
    "Square One Mall Saugus",
    "Route 128 Massachusetts",
    "Middlesex County Massachusetts",

    # Claim 28 — Pablo Ozuna, Marcell Ozuna
    "Pablo Ozuna",
    "Marcell Ozuna",

    # Claim 29 — Aaron Persky, People v. Turner
    "Aaron Persky",
    "People v. Turner",
    "Santa Clara County Superior Court",

    # Claim 30 — Battle of Saipan, Battle of Manila
    "Battle of Saipan",
    "Battle of Manila (1945)",

    # Claim 31 — Fizz, Painkiller cocktail
    "Fizz (cocktail)",
    "Painkiller (cocktail)",

    # Extra context
    "Double Dutchess",
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
                    if len(para) > 50:
                        passages.append(para)

        except Exception as e:
            print("Error:", e)

    return passages

# ============================================================
# LOAD ALL PASSAGES
# ============================================================

print("\nLoading Wikipedia Passages...")

all_passages = load_wikipedia_passages()
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
# FIX: RETRIEVE GUARANTEED TOP 50 UNIQUE PASSAGES
# ============================================================

def retrieve_top50(subclaim, parent_claim, top_k=50):

    # FIX 1: BGE correct query prefix + parent claim context
    query = (
        f"Represent this sentence for searching relevant passages: "
        f"{parent_claim} {subclaim}"
    )

    query_embedding = retriever.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    # FIX 2: search 3x to guarantee 50 unique
    search_k = min(top_k * 3, len(all_passages))

    scores, indices = index.search(
        query_embedding,
        search_k
    )

    retrieved = []
    seen = set()

    for score, idx in zip(scores[0], indices[0]):

        passage = all_passages[idx].strip()

        if passage not in seen:
            seen.add(passage)
            retrieved.append({
                "score":   float(score),
                "passage": passage
            })

        # FIX 3: exactly 50 stop
        if len(retrieved) == top_k:
            break

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
        out.write(f"\nCLAIM ID: {item['claim_id']}\n")
        out.write(f"CLAIM: {item['claim']}\n")

        for idx, sc in enumerate(item["subclaims"], start=1):

            out.write("\n")
            out.write("-" * 80)
            out.write("\n")
            out.write(f"\nSUBCLAIM {idx}: {sc}\n")

            # FIX: pass parent claim for context
            retrieved = retrieve_top50(
                subclaim=sc,
                parent_claim=item["claim"],
                top_k=50
            )

            print(
                f"  {item['claim_id']} | subclaim {idx} "
                f"→ {len(retrieved)} passages"
            )

            for p_idx, passage_item in enumerate(retrieved, start=1):

                out.write(f"\nPASSAGE {p_idx}:\n")
                out.write(
                    f"Retrieval Score: "
                    f"{passage_item['score']:.4f}\n\n"
                )
                out.write(passage_item["passage"] + "\n")
                out.write("-" * 60 + "\n")

# ============================================================
# FINISHED
# ============================================================

print("\nSTAGE 1 COMPLETED SUCCESSFULLY")
print("Saved File:", output_file)