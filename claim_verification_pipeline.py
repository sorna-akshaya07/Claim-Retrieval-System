# ============================================================
# FULL PIPELINE — CLAIM VERIFICATION SYSTEM
# Stage 1 : Sub-claim Generation   (Qwen via Groq)
# Stage 2 : Dense Retrieval        (Dynamic Wikipedia, Top 50)
# Stage 3 : Reranking (Top 20) → Hard Negative + Adversarial Elimination (Top 10)
# Stage 4 : Verification           (NLI)
# ============================================================

# ============================================================
# IMPORTS
# ============================================================

import os
import re
import json
import time
import torch
import faiss
import numpy as np
import wikipediaapi
import wikipedia

from groq import Groq
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline
)

# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()
groq_key = os.getenv("GROQ_API_KEY")

# ============================================================
# OUTPUT FILE — all stage results are saved here
# ============================================================

OUTPUT_FILE = "pipeline_output.txt"

# ============================================================
# GROQ CLIENT
# ============================================================

groq_client = Groq(api_key=groq_key)

# ============================================================
# USER INPUT
# Accepts claim from terminal (option 1) or from file (option 2)
# Supported file formats: .jsonl  |  .txt
# ============================================================

def show_menu():
    """Print the main menu."""
    print("\n" + "=" * 60)
    print("   CLAIM VERIFICATION PIPELINE")
    print("=" * 60)
    print("  1 -> Enter a claim directly in the terminal")
    print("  2 -> Provide a file path  (.jsonl or .txt)")
    print("  3 -> Exit")
    print("-" * 60)


def load_claims_from_file(file_path):
    """
    Load claims from a .jsonl or .txt file.
    Returns a list of dicts: [{\"claim_id\": ..., \"claim\": ...}, ...]
    """
    claims = []

    if not os.path.exists(file_path):
        print(f"[ERROR] File not found: {file_path}")
        return []

    if file_path.endswith(".jsonl"):
        with open(file_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                # Use exact claim_id from file — fallback to index only if missing
                claim_id = (
                    data.get("claim_id") or
                    data.get("id")       or
                    data.get("ID")       or
                    str(idx)
                )
                label_raw = data.get("label", None)
                if label_raw is not None:
                    label_map    = {0: "SUPPORTED", 1: "REFUTED", 2: "NOT ENOUGH INFO"}
                    ground_truth = label_map.get(int(label_raw), str(label_raw))
                else:
                    ground_truth = None

                claims.append({
                    "claim_id":     str(claim_id),
                    "claim":        data.get("claim", "").strip(),
                    "ground_truth": ground_truth
                })

    elif file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if line:
                    claims.append({
                        "claim_id": str(idx),
                        "claim":    line
                    })

    else:
        print("[ERROR] Unsupported file format. Please use .jsonl or .txt")
        return []

    print(f"[INFO] {len(claims)} claim(s) loaded from file.")
    return claims



def clean_output(text):
    """
    Remove all <think> ... </think> blocks from model output.
    Handles both complete and incomplete think blocks robustly.
    """
    # Remove complete <think>...</think> blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Remove any leftover opening <think> tag and everything after it
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL)
    return text.strip()


def generate_subclaims(claim):
    """
    Send a claim to the Qwen model (via Groq) and receive a structured
    list of sub-claims along with a reasoning chain.
    Uses /no_think flag and higher max_tokens to avoid truncated think blocks.
    """
    prompt = f"""
TASK:
Decompose the given claim into clear, logical sub-claims and build a reasoning chain.

RULES:
SUB-CLAIM RULES:
1. Break the claim into clear, logical sub-claims
2. Each sub-claim must contain ONLY one idea
3. Sub-claims need NOT match exact wording
4. Do NOT add new or external information
5. Do NOT assume anything not in the claim
6. Each sub-claim must be independently fact-checkable
7. Ensure decomposition is meaningful

REASONING RULES:
8. Arrange sub-claims in logical order
9. Each sub-claim must support or lead to the next
10. Build a reasoning chain using sub-claim numbers
11. Use format: 1 -> 2 -> 3 -> Final Claim
12. Do NOT write "Step 1", "Step 2"
13. Use only numbers referring to sub-claims

IMPORTANT:
- DO NOT output thinking process
- DO NOT include <think> tags
- Return ONLY the final formatted answer below

OUTPUT FORMAT:
Sub-claims:
1. ...
2. ...
3. ...

Reasoning Chain:
1 -> 2 -> 3 -> Final Claim

CLAIM:
{claim}
/no_think
"""

    response = groq_client.chat.completions.create(
        model="qwen/qwen3-32b",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a strict claim decomposition engine. "
                    "Never output chain-of-thought or reasoning steps. "
                    "Never use <think> tags. "
                    "Return only the final formatted answer."
                )
            },
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=600   # increased so think block doesn't get truncated mid-way
    )

    raw = response.choices[0].message.content

    # Guard against None response
    if not raw:
        raise ValueError("Groq API returned empty response.")

    # Strategy 1: split on </think> — take everything after it
    if "</think>" in raw:
        raw = raw.split("</think>")[-1].strip()

    # Strategy 2: if <think> still present (incomplete block), cut from there
    if "<think>" in raw:
        raw = raw.split("<think>")[0].strip()

    return clean_output(raw)


def parse_subclaims(text):
    """Extract individual sub-claim strings from the model output."""
    if not text:
        return []
    return re.findall(r'\d+\.\s*(.*)', text)


def parse_reasoning_chain(text):
    """Extract the reasoning chain line from sub-claim output."""
    if not text:
        return ""
    match = re.search(r"Reasoning Chain[:\s]*(.*)", text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ""


# ============================================================
# STAGE 2 — DYNAMIC WIKIPEDIA RETRIEVAL
# Extracts keywords from the claim and sub-claims, searches
# Wikipedia automatically, and returns the top 50 passages
# via FAISS dense retrieval. No hardcoded page titles needed.
# ============================================================

wiki_wiki = wikipediaapi.Wikipedia(
    language="en",
    user_agent="ClaimVerificationPipeline/1.0"
)


def extract_keywords(claim, subclaims):
    """
    Extract search keywords from claim and subclaims.

    Priority order:
    1. Words inside parentheses  — e.g. (South Australia), (1933 film)
    2. Multi-word named entities — e.g. "Margaret Thatcher", "Klemzig"
    3. Single capitalised words  — e.g. "Brooklyn", "Fergie"
    4. Content words fallback    — if above give too few results
    """
    stop_words = {
        "the", "a", "an", "is", "was", "were", "are", "be",
        "been", "being", "have", "has", "had", "do", "does",
        "did", "will", "would", "shall", "should", "may",
        "might", "must", "can", "could", "and", "or", "but",
        "in", "on", "at", "to", "for", "of", "with", "by",
        "from", "that", "this", "which", "who", "whom",
        "whose", "what", "where", "when", "how", "not",
        "no", "nor", "so", "yet", "both", "either", "neither",
        "each", "than", "too", "very", "just", "about", "also",
        "same", "common", "suburb", "road", "city", "town",
        "north", "south", "east", "west", "bounded", "includes",
        "located", "known", "called", "named", "born", "died",
        "film", "movie", "song", "album", "book", "show"
    }

    all_text = claim + " " + " ".join(subclaims)

    # Priority 1: Words inside parentheses (most specific)
    paren_terms = re.findall(r'\(([^)]+)\)', all_text)

    # Priority 2: Multi-word named entities (e.g. "South Australia")
    multi_entities = re.findall(
        r'\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)+\b', all_text
    )

    # Priority 3: Single capitalised words
    single_caps = re.findall(r'\b[A-Z][a-z]{2,}\b', all_text)

    # Priority 4: Content words (fallback)
    all_words = re.findall(r'\b[a-zA-Z]{4,}\b', all_text)
    content_words = [w for w in all_words if w.lower() not in stop_words]

    # Build final list in priority order
    candidates = []
    for t in paren_terms:
        candidates.append(t.strip())
    for t in multi_entities:
        candidates.append(t.strip())
    for t in single_caps:
        candidates.append(t.strip())

    # Deduplicate
    seen, unique_queries = set(), []
    for q in candidates:
        q = q.strip().rstrip(",")
        if q.lower() not in seen and len(q) > 3:
            seen.add(q.lower())
            unique_queries.append(q)

    # Fallback — add content words if not enough
    if len(unique_queries) < 4:
        for w in content_words:
            if w.lower() not in seen:
                seen.add(w.lower())
                unique_queries.append(w)
            if len(unique_queries) >= 8:
                break

    return unique_queries[:8]


def load_page_passages(title, loaded_titles, seen_texts, all_passages):
    """Load a single Wikipedia page and extract passages."""
    if title in loaded_titles:
        return
    loaded_titles.add(title)
    try:
        page = wiki_wiki.page(title)
        if page.exists():
            text       = page.text[:15000]
            paragraphs = text.split("\n")
            for para in paragraphs:
                para = para.strip()
                if len(para) > 50 and para not in seen_texts:
                    seen_texts.add(para)
                    all_passages.append({
                        "text":   para,
                        "source": title
                    })
            return True
    except Exception as e:
        print(f"  [WARNING] Failed to load page '{title}': {e}")
    return False


def extract_wikipedia_titles(claim, subclaims):
    """
    Use Qwen model to extract relevant Wikipedia page titles
    directly from the claim and sub-claims.
    Returns a list of Wikipedia page title strings.
    """
    subclaims_text = "\n".join([f"{i+1}. {sc}" for i, sc in enumerate(subclaims)])

    prompt = f"""Given this claim and its sub-claims, list the most relevant Wikipedia page titles needed to verify them.

CLAIM: {claim}

SUB-CLAIMS:
{subclaims_text}

RULES:
- Return ONLY Wikipedia page titles, one per line
- Use exact Wikipedia page title format (e.g. "Aston Villa F.C." not "Aston Villa")
- Return 5 to 8 titles maximum
- No numbering, no bullets, no explanation
- Only titles that directly help verify the claim

OUTPUT:"""

    try:
        response = groq_client.chat.completions.create(
            model="qwen/qwen3-32b",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a Wikipedia title extractor. "
                        "Return only Wikipedia page titles, one per line. "
                        "No thinking, no explanation, no formatting."
                    )
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=200
        )

        raw = response.choices[0].message.content or ""

        # Strip think blocks
        if "</think>" in raw:
            raw = raw.split("</think>")[-1].strip()
        if "<think>" in raw:
            raw = raw.split("<think>")[0].strip()

        # Extract titles — one per line
        titles = []
        for line in raw.splitlines():
            line = line.strip()
            # Skip empty, numbered, or bulleted lines
            line = re.sub(r"^[\d\.\-\*\•]+\s*", "", line).strip()
            if len(line) > 3:
                titles.append(line)

        print(f"  [INFO] Wikipedia titles from Qwen: {titles}")
        return titles[:8]

    except Exception as e:
        print(f"  [WARNING] Qwen title extraction failed: {e}")
        # Fallback to keyword extraction
        return extract_keywords(claim, subclaims)


def fetch_wikipedia_passages(claim, subclaims):
    """
    Use Qwen model to extract relevant Wikipedia page titles,
    then directly load those pages — same as original hardcoded approach
    but fully automatic.
    """
    # Step 1: Qwen extracts Wikipedia titles for this claim
    titles = extract_wikipedia_titles(claim, subclaims)

    all_passages  = []
    loaded_titles = set()
    seen_texts    = set()

    # Step 2: Direct page load — no search needed
    for title in titles:
        load_page_passages(title, loaded_titles, seen_texts, all_passages)
        time.sleep(0.3)

    print(f"  [INFO] Total unique passages loaded: {len(all_passages)}")
    return all_passages



def retrieve_top50(subclaim, claim, passages, retriever_model):
    """
    Encode passages and query with BGE, build a FAISS index,
    and return the top 50 most relevant passages.
    """
    if not passages:
        return []

    # Support both plain strings and dicts (with source field)
    if passages and isinstance(passages[0], dict):
        texts   = [p["text"] for p in passages]
        sources = [p["source"] for p in passages]
    else:
        texts   = [p for p in passages]
        sources = ["Wikipedia"] * len(passages)

    passage_embeddings = retriever_model.encode(
        texts,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    dimension = passage_embeddings.shape[1]
    index     = faiss.IndexFlatIP(dimension)
    index.add(passage_embeddings)

    query = (
        f"Represent this sentence for searching relevant passages: "
        f"{claim} {subclaim}"
    )
    query_embedding = retriever_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    search_k        = min(50 * 3, len(passages))
    scores, indices = index.search(query_embedding, search_k)

    retrieved, seen = [], set()
    for score, idx in zip(scores[0], indices[0]):
        passage_text = texts[idx].strip()
        if passage_text not in seen:
            seen.add(passage_text)
            retrieved.append({
                "score":   float(score),
                "passage": passage_text,
                "source":  sources[idx]
            })
        if len(retrieved) == 50:
            break

    return retrieved


# ============================================================
# STAGE 3 — RERANKING + HARD NEGATIVE / ADVERSARIAL ELIMINATION
# 3A: BGE reranker scores each passage; top 20 are kept.
# 3B: Low-scoring (hard negative) and contradicting (adversarial)
#     passages are removed; top 10 are returned.
# ============================================================

def rerank_score(query, passage, tokenizer, model):
    """Return a sigmoid-normalised relevance score (0–1)."""
    inputs = tokenizer(
        query, passage,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )
    with torch.no_grad():
        score = torch.sigmoid(model(**inputs).logits).item()
    return score


def rerank_score_raw(query, passage, tokenizer, model):
    """Return the raw logit score used during elimination."""
    inputs = tokenizer(
        query, passage,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )
    with torch.no_grad():
        score = model(**inputs).logits.item()
    return score


def get_passage_label(score):
    """Map sigmoid rerank score to a human-readable relevance label."""
    if score >= 0.85:
        return "Highly Relevant"
    elif score >= 0.50:
        return "Relevant"
    else:
        return "Irrelevant"


def get_elimination_label(score):
    """Map raw logit score to a passage-type label used in elimination."""
    if score >= 6:
        return "Highly Relevant"
    elif score >= 3:
        return "Relevant"
    elif score >= 1:
        return "Weakly Relevant"
    else:
        return "Irrelevant"


def get_nli_label(subclaim, passage, nli_model):
    """
    Run NLI to detect contradicting passages.
    Returns: ENTAILMENT | CONTRADICTION | NEUTRAL
    """
    input_text = f"{passage} [SEP] {subclaim}"
    result     = nli_model(input_text, truncation=True, max_length=512)
    label      = result[0]["label"].upper()
    if "ENTAIL" in label:
        return "ENTAILMENT"
    elif "CONTRA" in label:
        return "CONTRADICTION"
    else:
        return "NEUTRAL"


def rerank_top20(subclaim, claim, passages_top50, rerank_tok, rerank_mod):
    """
    Score each of the top-50 passages with the BGE reranker and
    return the top 20 sorted by rerank score (descending).
    """
    query  = f"{claim} {subclaim}"
    scored = []

    for p in passages_top50[:20]:
        passage_text = p["passage"]
        if len(passage_text) < 20:
            continue

        score = rerank_score(query, passage_text, rerank_tok, rerank_mod)
        label = get_passage_label(score)

        scored.append({
            "passage":          passage_text,
            "retrieval_score":  p["score"],
            "rerank_score":     score,
            "passage_type":     label,
            "source":           p.get("source", "Wikipedia")
        })

    scored.sort(key=lambda x: x["rerank_score"], reverse=True)
    return scored[:20]


def eliminate_hard_negatives(subclaim, passages_top20, rerank_tok, rerank_mod, nli_mod):
    """
    Remove hard negatives (low raw score) and adversarial distractors
    (NLI contradiction). Returns up to 10 passages; minimum 3 guaranteed
    via fallback and emergency selection.
    """
    scored_passages   = []
    fallback_passages = []
    emergency_passages = []

    for p in passages_top20:
        passage = p["passage"]
        if len(passage) < 20:
            continue

        score = rerank_score_raw(subclaim, passage, rerank_tok, rerank_mod)
        ptype = get_elimination_label(score)

        # Always save to emergency pool before any filtering
        emergency_passages.append({
            "passage":          passage,
            "retrieval_score":  p["retrieval_score"],
            "rerank_score":     score,
            "passage_type":     ptype,
            "nli_label":        "NEUTRAL",
            "source":           p.get("source", "Wikipedia")
        })

        # Hard negative — score too low
        if score < 1:
            print(f"  [REMOVED] Hard negative  (score={score:.4f})")
            continue

        # Adversarial distractor — NLI contradiction
        nli_label = get_nli_label(subclaim, passage, nli_mod)
        if nli_label == "CONTRADICTION":
            print(f"  [REMOVED] Adversarial distractor (NLI=CONTRADICTION)")
            continue

        fallback_passages.append({
            "passage":          passage,
            "retrieval_score":  p["retrieval_score"],
            "rerank_score":     score,
            "passage_type":     ptype,
            "nli_label":        nli_label,
            "source":           p.get("source", "Wikipedia")
        })

        # Strict filter
        if score >= 2 and ptype != "Irrelevant":
            scored_passages.append({
                "passage":          passage,
                "retrieval_score":  p["retrieval_score"],
                "rerank_score":     score,
                "passage_type":     ptype,
                "nli_label":        nli_label,
                "source":           p.get("source", "Wikipedia")
            })

    scored_passages.sort(key=lambda x: x["rerank_score"],   reverse=True)
    fallback_passages.sort(key=lambda x: x["rerank_score"], reverse=True)

    if len(scored_passages) >= 3:
        final = scored_passages[:10]
        print(f"  [INFO] Strict filter applied — {len(final)} passages selected.")

    elif len(scored_passages) > 0:
        final = scored_passages[:]
        for p in fallback_passages:
            if p not in final:
                final.append(p)
            if len(final) >= 3:
                break
        final.sort(key=lambda x: x["rerank_score"], reverse=True)
        final = final[:10]
        print(f"  [INFO] Scored + fallback merge — {len(final)} passages selected.")

    elif len(fallback_passages) >= 3:
        final = fallback_passages[:10]
        print(f"  [INFO] Fallback passages used — {len(final)} passages selected.")

    else:
        emergency_passages.sort(key=lambda x: x["rerank_score"], reverse=True)
        final = emergency_passages[:3]
        print(f"  [WARNING] Emergency top-3 selected (insufficient passages after filtering).")

    return final


# ============================================================
# STAGE 4 — CLAIM VERIFICATION
# Runs NLI on each top-10 passage to determine per-passage
# verdict, then aggregates via majority vote per sub-claim
# and across all sub-claims for the final claim verdict.
# ============================================================

def verify_claim_nli(subclaim, passage, nli_model):
    """
    Classify the relationship between a passage and a sub-claim.

    Uses score thresholds instead of raw label to reduce false NEI:
    - ENTAILMENT score >= 0.40  → SUPPORTED
    - CONTRADICTION score >= 0.40 → REFUTED
    - Otherwise                 → NEI

    Returns: SUPPORTED | REFUTED | NEI
    """
    input_text = f"{passage} [SEP] {subclaim}"
    results    = nli_model(
        input_text,
        truncation=True,
        max_length=512,
        top_k=None          # get all label scores
    )

    # Build a score dict for easy lookup
    scores = {r["label"].upper(): r["score"] for r in results}

    entail_score  = scores.get("ENTAILMENT",    scores.get("ENTAIL",    0.0))
    contra_score  = scores.get("CONTRADICTION", scores.get("CONTRADICT", 0.0))
    neutral_score = scores.get("NEUTRAL",       scores.get("NEI",        0.0))

    # Lower threshold so relevant but indirect passages count as SUPPORTED
    SUPPORT_THRESHOLD = 0.40
    REFUTE_THRESHOLD  = 0.40

    if entail_score >= SUPPORT_THRESHOLD and entail_score >= contra_score:
        return "SUPPORTED"
    elif contra_score >= REFUTE_THRESHOLD and contra_score >= entail_score:
        return "REFUTED"
    else:
        return "NEI"


def get_subclaim_verdict(verdicts):
    """
    Determine the final verdict for a single sub-claim using majority vote.
    Returns: SUPPORTED | REFUTED | NOT ENOUGH INFO
    """
    supported = verdicts.count("SUPPORTED")
    refuted   = verdicts.count("REFUTED")
    total     = len(verdicts)

    if total == 0:
        return "NOT ENOUGH INFO"
    elif supported > refuted and supported >= 1:
        return "SUPPORTED"
    elif refuted > supported and refuted >= 1:
        return "REFUTED"
    else:
        return "NOT ENOUGH INFO"


def get_final_claim_verdict(subclaim_verdicts):
    """
    Aggregate sub-claim verdicts into a single claim-level verdict.
    A single REFUTED sub-claim marks the entire claim as REFUTED.
    Returns: SUPPORTED | REFUTED | NOT ENOUGH INFO
    """
    supported = subclaim_verdicts.count("SUPPORTED")
    refuted   = subclaim_verdicts.count("REFUTED")
    total     = len(subclaim_verdicts)

    if total == 0:
        return "NOT ENOUGH INFO"
    elif refuted >= 1:
        return "REFUTED"
    elif supported == total:
        return "SUPPORTED"
    else:
        return "NOT ENOUGH INFO"


# ============================================================
# MAIN PIPELINE
# Orchestrates all four stages end-to-end.
# ============================================================

def process_claims(claims, out, retriever_model, rerank_tokenizer, rerank_model, nli_model):
    """
    Run all 4 pipeline stages for a given list of claims.
    Results are written to the already-open output file handle `out`.
    """
    for claim_idx, claim_data in enumerate(claims):

        claim_id   = claim_data["claim_id"]
        claim_text = claim_data["claim"]

        print("\n" + "=" * 70)
        print(f"[CLAIM {claim_idx + 1}/{len(claims)}] ID: {claim_id}")
        print(f"TEXT: {claim_text}")
        print("=" * 70)

        # ── Stage 1: Sub-claim generation ──────────────────────
        print("\n[STAGE 1] Generating sub-claims...")
        try:
            subclaim_output = generate_subclaims(claim_text)
            subclaims       = parse_subclaims(subclaim_output)
            print(f"  [INFO] {len(subclaims)} sub-claim(s) generated.")
            time.sleep(1)
        except Exception as e:
            print(f"  [ERROR] Sub-claim generation failed: {e}")
            continue

        if not subclaims:
            print("  [WARNING] No sub-claims parsed. Skipping.")
            continue

        # Write claim header only after successful sub-claim generation
        out.write("\n" + "=" * 100 + "\n")
        out.write(f"CLAIM ID : {claim_id}\n")
        out.write(f'CLAIM    : "{claim_text}"\n')

        # Save reasoning chain separately; strip it from stage1 output
        reasoning_chain_text = parse_reasoning_chain(subclaim_output)

        # Remove reasoning chain lines from stage1 output before writing
        # Catches: "Reasoning Chain:", "1 -> 2 -> 3", "1 → 2 → 3" patterns
        stage1_lines = []
        for line in subclaim_output.splitlines():
            stripped = line.strip()
            # Skip "Reasoning Chain:" label line
            if re.search(r"reasoning chain", stripped, re.IGNORECASE):
                continue
            # Skip lines that are ONLY numbers + arrows (the chain line itself)
            if re.match(r"^[0-9\s\->?→]+(?:Final Claim)?$", stripped):
                continue
            stage1_lines.append(line)
        stage1_clean = "\n".join(stage1_lines).strip()

        # Stage 1 sub-claims written directly below claim — no header needed

        # ── Wikipedia fetch (once per claim) ───────────────────
        print("\n[INFO] Fetching Wikipedia passages dynamically...")
        all_passages = fetch_wikipedia_passages(claim_text, subclaims)

        if not all_passages:
            print("  [WARNING] No Wikipedia passages retrieved. Skipping.")
            out.write("\n[WARNING] No Wikipedia passages fetched.\n")
            continue

        # ── Per sub-claim processing ───────────────────────────
        subclaim_verdicts       = []
        global_passage_counter  = [1]   # mutable list so inner scope can modify

        for sc_idx, subclaim in enumerate(subclaims, start=1):

            print(f"\n{'─' * 60}")
            print(f"[SUB-CLAIM {sc_idx}] {subclaim}")
            print(f"{'─' * 60}")

            out.write(f"\n{'─' * 80}\n")
            out.write(f"SUB-CLAIM {sc_idx}: {subclaim}\n")

            # Stage 2 — Dense retrieval
            print(f"\n  [STAGE 2] Dense retrieval — top 50...")
            top50 = retrieve_top50(subclaim, claim_text, all_passages, retriever_model)
            print(f"  [INFO] {len(top50)} passages retrieved.")

            # Stage 3A — Reranking
            print(f"\n  [STAGE 3A] Reranking top 50 -> top 20...")
            top20 = rerank_top20(subclaim, claim_text, top50, rerank_tokenizer, rerank_model)
            print(f"  [INFO] {len(top20)} passages after reranking.")

            # Stage 3B — Elimination
            print(f"\n  [STAGE 3B] Eliminating hard negatives and adversarial distractors...")
            top10 = eliminate_hard_negatives(subclaim, top20, rerank_tokenizer, rerank_model, nli_model)
            print(f"  [INFO] {len(top10)} passages retained after elimination.")

            # Stage 4 — Verification (run on all top10)
            print(f"\n  [STAGE 4] Verifying sub-claim...")
            passage_verdicts = []
            for p in top10:
                verdict = verify_claim_nli(subclaim, p["passage"], nli_model)
                passage_verdicts.append(verdict)

            sc_verdict = get_subclaim_verdict(passage_verdicts)
            subclaim_verdicts.append(sc_verdict)
            print(f"  [RESULT] Sub-claim {sc_idx} verdict: {sc_verdict}")

            # ── Output: subclaim header ───────────────────────
            # Last subclaim gets [depends on: previous] tag
            if sc_idx == len(subclaims) and sc_idx > 1:
                depends = ", ".join(str(i) for i in range(1, sc_idx))
                sc_header = f"  Sub-claim {sc_idx}: {subclaim} [depends on: {depends}]"
            else:
                sc_header = f"  Sub-claim {sc_idx}: {subclaim}"
            out.write(f"\n{sc_header}\n")

            # ── Output: top 5 diverse passages (different source pages) ──
            # Pick best passages from unique sources to avoid repetition
            seen_sources = set()
            diverse5     = []
            for p in top10:
                src = p.get("source", "Wikipedia")
                if src not in seen_sources:
                    seen_sources.add(src)
                    diverse5.append(p)
                if len(diverse5) == 5:
                    break
            # Fallback: if not enough unique sources, fill from top10
            if len(diverse5) < 5:
                for p in top10:
                    if p not in diverse5:
                        diverse5.append(p)
                    if len(diverse5) == 5:
                        break

            for p in diverse5:
                global_idx = global_passage_counter[0]
                snippet    = p["passage"][:100].replace("\n", " ").strip() + "..."
                page_title = p.get("source", "Wikipedia")
                out.write(f'    [{global_idx}] ({page_title}) "{snippet}"\n')
                global_passage_counter[0] += 1

            out.write(f"    Verdict: {sc_verdict}\n")

        # ── Final claim verdict + reasoning chain ─────────────
        final_verdict = get_final_claim_verdict(subclaim_verdicts)
        # reasoning_chain_text already parsed above (after stage 1)

        print(f"\n{'=' * 60}")
        print(f"[FINAL VERDICT] {claim_id}: {final_verdict}")
        print(f"{'=' * 60}")

        ground_truth = claim_data.get("ground_truth", None)
        if ground_truth:
            out.write(f"\nFINAL VERDICT: {final_verdict}  (true: {ground_truth})\n")
        else:
            out.write(f"\nFINAL VERDICT: {final_verdict}\n")
        if reasoning_chain_text:
            out.write(f"Reasoning chain: {reasoning_chain_text}\n")
        out.write("=" * 100 + "\n")


def run_pipeline():
    """
    Main entry point.
    - Option 1: Enter claims one at a time in the terminal.
                Type 'exit' to stop entering claims and quit.
    - Option 2: Provide a .jsonl or .txt file; all claims are processed.
    - Option 3: Exit immediately.

    Models are loaded once and reused across all claims / sessions.
    """
    print("\n" + "=" * 60)
    print("   CLAIM VERIFICATION PIPELINE — STARTING")
    print("=" * 60)

    # Load models once
    print("\n[INFO] Loading models...")
    print("  [1/3] Loading BGE Retriever (BAAI/bge-small-en-v1.5)...")
    retriever_model = SentenceTransformer("BAAI/bge-small-en-v1.5")

    print("  [2/3] Loading BGE Reranker (BAAI/bge-reranker-base)...")
    rerank_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-base")
    rerank_model     = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-base")
    rerank_model.eval()

    print("  [3/3] Loading NLI Model (cross-encoder/nli-deberta-v3-small)...")
    nli_model = pipeline(
        "text-classification",
        model="cross-encoder/nli-deberta-v3-small",
        device=0 if torch.cuda.is_available() else -1
    )
    print("[INFO] All models loaded successfully.\n")

    # Open output file once — all results appended here
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:

        out.write("=" * 100 + "\n")
        out.write("   CLAIM VERIFICATION PIPELINE — FULL OUTPUT\n")
        out.write("=" * 100 + "\n\n")

        terminal_claim_counter = 0

        while True:

            show_menu()
            choice = input("\nEnter your choice (1 / 2 / 3): ").strip()

            # ── Option 1: Terminal loop ────────────────────────
            if choice == "1":
                print("\n[INFO] Type your claim and press Enter.")
                print("[INFO] Type  \'exit\'  to return to the main menu.\n")

                while True:
                    claim_text = input("Enter claim: ").strip()

                    if claim_text.lower() == "exit":
                        print("\n[INFO] Returning to main menu...\n")
                        break

                    if not claim_text:
                        print("[WARNING] Empty input — please type a claim.")
                        continue

                    terminal_claim_counter += 1
                    claims = [{
                        "claim_id":     f"claim_terminal_{terminal_claim_counter}",
                        "claim":        claim_text,
                        "ground_truth": None
                    }]

                    process_claims(
                        claims, out,
                        retriever_model, rerank_tokenizer,
                        rerank_model, nli_model
                    )
                    out.flush()   # write to disk immediately after each claim

                    print("\n[INFO] Claim processed. Enter another claim or type \'exit\'.")

            # ── Option 2: File input ───────────────────────────
            elif choice == "2":
                file_path = input("\nEnter file path (.jsonl or .txt): ").strip()
                claims    = load_claims_from_file(file_path)

                if claims:
                    process_claims(
                        claims, out,
                        retriever_model, rerank_tokenizer,
                        rerank_model, nli_model
                    )
                    out.flush()

            # ── Option 3: Exit ─────────────────────────────────
            elif choice == "3":
                print("\n[INFO] Exiting pipeline. Output saved to:", OUTPUT_FILE)
                print("=" * 60 + "\n")
                break

            else:
                print("[ERROR] Invalid choice. Please enter 1, 2, or 3.")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    run_pipeline()