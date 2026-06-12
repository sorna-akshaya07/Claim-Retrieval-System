Stage Details 

Stage 1 — Dense Retrieval (stage1_dense_retriever.py)

Model: BAAI/bge-small-en-v1.5
Fetches Wikipedia pages using wikipediaapi
Splits pages into passages
Encodes passages using FAISS index
Retrieves Top 50 unique passages per subclaim
Output: stage1_output.txt

Stage 2 — Reranking (stage2_reranker.py)

Model: BAAI/bge-reranker-base
Reranks Top 20 passages per subclaim
Labels passages as Highly Relevant, Relevant, or Irrelevant
Output: stage2_output.txt

Stage 3 — Hard Negative + Adversarial Elimination (crosscheck_hardadver.py)

Models:

BAAI/bge-reranker-base — reranking
cross-encoder/nli-deberta-v3-small — NLI filtering


Removes contradiction passages
Removes low-score passages (score < 1)
Guarantees minimum 3 passages per subclaim using 4-level fallback
Output: hard_adver_output.txt

Stage 4 — Claim Verification (stage4_verification.py)

Model: cross-encoder/nli-deberta-v3-small
Runs NLI on each passage against the subclaim
Labels each passage as SUPPORTED, REFUTED, or NEI
Subclaim verdict using majority vote:

support > refute and support >= 1
    → SUPPORTED

refute > support and refute >= 1
    → REFUTED

support = refute
    → NOT ENOUGH INFO

no evidence
    → NOT ENOUGH INFO


Final claim verdict:
Any subclaim REFUTED
    → FINAL CLAIM = REFUTED

All subclaims SUPPORTED
    → FINAL CLAIM = SUPPORTED

No subclaims available
    → FINAL CLAIM = NOT ENOUGH INFO

Otherwise
    → FINAL CLAIM = NOT ENOUGH INFO
    
Output: verification_output.txt

## Project Structure

| File | Description |
|------|-------------|
| `stage1_dense_retriever.py` | Stage 1 — Dense Retrieval |
| `stage2_reranker.py` | Stage 2 — Reranking |
| `crosscheck_hardadver.py` | Stage 3 — Hard Negative Elimination |
| `stage4_verification.py` | Stage 4 — NLI Verification |
| `output1.txt` | Claim decomposition input |
| `stage1_output.txt` | Stage 1 output |
| `stage2_output.txt` | Stage 2 output |
| `hard_adver_output.txt` | Stage 3 output |
| `verification_output.txt` | Final verification output |
| `requirements.txt` | Dependencies |
| `Human_Evaluation.xlsx` | Human Evaluation |

## Models Used

| Stage | Model | Purpose |
|-------|-------|---------|
| Stage 1 | `BAAI/bge-small-en-v1.5` | Dense passage retrieval |
| Stage 2 | `BAAI/bge-reranker-base` | Passage reranking |
| Stage 3 | `BAAI/bge-reranker-base` + `cross-encoder/nli-deberta-v3-small` | Hard negative filtering |
| Stage 4 | `cross-encoder/nli-deberta-v3-small` | NLI verification |

## Human Evaluation

Human Evaluation is a manual assessment process used to measure the quality and correctness of model predictions based on predefined criteria.
It helps in understanding how well the system performs beyond automated metrics by evaluating reasoning, evidence, and final outputs.

| Column Name           | Description                                                          |
| --------------------- | -------------------------------------------------------------------- |
| claim_id              | Unique identifier assigned to each claim for tracking purposes       |
| claim_sort            | Original or shortened version of the claim used for evaluation       |
| y_true                | Ground truth label (SUPPORT / REFUTE / NEI)                          |
| y_pred                | Model’s predicted label after reasoning                              |
| n_sub_claims          | Number of sub-claims generated during decomposition                  |
| citation_correctness  | Measures relevance and correctness of retrieved evidence (1–5 scale) |
| reasoning_quality     | Evaluates logical consistency and reasoning strength (1–5 scale)     |
| decomposition_quality | Checks how well the claim is broken into sub-claims (1–5 scale)      |
| overall_1_to_5        | Final combined score based on all evaluation metrics                 |
| notes                 | Short justification explaining why the score was assigned            |


## Commit History

> Note: Some files were committed together in early stages.
> Each stage is independently implemented and documented.

# Claim Verification Pipeline

An automated fact-checking system that verifies claims against Wikipedia evidence using a 4-stage NLP pipeline.

---

## What This System Does

Given a claim (a statement), the pipeline automatically determines whether it is:
- **SUPPORTED** — Wikipedia evidence confirms it
- **REFUTED** — Wikipedia evidence contradicts it
- **NOT ENOUGH INFO** — Insufficient evidence found

---

## Prerequisites

### 1. Python Version
Python 3.8 or higher is required.

### 2. Groq API Key
This pipeline uses the **Qwen model via Groq API** for sub-claim generation and Wikipedia title extraction.

**Steps to get a Groq API key:**
1. Go to [https://console.groq.com]
2. Sign up / Log in
3. Navigate to **API Keys** → Create a new key
4. Copy the key

**Set up the key in a `.env` file** in the same folder as the script:
```
GROQ_API_KEY=your_api_key_here
```

### 3. Install Required Libraries

Run the following command in your terminal:

```bash
pip install groq python-dotenv sentence-transformers transformers faiss-cpu torch wikipedia wikipedia-api numpy
```

> **Note:** If you have a GPU, install `faiss-gpu` instead of `faiss-cpu` for faster retrieval.

---

## Models Used (Auto-Downloaded on First Run)

| Model | Purpose | Downloaded From |
|---|---|---|
| `qwen/qwen3-32b` | Sub-claim generation + Wikipedia title extraction | Groq API (no download needed) |
| `BAAI/bge-small-en-v1.5` | Dense retrieval (Stage 2) | Hugging Face |
| `BAAI/bge-reranker-base` | Reranking + Hard negative elimination (Stage 3) | Hugging Face |
| `cross-encoder/nli-deberta-v3-small` | Adversarial elimination + Verification (Stage 3 & 4) | Hugging Face |

> All Hugging Face models are downloaded automatically on first run and cached locally.

---

## How to Run

```bash
python claim_verification_pipeline.py
```

---

## User Input Options

When you run the script, you will see this menu in the terminal:

```
============================================================
   CLAIM VERIFICATION PIPELINE
============================================================
  1 -> Enter a claim directly in the terminal
  2 -> Provide a file path  (.jsonl or .txt)
  3 -> Exit
------------------------------------------------------------
Enter your choice (1 / 2 / 3):
```

### Option 1 — Terminal Input

Type your claim directly and press Enter. The pipeline processes it immediately.

```
Enter claim: Carnegie Hall Tower is located in the same city as Staten Island.
```

After processing, you can enter another claim or type `exit` to return to the menu.

```
[INFO] Claim processed. Enter another claim or type 'exit'.
Enter claim: exit
```

### Option 2 — File Input

Provide a path to a `.jsonl` or `.txt` file containing multiple claims.

```
Enter file path (.jsonl or .txt): C:\Users\name\Desktop\test.jsonl
```

**Supported file formats:**

**.jsonl format** (recommended):
```json
{"id": 0, "claim": "Fergie recorded M.I.L.F.$", "label": 0}
{"id": 1, "claim": "Stephen King wrote The Shining.", "label": 0}
```

**.txt format — plain text (one claim per line):**
```
Fergie recorded M.I.L.F.$
Stephen King wrote The Shining.
```

**Label mapping** (for ground truth comparison):
```
0 → SUPPORTED
1 → REFUTED
2 → NOT ENOUGH INFO
```

### Option 3 — Exit

Exits the program. All results are saved to `pipeline_output.txt`.

---

## Pipeline Stages

### Stage 1 — Sub-claim Generation
The claim is sent to the **Qwen model via Groq API**. It breaks the claim into smaller, independently verifiable sub-claims.

**Terminal output:**
```
[STAGE 1] Generating sub-claims...
  [INFO] 3 sub-claim(s) generated.
```

### Wikipedia Fetch
The Qwen model is used again to extract relevant Wikipedia page titles. Those pages are loaded directly to collect evidence passages.

**Terminal output:**
```
[INFO] Fetching Wikipedia passages dynamically...
  [INFO] Wikipedia titles from Qwen: ['Carnegie Hall Tower', 'Staten Island', 'New York City']
  [INFO] Loading Wikipedia page: Carnegie Hall Tower
  [INFO] Total unique passages loaded: 214
```

### Stage 2 — Dense Retrieval (Top 50)
Using the **BGE retriever model** and **FAISS vector index**, the top 50 most relevant passages are retrieved for each sub-claim.

**Terminal output:**
```
  [STAGE 2] Dense retrieval — top 50...
  [INFO] 50 passages retrieved.
```

### Stage 3A — Reranking (Top 50 → Top 20)
The **BGE reranker model** scores each passage more accurately and selects the top 20.

**Terminal output:**
```
  [STAGE 3A] Reranking top 50 -> top 20...
  [INFO] 20 passages after reranking.
```

### Stage 3B — Elimination (Top 20 → Top 10)
Two types of passages are removed:

- **Hard Negatives** — Passages with raw rerank score < 1 (not relevant enough)
- **Adversarial Distractors** — Passages where NLI returns CONTRADICTION (misleading)

**Terminal output:**
```
  [STAGE 3B] Eliminating hard negatives and adversarial distractors...
  [REMOVED] Hard negative  (score=-0.71)
  [REMOVED] Adversarial distractor (NLI=CONTRADICTION)
  [INFO] Strict filter applied — 8 passages selected.
```

**Fallback messages:**
```
[INFO] Scored + fallback merge — 3 passages selected.   ← Some relevant passages found
[WARNING] Emergency top-3 selected                      ← Very few relevant passages found
```

### Stage 4 — Verification
The **NLI DeBERTa model** checks each of the top 10 passages against the sub-claim. Majority vote determines the sub-claim verdict.

**Terminal output:**
```
  [STAGE 4] Verifying sub-claim...
  [RESULT] Sub-claim 1 verdict: SUPPORTED
```

**Final verdict logic:**
```
Any sub-claim REFUTED    → FINAL: REFUTED
All sub-claims SUPPORTED → FINAL: SUPPORTED
Otherwise                → FINAL: NOT ENOUGH INFO
```

---

## Output File

All results are saved to `pipeline_output.txt` in the same folder as the script.

## Common Terminal Messages Explained

| Message | Meaning |
|---|---|
| `[INFO] All models loaded successfully` | All 3 models ready |
| `[INFO] Wikipedia titles from Qwen` | Qwen extracted page titles |
| `[REMOVED] Hard negative` | Low-relevance passage removed |
| `[REMOVED] Adversarial distractor` | Misleading passage removed |
| `[INFO] Strict filter applied` | Enough good passages found |
| `[WARNING] Emergency top-3 selected` | Very few relevant passages — Wikipedia may not have enough info |
| `[RESULT] Sub-claim verdict: SUPPORTED` | Sub-claim verified |
| `[FINAL VERDICT]` | Overall claim verdict |

---

## Troubleshooting

**`GROQ_API_KEY not found`**
→ Make sure `.env` file exists in the same folder with `GROQ_API_KEY=your_key`

**`[WARNING] No Wikipedia passages fetched`**
→ Wikipedia could not find relevant pages for the claim. Try a more specific claim.

**`[ERROR] File not found`**
→ Check the file path. Do not include quotes around the path when entering it.

**Models downloading slowly**
→ First run downloads ~500MB of models. Subsequent runs use cached versions.