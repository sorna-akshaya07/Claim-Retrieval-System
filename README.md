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
