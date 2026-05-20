Step 1 (Decomposition):

Two decomposition pipelines were implemented:

1. Qwen3-32B
   - File: qwen_claim.py
   - Output: output1.txt

2. Qwen2.5-3B
   - File: qwen_newmain.py
   - Output: output_newqwen(1).txt

Step 2 (Retrieval):

## Stage 1 - Dense Retrieval

- Wikipedia passage collection
- Manual entity title retrieval
- BGE embedding generation
- FAISS dense retrieval
- Top-50 passage retrieval

Model Used:
- BAAI/bge-small-en-v1.5
File Name:
- stage1_dense_retrieval.py
Output:
- stage1_output.txt

---

## Stage 2 - Reranking

- Cross-encoder reranking
- Passage relevance scoring
- Label generation

Labels:
- Highly Relevant
- Relevant
- Irrelevant

Model Used:
- BAAI/bge-reranker-base
File name:
- stage2_reranker.py
Output:
- stage2_output.txt


## Files

- stage1_dense_retrieval.py
- stage2_reranker.py
- stage1_output.txt
- stage2_output.txt
