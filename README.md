Step 1 (Decomposition):

Two decomposition pipelines were implemented:

1. Qwen3-32B
   - File: qwen_claim.py
   - Output: output1.txt

2. Qwen2.5-3B
   - File: qwen_newmain.py
   - Output: output_newqwen(1).txt

Step 2 (Retrieval):
  

- Claims and previously generated subclaims are extracted from output1.txt.
- Dense retrieval is performed using the BAAI/bge-small-en-v1.5 embedding model and FAISS similarity indexing.
- Relevant Wikipedia passages are retrieved for each subclaim.
- Similarity scores are generated and classified as Highly Relevant, Relevant, or Irrelevant.

File:
retrieval_claim.py

Output:
retrieval_output.txt