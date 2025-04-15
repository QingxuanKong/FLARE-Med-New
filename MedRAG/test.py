# from medrag import MedRAG

# # Example question (from MMLU-style prompt)
# question = "A lesion causing compression of the facial nerve at the stylomastoid foramen will cause ipsilateral"
# options = {
#     "A": "paralysis of the facial muscles.",
#     "B": "paralysis of the facial muscles and loss of taste.",
#     "C": "paralysis of the facial muscles, loss of taste and lacrimation.",
#     "D": "paralysis of the facial muscles, loss of taste, lacrimation and decreased salivation."
# }

# # Initialize MedRAG with LLaMA + BM25 + MedCorp corpus
# medrag = MedRAG(
#     llm_name="openai/gpt-3.5-turbo",  # or path to local model
#     rag=False,
#     retriever_name="BM25",
#     corpus_name="MedCorp",       # Must match folder in corpus/
#     db_dir="./corpus",             # Parent directory containing MedCorp/
#     corpus_cache=True            # Optional: speed up repeated runs
# )

# # Generate answer using retrieval
# answer, snippets, scores = medrag.answer(
#     question=question,
#     options=options,
#     k=32
# )

# # Output results
# print("===== Final Answer =====")
# print(answer)

# print("\n===== Top Retrieved Snippets =====")
# for i, snippet in enumerate(snippets[:3]):
#     print(f"[{i+1}] {snippet['title']}:\n{snippet['content'][:300]}...\n")

# print("\n===== Retrieval Scores =====")
# print(scores[:3])


# ----------------------------------------------------
# --------------test the retrieval--------------------
# ----------------------------------------------------

from src.utils import RetrievalSystem

# Set the corpus name and path you used when chunking
retriever = RetrievalSystem(
    retriever_name="RRF-2",               # or "MedCPT"
    corpus_name="StatPearls",              # make sure this matches your folder name
    db_dir="src/data/corpus",                  # or absolute path to your corpus folder
    cache=True
)

query = "facial nerve lesion symptoms"
snippets, scores = retriever.retrieve(query, k=5)

for i, snippet in enumerate(snippets):
    print(f"[{i}] {snippet['title']}: {snippet['content'][:200]}...")
print("\nScores:", scores)
