from src.medrag import MedRAG
import os
import torch, gc

# Example question (from MMLU-style prompt)
question = "A lesion causing compression of the facial nerve at the stylomastoid foramen will cause ipsilateral"
options = {
    "A": "paralysis of the facial muscles.",
    "B": "paralysis of the facial muscles and loss of taste.",
    "C": "paralysis of the facial muscles, loss of taste and lacrimation.",
    "D": "paralysis of the facial muscles, loss of taste, lacrimation and decreased salivation.",
}

# Initialize MedRAG with LLaMA + BM25 + MedCorp corpus
current_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(current_dir, "src/data/corpus")

medrag = MedRAG(
    # llm_name="axiong/PMC_LLaMA_13B",  # or path to local model
    llm_name="meta-llama/Llama-3.2-3B-Instruct",  # or path to local model
    rag=True,
    retriever_name="RRF-4",
    corpus_name="StatPearls",  # Must match folder in corpus/
    db_dir=db_dir,  # Parent directory containing MedCorp/
    corpus_cache=False,  # Optional: speed up repeated runs
    cache_dir="src/llm/cache",  # Optional: cache directory for LLM
)

# Generate answer using retrieval
answer, snippets, scores = medrag.answer(question=question, options=options, k=32)

# Output results
print("===== Final Answer =====")
print(answer)

print("\n===== Top Retrieved Snippets =====")
for i, snippet in enumerate(snippets[:3]):
    print(f"[{i+1}] {snippet['title']}:\n{snippet['content'][:300]}...\n")

print("\n===== Retrieval Scores =====")
print(scores[:3])


# # ----------------------------------------------------
# # --------------test the retrieval--------------------
# # ----------------------------------------------------

# from src.utils import RetrievalSystem

# # Set the corpus name and path you used when chunking
# retriever = RetrievalSystem(
#     retriever_name="MedCPT",  # or "MedCPT"
#     corpus_name="MedText",  # make sure this matches your folder name
#     db_dir="src/corpus",  # or absolute path to your corpus folder
#     cache=True,
# )

# query = "facial nerve lesion symptoms"
# snippets, scores = retriever.retrieve(query, k=5)

# for i, snippet in enumerate(snippets):
#     print(f"[{i}] {snippet['title']}: {snippet['content'][:200]}...")
# print("\nScores:", scores)


# ----------------------------------------------------
# --------------test the llama flare agent------------
# ----------------------------------------------------

""" from src.utils import RetrievalSystem
from src.isla_llama_flare_agent import LlamaFlareAgent

# Set the corpus name and path you used when chunking
retriever = RetrievalSystem(
    retriever_name="RRF-4",  # or "MedCPT"
    corpus_name="Textbooks",  # make sure this matches your folder name
    db_dir="./src/data/corpus",  # or absolute path to your corpus folder
    cache=True,
) """

""" agent = LlamaFlareAgent(
    model_name_or_path="meta-llama/Llama-3.2-3B-Instruct",  # or path to local model
    retriever=retriever,
    # --- FLARE Control Parameters ---
    uncertainty_threshold=0.6,
    lookahead_tokens=100,
    max_flare_rounds=3,
    # --- Retriever Parameters ---
    dynamic_k=3,
    # --- LLM Parameters ---
    temperature=0.1,
    top_p=1.0,
    max_total_tokens=1024,
    debug=True,
) """

""" question = "What are the main treatments for type 2 diabetes?"
base_prompt = "You are a medical assistant. Answer the following question accurately based on the provided context, thinking step-by-step."

final_answer, history = agent.generate_answer(
    question=question,
    base_prompt=base_prompt,
    # initial_context_snippets=initial_snippets # Optional
)

print("\n--- Final Answer ---")
print(final_answer) """

