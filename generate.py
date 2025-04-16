import json
from MedRAG.src.medrag import MedRAG
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, help="Path to config JSON file")
parser.add_argument("--corpus_name", type=str, default="MedText")
parser.add_argument("--retriever_name", type=str, default="BM25")
parser.add_argument("--llm_name", type=str, default="OpenAI/gpt-3.5-turbo-16k")
parser.add_argument("--rag", action=argparse.BooleanOptionalAction)
parser.add_argument("--k", type=int, default=32)
parser.add_argument("--results_dir", type=str, default="./prediction")
parser.add_argument("--dataset_name", type=str, default="bioasq")

args = parser.parse_args()

# Load from specified config file if provided
if args.config:
    with open(args.config) as f:
        file_config = json.load(f)
        print(f"Loaded configuration from {args.config}\n")

        # Update args with values from specified config file
        # but only for values not explicitly set in command line
        for key, value in file_config.items():
            if key in vars(args) and parser.get_default(key) == getattr(args, key):
                setattr(args, key, value)

# Print final configuration
print("Final arguments:")
for key in [
    "corpus_name",
    "retriever_name",
    "llm_name",
    "rag",
    "k",
    "results_dir",
    "dataset_name",
]:
    print(f"{key}: {getattr(args, key)}")


# Read questions and options from json file
with open("MIRAGE/rawdata/benchmark.json", "r") as f:
    data = json.load(f)

task_key = args.dataset_name
dataset = data[task_key]

qa_list = []
for qid, qdata in dataset.items():
    qa_list.append(
        {
            "id": qid,
            "question": qdata["question"],
            "options": qdata["options"],
            "answer": qdata["answer"],
        }
    )
print(f"Total {len(qa_list)} questions in {task_key} dataset\n")

# Initialize MedRAG
current_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(current_dir, "MedRAG/src/data/corpus")

medrag = MedRAG(
    llm_name=args.llm_name,  # or path to local model
    rag=args.rag,
    retriever_name=args.retriever_name,
    corpus_name=args.corpus_name,  # Must match folder in corpus/
    db_dir=db_dir,  # Parent directory containing MedCorp/
    corpus_cache=True,  # Optional: speed up repeated runs
    cache_dir="MedRAG/src/llm/cache",  # Optional: cache directory for LLM
)

# Output results
if args.rag:
    save_dir = os.path.join(
        args.results_dir,
        args.dataset_name,
        "rag_" + str(args.k),
        args.llm_name,
        args.corpus_name,
        args.retriever_name,
    )
else:
    save_dir = os.path.join(args.results_dir, args.dataset_name, "cot", args.llm_name)
os.makedirs(save_dir, exist_ok=True)
print(f"Results will be saved to {save_dir}\n")

# Generate answer using retrieval
for idx, qa in enumerate(qa_list):
    if idx % 10 == 0:
        print(f"Processing question {idx + 1}/{len(qa_list)} ...")

    question = qa["question"]
    options = qa["options"]
    answer, snippets, scores = medrag.answer(
        question=question, options=options, k=args.k
    )
    qa["predict"] = answer
    qa["snippets"] = snippets
    qa["scores"] = scores

    filename = os.path.join(save_dir, qa["id"] + ".json")

    with open(filename, "w") as f:
        json.dump(qa, f, indent=4)
