import os
import sys
sys.path.append(os.path.abspath(".."))
from MedRAG.src.medrag import MedRAG
import json
import argparse
from datasets import Dataset
import torch, gc
import time

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, help="Path to config JSON file")
parser.add_argument("--corpus_name", type=str, default="MedText")
parser.add_argument("--retriever_name", type=str, default="RRF-4")
parser.add_argument("--llm_name", type=str, default="meta-llama/Llama-3.2-3B-Instruct")
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
with open("./rawdata/benchmark.json", "r") as f:
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
medrag = MedRAG(
    llm_name=args.llm_name,  # or path to local model
    rag=args.rag,
    retriever_name=args.retriever_name,
    corpus_name=args.corpus_name,  # Must match folder in corpus/
    db_dir="../MedRAG/src/data/corpus",  # Parent directory containing MedCorp/
    corpus_cache=True,  # Optional: speed up repeated runs
    cache_dir="../MedRAG/src/llm/cache",  # Optional: cache directory for LLM
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


# Existing files
existing_files = set(os.listdir(save_dir))
print(f"Found {len(existing_files)} existing files in {save_dir}\n")

start_time = time.time()
last_time = start_time

# Generate answer using retrieval
for idx, qa in enumerate(qa_list):
    if idx % 5 == 0:
        print(f"Processing question {idx + 1}/{len(qa_list)} ...")

        this_time = time.time()
        loop_time = this_time - last_time
        last_time = this_time

        elapsed = this_time - start_time
        avg_time = elapsed / (idx + 1)
        remaining = (len(qa_list) - (idx + 1)) * avg_time
        
        print(f"Time elapsed: {int(loop_time // 60)}m {int(loop_time % 60)}s, ETA: {int(remaining // 60)}m {int(remaining % 60)}s remaining")

    if qa["id"] + ".json" in existing_files:
        print(f"Skipping {qa['id']}, already exists")
        continue

    question = qa["question"]
    options = qa["options"]
    answer, snippets, scores = medrag.answer(
        question=question, options=options, k=args.k
    )
    qa["predict"] = answer
    qa["snippets"] = snippets
    qa["scores"] = scores
    qa["execution_time"] = time.time() - this_time

    with open(os.path.join(save_dir, qa["id"] + ".json"), "w") as f:
        json.dump(qa, f, indent=4)

    gc.collect()
    torch.cuda.empty_cache()
