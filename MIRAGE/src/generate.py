import os
import sys
sys.path.append(os.path.abspath(".."))
from MedRAG.src.medrag import MedRAG
import json
import argparse
from datasets import Dataset
import torch, gc
import time
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import threading
from tqdm import tqdm

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, help="Path to config JSON file")
parser.add_argument("--corpus_name", type=str, default="MedText")
parser.add_argument("--retriever_name", type=str, default="RRF-4")
parser.add_argument("--llm_name", type=str, default="OpenAI/gpt-4.1-nano")
parser.add_argument("--rag", action=argparse.BooleanOptionalAction)
parser.add_argument("--k", type=int, default=32)
parser.add_argument("--results_dir", type=str, default="./prediction")
parser.add_argument("--dataset_name", type=str, default="bioasq")
# FLARE parameters
parser.add_argument("--enable_flare", action=argparse.BooleanOptionalAction)
parser.add_argument("--look_ahead_steps", type=int, default=50)
parser.add_argument("--look_ahead_truncate_at_boundary", type=str, default=".,?,!")
parser.add_argument("--max_query_length", type=int, default=300)
# Follow-up parameters
parser.add_argument("--follow_up", action=argparse.BooleanOptionalAction)
parser.add_argument("--n_rounds", type=int, default=3)
parser.add_argument("--n_queries", type=int, default=2)
# Threading parameters
parser.add_argument("--threads", type=int, default=4, help="Number of concurrent threads to use")

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
    "enable_flare",
    "look_ahead_steps",
    "look_ahead_truncate_at_boundary",
    "max_query_length",
    "follow_up",
    "n_rounds",
    "n_queries",
    "threads",
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


# Output results
if args.rag:
    flare_part = "_flare" if args.enable_flare else ""
    followup_part = "_followup" if args.follow_up else ""
    save_dir = os.path.join(
        args.results_dir,
        args.dataset_name,
        f"rag{flare_part}{followup_part}_{args.k}",
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

# Create ONE MedRAG instance to be shared across threads
print("Initializing shared MedRAG instance...")
shared_medrag_instance = MedRAG(
    llm_name=args.llm_name,
    rag=args.rag,
    retriever_name=args.retriever_name,
    corpus_name=args.corpus_name,
    db_dir="../MedRAG/src/data/corpus",
    corpus_cache=True,
    cache_dir="../MedRAG/src/llm/cache",
    enable_flare=args.enable_flare,
    look_ahead_steps=args.look_ahead_steps,
    look_ahead_truncate_at_boundary=list(args.look_ahead_truncate_at_boundary),
    max_query_length=args.max_query_length,
    follow_up=args.follow_up,
)
print("Shared MedRAG instance initialized.")


# For thread safety when writing to console
print_lock = threading.Lock()

# Modify process_question to accept the shared medrag instance
def process_question(qa, medrag_instance):
    # Skip if already processed
    if qa["id"] + ".json" in existing_files:
        with print_lock:
            print(f"Skipping {qa['id']}, already exists")
        return None
    
    start_time = time.time()
    question = qa["question"]
    options = qa["options"]
    
    try:
        # Use the passed MedRAG instance directly
        medrag = medrag_instance
        
        # Handle different answer methods based on follow-up setting
        if args.follow_up:
            answer, messages = medrag.answer(
                question=question, 
                options=options, 
                k=args.k,
                n_rounds=args.n_rounds,
                n_queries=args.n_queries
            )
            # Extract snippets from messages
            all_snippets = []
            for msg in messages:
                if isinstance(msg, dict) and msg.get("role") == "system" and msg.get("snippets"):
                    all_snippets.extend(msg.get("snippets", []))
                elif hasattr(msg, "role") and msg.role == "system" and hasattr(msg, "snippets"):
                    all_snippets.extend(getattr(msg, "snippets", []))
            
            snippets = all_snippets
            scores = None  # Scores might not be available in this format
            qa["messages"] = [m if isinstance(m, dict) else m.model_dump() for m in messages]
        else:
            answer, snippets, scores = medrag.answer(
                question=question, 
                options=options, 
                k=args.k
            )
        
        qa["predict"] = answer
        qa["snippets"] = snippets
        if scores is not None:
            qa["scores"] = scores
        qa["execution_time"] = time.time() - start_time
        
        # Save the result
        with open(os.path.join(save_dir, qa["id"] + ".json"), "w") as f:
            json.dump(qa, f, indent=4)
        
        # Clean up
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except:
            pass  # In case CUDA is not available
            
        return qa["id"]
    except Exception as e:
        with print_lock:
            print(f"Error processing question {qa['id']}: {str(e)}")
        return None

# Calculate how many questions need to be processed
questions_to_process = [qa for qa in qa_list if qa["id"] + ".json" not in existing_files]
print(f"Processing {len(questions_to_process)} out of {len(qa_list)} questions with {args.threads} threads\n")

# Process questions using thread pool
start_time = time.time()
with ThreadPoolExecutor(max_workers=args.threads) as executor:
    # Pass the shared medrag instance to each task
    futures = {executor.submit(process_question, qa, shared_medrag_instance): qa["id"] for qa in questions_to_process}
    
    # Setup progress bar
    pbar = tqdm(total=len(questions_to_process), desc="Processing questions")
    
    for future in concurrent.futures.as_completed(futures):
        result = future.result()
        if result:
            pbar.update(1)
    
    pbar.close()

# Final summary
total_time = time.time() - start_time
print(f"\nBenchmark completed in {int(total_time // 60)}m {int(total_time % 60)}s")
print(f"Processed {len(questions_to_process)} questions using {args.threads} threads")
print(f"Average time per question: {total_time / max(1, len(questions_to_process)):.2f}s")
print(f"Results saved to {save_dir}")
