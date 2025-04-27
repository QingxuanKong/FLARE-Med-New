import os
import json
import argparse
import re
from utils import QADataset, locate_answer, locate_answer4pub_llama
from sklearn.metrics import accuracy_score
import numpy as np
import statistics

def evaluate(dataset, save_dir, split="test", locate_fun=locate_answer):

    flag = False
    pred = []
    empty_count = 0
    na_count = 0
    answer_list = ["A", "B", "C", "D"]
    answer2idx = {ans:i for i, ans in enumerate(answer_list)}
    
    total_len = len(dataset)

    # for i, fpath in enumerate(sorted([f for f in os.listdir(save_dir) if f.endswith(".json")])[:total_len]):
    for q_idx in range(len(dataset)):
        # fpath = os.path.join(save_dir, split + "_" + dataset.index[q_idx] + ".json")
        fpath = os.path.join(save_dir, dataset.index[q_idx] + ".json")
        with open(fpath, "r") as f:
            medrag_data = json.load(f)

        predict_str = medrag_data.get("predict", "")
        answer_match = re.search(r'"(?:answer_choice|answer)":\s*"([A-Z])"', predict_str)
        ans = answer_match.group(1) if answer_match else None
        if ans in answer_list:
            pred.append(answer_list.index(ans))
        else:
            pred.append(-1)
    
    truth = [answer2idx[item['answer']] for item in dataset]
    if len(pred) < len(truth):
        truth = truth[:len(pred)]
        flag = True
        
    acc = (np.array(truth) == np.array(pred)).mean()
    std = np.sqrt(acc * (1-acc) / len(truth))
    return acc, std, flag

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to config JSON file")
    parser.add_argument("--corpus_name", type=str, default="MedText")
    parser.add_argument("--retriever_name", type=str, default="RRF-4")
    parser.add_argument("--llm_name", type=str, default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--rag", action=argparse.BooleanOptionalAction)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--results_dir", type=str, default="./prediction")
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--enable_flare", action=argparse.BooleanOptionalAction)
    parser.add_argument("--look_ahead_steps", type=int, default=50)
    parser.add_argument("--look_ahead_truncate_at_boundary", type=str, default=".,?,!")
    parser.add_argument("--max_query_length", type=int, default=300)
    parser.add_argument("--follow_up", action=argparse.BooleanOptionalAction)
    parser.add_argument("--n_rounds", type=int, default=3)
    parser.add_argument("--n_queries", type=int, default=2)
    parser.add_argument("--threads", type=int, default=8)

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
    print()

    # Set params values
    corpus_name = args.corpus_name
    retriever_name = args.retriever_name
    llm_name = args.llm_name
    rag = args.rag
    k = args.k
    results_dir = args.results_dir
    enable_flare = args.enable_flare
    follow_up = args.follow_up
    
    # Config dataset and benchmark
    if args.dataset_name is not None:
        dataset_names = [args.dataset_name]
    else:
        dataset_names = ['mmlu', 'medqa', 'medmcqa', 'pubmedqa', 'bioasq']

    datasets = {key:QADataset(key, dir="./rawdata") for key in dataset_names}

    # Evaluate results
    scores = []
    for dataset_name in dataset_names:
        print("[{:s}] ".format(dataset_name), end="")
        
        split = "test"
        if dataset_name == "medmcqa":
            split = "dev"
        if rag:
            # Use the exact same directory structure as in generate.py
            flare_part = "_flare" if enable_flare else ""
            followup_part = "_followup" if follow_up else ""
            save_dir = os.path.join(
                results_dir,
                dataset_name,
                f"rag{flare_part}{followup_part}_{k}",
                llm_name,
                corpus_name,
                retriever_name,
            )
        else:
            save_dir = os.path.join(results_dir, dataset_name, "cot", llm_name)

        if os.path.exists(save_dir):
            if "pmc_llama" in llm_name.lower():
                acc, std, flag = evaluate(datasets[dataset_name], save_dir, split, locate_answer4pub_llama)
            else:
                acc, std, flag = evaluate(datasets[dataset_name], save_dir, split)
            scores.append(acc)
            print("mean acc: {:.4f}; proportion std: {:.4f}".format(acc, std), end="")
            if flag:
                print(" (NOT COMPLETED)")
            else:
                print("")
        else:
            print("NOT STARTED.")
            # scores.append(0)

    if len(scores) > 0:
        print("[Average] mean acc: {:.4f}".format(sum(scores) / len(scores)))
