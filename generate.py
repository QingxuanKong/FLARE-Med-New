import json
from MedRAG.src import MedRAG
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--corpus_name", type=str, default="medcorp")
parser.add_argument("--retriever_name", type=str, default="RRF-4")
parser.add_argument("--llm_name", type=str, default="OpenAI/gpt-35-turbo-16k")
parser.add_argument("--rag", action=argparse.BooleanOptionalAction)
parser.add_argument("--k", type=int, default=32)

parser.add_argument("--results_dir", type=str, default="./prediction")
parser.add_argument("--dataset_name", type=str, default="bioasq")


args = parser.parse_args()

# Read questions and options from json file
with open("rawdata/benchmark.json", "r") as f:
    data = json.load(f)

task_key = "bioasq"
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

print(qa_list[0])


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

# Generate answer using retrieval
for qa in qa_list:
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


# if os.path.exists(save_dir):
#     if "pmc_llama" in args.llm_name.lower():
#         acc, std, flag = evaluate(
#             datasets[dataset_name], save_dir, split, locate_answer4pub_llama
#         )
#     else:
#         acc, std, flag = evaluate(datasets[dataset_name], save_dir, split)
#     scores.append(acc)
#     print("mean acc: {:.4f}; proportion std: {:.4f}".format(acc, std), end="")
#     if flag:
#         print(" (NOT COMPLETED)")
#     else:
#         print("")
# else:
#     print("NOT STARTED.")
#     # scores.append(0)
