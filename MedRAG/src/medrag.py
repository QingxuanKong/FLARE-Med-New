import os
import re
import json
import tqdm
import torch
import time
import argparse
import transformers
from transformers import AutoTokenizer
import openai
from openai import RateLimitError
from transformers import StoppingCriteria, StoppingCriteriaList
import tiktoken
import sys
import gc
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

sys.path.append("src")
from .utils import RetrievalSystem, DocExtracter
from .template import *

from .config import config

openai.api_type = (
    openai.api_type or os.getenv("OPENAI_API_TYPE") or config.get("api_type")
)
openai.api_version = (
    openai.api_version or os.getenv("OPENAI_API_VERSION") or config.get("api_version")
)
openai.api_key = openai.api_key or os.getenv("OPENAI_API_KEY") or config["api_key"]

if openai.__version__.startswith("0"):
    openai.api_base = (
        openai.api_base or os.getenv("OPENAI_API_BASE") or config.get("api_base")
    )
    if openai.api_type == "azure":
        openai_client = lambda **x: openai.ChatCompletion.create(
            **{"engine" if k == "model" else k: v for k, v in x.items()}
        )["choices"][0]["message"]["content"]
    else:
        openai_client = lambda **x: openai.ChatCompletion.create(**x)["choices"][0][
            "message"
        ]["content"]
else:
    if openai.api_type == "azure":
        openai.azure_endpoint = (
            openai.azure_endpoint
            or os.getenv("OPENAI_ENDPOINT")
            or config.get("azure_endpoint")
        )
        openai_client = (
            lambda **x: openai.AzureOpenAI(
                api_version=openai.api_version,
                azure_endpoint=openai.azure_endpoint,
                api_key=openai.api_key,
            )
            .chat.completions.create(**x)
            .choices[0]
            .message.content
        )
    else:
        openai_client = (
            lambda **x: openai.OpenAI(
                api_key=openai.api_key,
            )
            .chat.completions.create(**x)
            .choices[0]
            .message.content
        )


class MedRAG:

    def __init__(
        self,
        llm_name="OpenAI/gpt-4.1-nano",
        rag=True,
        follow_up=False,
        retriever_name="MedCPT",
        corpus_name="MedText",
        db_dir="./corpus",
        cache_dir=None,
        corpus_cache=False,
        HNSW=False,
        enable_flare=False,
        look_ahead_steps=0,
        look_ahead_boundary=None,
        look_ahead_truncate_at_boundary=None,
        look_ahead_filter_prob=0,
        look_ahead_mask_prob=0,
        look_ahead_mask_method="simple",
        look_ahead_pre_retrieval="",
        only_use_look_ahead=False,
        use_full_input_as_query=False,
        max_query_length=None,
        final_stop_sym="\n\n",
        verbose=False,
    ):
        self.llm_name = llm_name
        self.rag = rag
        self.retriever_name = retriever_name
        self.corpus_name = corpus_name
        self.db_dir = db_dir
        self.cache_dir = cache_dir
        self.docExt = None
        self.verbose = verbose
        
        if self.verbose:
            print(f"[VERBOSE] Initializing MedRAG with {llm_name}")
            
        if rag:
            if self.verbose:
                print(f"[VERBOSE] Setting up retrieval system: {retriever_name}, {corpus_name}")
            self.retrieval_system = RetrievalSystem(
                self.retriever_name,
                self.corpus_name,
                self.db_dir,
                cache=corpus_cache,
                HNSW=HNSW,
            )
        else:
            self.retrieval_system = None
        self.templates = {
            "cot_system": general_cot_system,
            "cot_prompt": general_cot,
            "medrag_system": general_medrag_system,
            "medrag_prompt": general_medrag,
        }
        if self.llm_name.split("/")[0].lower() == "openai":
            self.model = self.llm_name.split("/")[-1]
            if "gpt-3.5" in self.model or "gpt-35" in self.model:
                self.max_length = 16384
                self.context_length = 15000
            elif "gpt-4" in self.model:
                self.max_length = 32768
                self.context_length = 30000
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        elif "gemini" in self.llm_name.lower():
            import google.generativeai as genai

            genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
            self.model = genai.GenerativeModel(
                model_name=self.llm_name.split("/")[-1],
                generation_config={
                    "temperature": 0,
                    "max_output_tokens": 2048,
                },
            )
            if "1.5" in self.llm_name.lower():
                self.max_length = 1048576
                self.context_length = 1040384
            else:
                self.max_length = 30720
                self.context_length = 28672
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        else:
            self.max_length = 2048
            self.context_length = 1024
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.llm_name, cache_dir=self.cache_dir
            )
            if "mixtral" in llm_name.lower():
                self.tokenizer.chat_template = (
                    open("./templates/mistral-instruct.jinja")
                    .read()
                    .replace("    ", "")
                    .replace("\n", "")
                )
                self.max_length = 32768
                self.context_length = 30000
            elif "llama-2" in llm_name.lower():
                self.max_length = 4096
                self.context_length = 3072
            elif "llama-3" in llm_name.lower():
                self.max_length = 8192
                self.context_length = 7168
                if ".1" in llm_name or ".2" in llm_name:
                    self.max_length = 131072
                    self.context_length = 128000
            elif "meditron-70b" in llm_name.lower():
                self.tokenizer.chat_template = (
                    open("./templates/meditron.jinja")
                    .read()
                    .replace("    ", "")
                    .replace("\n", "")
                )
                self.max_length = 4096
                self.context_length = 3072
                self.templates["cot_prompt"] = meditron_cot
                self.templates["medrag_prompt"] = meditron_medrag
            elif "pmc_llama" in llm_name.lower():
                self.tokenizer.chat_template = (
                    open("./templates/pmc_llama.jinja")
                    .read()
                    .replace("    ", "")
                    .replace("\n", "")
                )
                self.max_length = 2048
                self.context_length = 1024
            
            gc.collect()
            torch.cuda.empty_cache()

            self.model = transformers.pipeline(
                "text-generation",
                model=self.llm_name,
                torch_dtype=torch.float16,
                # torch_dtype=torch.bfloat16,
                device_map="auto",
                model_kwargs={"cache_dir": self.cache_dir},
            )

        # FLARE related parameters
        self.enable_flare = enable_flare
        self.look_ahead_steps = look_ahead_steps
        self.look_ahead_boundary = look_ahead_boundary
        self.look_ahead_truncate_at_boundary = look_ahead_truncate_at_boundary
        self.look_ahead_filter_prob = look_ahead_filter_prob
        self.look_ahead_mask_prob = look_ahead_mask_prob
        self.look_ahead_mask_method = look_ahead_mask_method
        self.look_ahead_pre_retrieval = look_ahead_pre_retrieval
        self.only_use_look_ahead = only_use_look_ahead
        self.use_full_input_as_query = use_full_input_as_query
        self.max_query_length = max_query_length
        self.final_stop_sym = final_stop_sym

        # Original follow-up and answer method selection
        self.follow_up = follow_up
        if self.rag:
            if self.follow_up and self.enable_flare:
                self.answer = self.flare_medrag_answer
                self.templates["medrag_system"] = simple_medrag_system
                self.templates["medrag_prompt"] = simple_medrag_prompt
                self.templates["i_medrag_system"] = i_medrag_system
                self.templates["follow_up_ask"] = follow_up_instruction_ask
                self.templates["follow_up_answer"] = follow_up_instruction_answer
            elif self.follow_up:
                self.answer = self.i_medrag_answer
                self.templates["medrag_system"] = simple_medrag_system
                self.templates["medrag_prompt"] = simple_medrag_prompt
                self.templates["i_medrag_system"] = i_medrag_system
                self.templates["follow_up_ask"] = follow_up_instruction_ask
                self.templates["follow_up_answer"] = follow_up_instruction_answer
            elif self.enable_flare:
                self.answer = self.flare_medrag_answer
            else:
                self.answer = self.medrag_answer
        else:
            self.answer = self.medrag_answer

    def custom_stop(self, stop_str, input_len=0):
        stopping_criteria = StoppingCriteriaList(
            [CustomStoppingCriteria(stop_str, self.tokenizer, input_len)]
        )
        return stopping_criteria

    # Add retry decorator for rate limiting
    @retry(
        stop=stop_after_attempt(5),  # Retry up to 5 times
        wait=wait_exponential(multiplier=1, min=2, max=60),  # Wait 2s, 4s, 8s, 16s, 32s(max capped at 60s)
        retry=retry_if_exception_type(RateLimitError) # Only retry on RateLimitError
    )
    def generate(self, messages, **kwargs):
        """
        generate response given messages
        """
        if self.verbose:
            print(f"[VERBOSE] Making API call to {self.llm_name}")
            print(f"[VERBOSE] Messages: {json.dumps(messages, indent=2)}")
            print(f"[VERBOSE] Parameters: {json.dumps(kwargs, indent=2)}")
            
        if "openai" in self.llm_name.lower():
            start_time = time.time()
            ans = openai_client(
                model=self.model, messages=messages, temperature=0.0, **kwargs
            )
            if self.verbose:
                print(f"[VERBOSE] API call completed in {time.time() - start_time:.2f} seconds")
                print(f"[VERBOSE] Response: {ans[:100]}...")
        elif "gemini" in self.llm_name.lower():
            start_time = time.time()
            response = self.model.generate_content(
                messages[0]["content"] + "\n\n" + messages[1]["content"], **kwargs
            )
            ans = response.candidates[0].content.parts[0].text
            if self.verbose:
                print(f"[VERBOSE] API call completed in {time.time() - start_time:.2f} seconds")
                print(f"[VERBOSE] Response: {ans[:100]}...")
        else:
            stopping_criteria = None
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            if "meditron" in self.llm_name.lower():
                # stopping_criteria = custom_stop(["###", "User:", "\n\n\n"], self.tokenizer, input_len=len(self.tokenizer.encode(prompt_cot, add_special_tokens=True)))
                stopping_criteria = self.custom_stop(
                    ["###", "User:", "\n\n\n"],
                    input_len=len(
                        self.tokenizer.encode(prompt, add_special_tokens=True)
                    ),
                )
            if self.verbose:
                print(f"[VERBOSE] Processing prompt with length: {len(prompt)}")
                
            start_time = time.time()
            if "llama-3" in self.llm_name.lower():
                response = self.model(
                    prompt,
                    do_sample=True,
                    temperature=0.6,
                    top_p=0.9,
                    eos_token_id=[
                        self.tokenizer.eos_token_id,
                        self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
                    ],
                    pad_token_id=self.tokenizer.eos_token_id,
                    max_length=self.max_length,
                    truncation=True,
                    stopping_criteria=stopping_criteria,
                    **kwargs,
                )
            else:
                response = self.model(
                    prompt,
                    do_sample=True,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.eos_token_id,
                    max_length=self.max_length,
                    truncation=True,
                    stopping_criteria=stopping_criteria,
                    **kwargs,
                )
            # ans = response[0]["generated_text"]
            ans = response[0]["generated_text"][len(prompt) :]
            if self.verbose:
                print(f"[VERBOSE] Generation completed in {time.time() - start_time:.2f} seconds")
                print(f"[VERBOSE] Response: {ans[:100]}...")
                
        return ans

    def medrag_answer(
        self,
        question,
        options=None,
        k=32,
        rrf_k=100,
        save_dir=None,
        snippets=None,
        snippets_ids=None,
        **kwargs,
    ):
        """
        question (str): question to be answered
        options (Dict[str, str]): options to be chosen from
        k (int): number of snippets to retrieve
        rrf_k (int): parameter for Reciprocal Rank Fusion
        save_dir (str): directory to save the results
        snippets (List[Dict]): list of snippets to be used
        snippets_ids (List[Dict]): list of snippet ids to be used
        """
        if self.verbose:
            print(f"[VERBOSE] medrag_answer called with question: {question[:50]}...")
            print(f"[VERBOSE] Options: {options}")
            print(f"[VERBOSE] k={k}, rrf_k={rrf_k}")

        if options is not None:
            options = "\n".join(
                [key + ". " + options[key] for key in sorted(options.keys())]
            )
        else:
            options = ""

        # retrieve relevant snippets
        if self.rag:
            if snippets is not None:
                if self.verbose:
                    print(f"[VERBOSE] Using provided snippets: {len(snippets)}")
                retrieved_snippets = snippets[:k]
                scores = []
            elif snippets_ids is not None:
                if self.verbose:
                    print(f"[VERBOSE] Using provided snippet IDs: {len(snippets_ids)}")
                if self.docExt is None:
                    self.docExt = DocExtracter(
                        db_dir=self.db_dir, cache=True, corpus_name=self.corpus_name
                    )
                retrieved_snippets = self.docExt.extract(snippets_ids[:k])
                scores = []
            else:
                if self.verbose:
                    print(f"[VERBOSE] Retrieving snippets for question: {question[:50]}...")
                assert self.retrieval_system is not None
                start_time = time.time()
                retrieved_snippets, scores = self.retrieval_system.retrieve(
                    question, k=k, rrf_k=rrf_k
                )
                if self.verbose:
                    print(f"[VERBOSE] Retrieval completed in {time.time() - start_time:.2f} seconds")
                    print(f"[VERBOSE] Retrieved {len(retrieved_snippets)} snippets")

            contexts = [
                "Document [{:d}] (Title: {:s}) {:s}".format(
                    idx,
                    retrieved_snippets[idx]["title"],
                    retrieved_snippets[idx]["content"],
                )
                for idx in range(len(retrieved_snippets))
            ]
            if len(contexts) == 0:
                contexts = [""]
            if self.verbose:
                print(f"[VERBOSE] Tokenizing and truncating contexts")
            if "openai" in self.llm_name.lower():
                contexts = [
                    self.tokenizer.decode(
                        self.tokenizer.encode("\n".join(contexts))[
                            : self.context_length
                        ]
                    )
                ]
            elif "gemini" in self.llm_name.lower():
                contexts = [
                    self.tokenizer.decode(
                        self.tokenizer.encode("\n".join(contexts))[
                            : self.context_length
                        ]
                    )
                ]
            else:
                contexts = [
                    self.tokenizer.decode(
                        self.tokenizer.encode(
                            "\n".join(contexts), add_special_tokens=False
                        )[: self.context_length]
                    )
                ]
        else:
            if self.verbose:
                print(f"[VERBOSE] RAG disabled, proceeding without retrieval")
            retrieved_snippets = []
            scores = []
            contexts = []

        if save_dir is not None and not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # generate answers
        answers = []
        if not self.rag:
            if self.verbose:
                print(f"[VERBOSE] Generating CoT answer without RAG")
            prompt_cot = self.templates["cot_prompt"].render(
                question=question, options=options
            )
            messages = [
                {"role": "system", "content": self.templates["cot_system"]},
                {"role": "user", "content": prompt_cot},
            ]
            ans = self.generate(messages, **kwargs)
            answers.append(re.sub("\s+", " ", ans))
        else:
            if self.verbose:
                print(f"[VERBOSE] Generating RAG-enhanced answers")
            for i, context in enumerate(contexts):
                if self.verbose:
                    print(f"[VERBOSE] Processing context {i+1}/{len(contexts)}")
                prompt_medrag = self.templates["medrag_prompt"].render(
                    context=context, question=question, options=options
                )
                messages = [
                    {"role": "system", "content": self.templates["medrag_system"]},
                    {"role": "user", "content": prompt_medrag},
                ]
                ans = self.generate(messages, **kwargs)
                answers.append(re.sub("\s+", " ", ans))

        if save_dir is not None:
            if self.verbose:
                print(f"[VERBOSE] Saving results to {save_dir}")
            with open(os.path.join(save_dir, "snippets.json"), "w") as f:
                json.dump(retrieved_snippets, f, indent=4)
            with open(os.path.join(save_dir, "response.json"), "w") as f:
                json.dump(answers, f, indent=4)

        return answers[0] if len(answers) == 1 else answers, retrieved_snippets, scores

    def i_medrag_answer(
        self,
        question,
        options=None,
        k=32,
        rrf_k=100,
        save_path=None,
        n_rounds=4,
        n_queries=3,
        qa_cache_path=None,
        **kwargs,
    ):
        if options is not None:
            options = "\n".join(
                [key + ". " + options[key] for key in sorted(options.keys())]
            )
        else:
            options = ""
        QUESTION_PROMPT = f"Here is the question:\n{question}\n\n{options}"

        context = ""
        qa_cache = []
        if qa_cache_path is not None and os.path.exists(qa_cache_path):
            qa_cache = eval(open(qa_cache_path, "r").read())[:n_rounds]
            if len(qa_cache) > 0:
                context = qa_cache[-1]
            n_rounds = n_rounds - len(qa_cache)
        last_context = None

        # Run in loop
        max_iterations = n_rounds + 3
        saved_messages = [
            {"role": "system", "content": self.templates["i_medrag_system"]}
        ]

        for i in range(max_iterations):
            if i < n_rounds:
                if context == "":
                    messages = [
                        {
                            "role": "system",
                            "content": self.templates["i_medrag_system"],
                        },
                        {
                            "role": "user",
                            "content": f"{QUESTION_PROMPT}\n\n{self.templates['follow_up_ask'].format(n_queries)}",
                        },
                    ]
                else:
                    messages = [
                        {
                            "role": "system",
                            "content": self.templates["i_medrag_system"],
                        },
                        {
                            "role": "user",
                            "content": f"{context}\n\n{QUESTION_PROMPT}\n\n{self.templates['follow_up_ask'].format(n_queries)}",
                        },
                    ]
            elif context != last_context:
                messages = [
                    {
                        "role": "system",
                        "content": self.templates["i_medrag_system"],
                    },
                    {
                        "role": "user",
                        "content": f"{context}\n\n{QUESTION_PROMPT}\n\n{self.templates['follow_up_answer']}",
                    },
                ]
            elif len(messages) == 1:
                messages = [
                    {
                        "role": "system",
                        "content": self.templates["i_medrag_system"],
                    },
                    {
                        "role": "user",
                        "content": f"{context}\n\n{QUESTION_PROMPT}\n\n{self.templates['follow_up_answer']}",
                    },
                ]
            saved_messages.append(messages[-1])
            if save_path:
                with open(save_path, "w") as f:
                    json.dump(
                        [
                            p if type(p) == dict else p.model_dump()
                            for p in saved_messages
                        ],
                        f,
                        indent=4,
                    )
            last_context = context
            last_content = self.generate(messages, **kwargs)
            response_message = {"role": "assistant", "content": last_content}
            saved_messages.append(response_message)
            if save_path:
                with open(save_path, "w") as f:
                    json.dump(
                        [
                            p if type(p) == dict else p.model_dump()
                            for p in saved_messages
                        ],
                        f,
                        indent=4,
                    )
            if i >= n_rounds and (
                "## Answer" in last_content or "answer is" in last_content.lower()
            ):
                messages.append(response_message)
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Output the answer in JSON: {'answer': your_answer (A/B/C/D)}"
                            if options
                            else "Output the answer in JSON: {'answer': your_answer}"
                        ),
                    }
                )
                saved_messages.append(messages[-1])
                answer_content = self.generate(messages, **kwargs)
                answer_message = {"role": "assistant", "content": answer_content}
                messages.append(answer_message)
                saved_messages.append(messages[-1])
                if save_path:
                    with open(save_path, "w") as f:
                        json.dump(
                            [
                                p if type(p) == dict else p.model_dump()
                                for p in saved_messages
                            ],
                            f,
                            indent=4,
                        )
                return messages[-1]["content"], messages
            elif "## Queries" in last_content:
                messages = messages[:-1]
                if last_content.split("## Queries")[-1].strip() == "":
                    print("Empty queries. Continue with next iteration.")
                    continue
                try:
                    action_str = self.generate(
                        [
                            {
                                "role": "user",
                                "content": f'Parse the following passage and extract the queries as a list: {last_content}.\n\nPresent the queries as they are. DO NOT merge or break down queries. Output the list of queries in JSON format: {{"output": ["query 1", ..., "query N"]}}',
                            }
                        ],
                        **kwargs,
                    )
                    action_str = re.search(
                        r"output\": (\[.*\])", action_str, re.DOTALL
                    ).group(1)
                    action_list = [
                        re.sub(r"^\d+\.\s*", "", s.strip()) for s in eval(action_str)
                    ]
                except Exception as E:
                    print("Error parsing action list. Continue with next iteration.")
                    error_class = E.__class__.__name__
                    error = f"{error_class}: {str(E)}"
                    print(error)
                    if save_path:
                        with open(save_path + ".error", "a") as f:
                            f.write(f"{error}\n")
                    continue
                for question in action_list:
                    if question.strip() == "":
                        continue
                    try:
                        rag_result = self.medrag_answer(
                            question, k=k, rrf_k=rrf_k, **kwargs
                        )[0]
                        context += f"\n\nQuery: {question}\nAnswer: {rag_result}"
                        context = context.strip()
                    except Exception as E:
                        error_class = E.__class__.__name__
                        error = f"{error_class}: {str(E)}"
                        print(error)
                        if save_path:
                            with open(save_path + ".error", "a") as f:
                                f.write(f"{error}\n")
                qa_cache.append(context)
                if qa_cache_path:
                    with open(qa_cache_path, "w") as f:
                        json.dump(qa_cache, f, indent=4)
            else:
                messages.append(response_message)
                print("No queries or answer. Continue with next iteration.")
                continue
        return messages[-1]["content"], messages

    def flare_medrag_answer(
        self,
        question,
        options=None,
        k=32,
        rrf_k=100,
        save_dir=None,
        snippets=None,
        snippets_ids=None,
        n_rounds=4,
        n_queries=3,
        qa_cache_path=None,
        **kwargs,
    ):
        """
        Implements FLARE's look ahead capability for retrieval augmented generation
        while supporting follow-up questions when enabled.
        
        Parameters:
        -----------
        question (str): question to be answered
        options (Dict[str, str]): options to be chosen from
        k (int): number of snippets to retrieve
        rrf_k (int): parameter for Reciprocal Rank Fusion
        save_dir (str): directory to save the results
        snippets (List[Dict]): list of snippets to be used
        snippets_ids (List[Dict]): list of snippet ids to be used
        n_rounds (int): number of rounds for follow-up queries
        n_queries (int): number of queries to generate per round
        qa_cache_path (str): path to save the QA cache
        """

        if options is not None:
            options = "\n".join(
                [key + ". " + options[key] for key in sorted(options.keys())]
            )
        else:
            options = ""

        # Handle follow-up questions functionality if enabled
        if self.follow_up:
            return self._flare_with_follow_up(
                question, 
                options, 
                k, 
                rrf_k, 
                save_dir,
                n_rounds, 
                n_queries, 
                qa_cache_path, 
                **kwargs
            )
        
        # Process with FLARE's look ahead retrieval without follow-up
        return self._flare_without_follow_up(
            question, 
            options, 
            k, 
            rrf_k, 
            save_dir, 
            snippets, 
            snippets_ids, 
            **kwargs
        )
    
    def _flare_without_follow_up(
        self,
        question,
        options,
        k=32,
        rrf_k=100,
        save_dir=None,
        snippets=None,
        snippets_ids=None,
        **kwargs,
    ):
        """
        Implements basic FLARE look ahead capability for standard RAG
        """
        if self.verbose:
            print(f"[VERBOSE] _flare_without_follow_up called with question: {question[:50]}...")
            print(f"[VERBOSE] FLARE parameters: steps={self.look_ahead_steps}, boundary={self.look_ahead_boundary}")
            
        # Initial retrieval (standard approach)
        if snippets is not None:
            if self.verbose:
                print(f"[VERBOSE] Using provided snippets: {len(snippets)}")
            retrieved_snippets = snippets[:k]
            scores = []
        elif snippets_ids is not None:
            if self.verbose:
                print(f"[VERBOSE] Using provided snippet IDs: {len(snippets_ids)}")
            if self.docExt is None:
                self.docExt = DocExtracter(
                    db_dir=self.db_dir, cache=True, corpus_name=self.corpus_name
                )
            retrieved_snippets = self.docExt.extract(snippets_ids[:k])
            scores = []
        else:
            if self.verbose:
                print(f"[VERBOSE] Initial retrieval for question: {question[:50]}...")
            assert self.retrieval_system is not None
            start_time = time.time()
            retrieved_snippets, scores = self.retrieval_system.retrieve(
                question, k=k, rrf_k=rrf_k
            )
            if self.verbose:
                print(f"[VERBOSE] Initial retrieval completed in {time.time() - start_time:.2f} seconds")
                print(f"[VERBOSE] Retrieved {len(retrieved_snippets)} snippets")
        
        # Format initial context
        initial_contexts = [
            "Document [{:d}] (Title: {:s}) {:s}".format(
                idx,
                retrieved_snippets[idx]["title"],
                retrieved_snippets[idx]["content"],
            )
            for idx in range(len(retrieved_snippets))
        ]
        
        if len(initial_contexts) == 0:
            initial_contexts = [""]
            
        # Tokenize and truncate initial context
        if "openai" in self.llm_name.lower() or "gemini" in self.llm_name.lower():
            contexts = [
                self.tokenizer.decode(
                    self.tokenizer.encode("\n".join(initial_contexts))[
                        : self.context_length
                    ]
                )
            ]
        else:
            contexts = [
                self.tokenizer.decode(
                    self.tokenizer.encode(
                        "\n".join(initial_contexts), add_special_tokens=False
                    )[: self.context_length]
                )
            ]
        
        # FLARE look ahead approach: generate partial answer to inform retrieval
        if self.look_ahead_steps > 0 or self.look_ahead_boundary:
            # Create initial prompt for look ahead
            prompt_medrag = self.templates["medrag_prompt"].render(
                context=contexts[0], question=question, options=options
            )
            messages = [
                {"role": "system", "content": self.templates["medrag_system"]},
                {"role": "user", "content": prompt_medrag},
            ]
            
            # Generate look ahead content using specified parameters
            if self.look_ahead_steps > 0:
                look_ahead_params = {'max_tokens': self.look_ahead_steps, 'stop': self.final_stop_sym}
                look_ahead_response = self.generate(messages, **look_ahead_params)
                # Truncate if needed
                if self.look_ahead_truncate_at_boundary:
                    for boundary in self.look_ahead_truncate_at_boundary:
                        if boundary in look_ahead_response:
                            look_ahead_response = look_ahead_response.split(boundary)[0]
            elif self.look_ahead_boundary:
                look_ahead_params = {'max_tokens': self.max_length, 'stop': self.look_ahead_boundary}
                look_ahead_response = self.generate(messages, **look_ahead_params)
            
            # Filter low-probability tokens if configured
            if self.look_ahead_filter_prob > 0:
                # Simple filtering - in a real implementation, we would need token probabilities
                # For now, we'll just use the text as is since we don't have token probabilities
                look_ahead_query = look_ahead_response
            else:
                look_ahead_query = look_ahead_response
            
            # Combine original question with look ahead content for better retrieval
            combined_query = question if self.only_use_look_ahead else question + " " + look_ahead_query
            
            # Limit query length if specified
            if self.max_query_length and not self.use_full_input_as_query:
                combined_query = combined_query[:self.max_query_length]
            
            # Second retrieval with enhanced query
            enhanced_snippets, enhanced_scores = self.retrieval_system.retrieve(
                combined_query, k=k, rrf_k=rrf_k
            )
            
            # Format enhanced context
            enhanced_contexts = [
                "Document [{:d}] (Title: {:s}) {:s}".format(
                    idx,
                    enhanced_snippets[idx]["title"],
                    enhanced_snippets[idx]["content"],
                )
                for idx in range(len(enhanced_snippets))
            ]
            
            if len(enhanced_contexts) == 0:
                enhanced_contexts = [""]
                
            # Tokenize and truncate enhanced context
            if "openai" in self.llm_name.lower() or "gemini" in self.llm_name.lower():
                contexts = [
                    self.tokenizer.decode(
                        self.tokenizer.encode("\n".join(enhanced_contexts))[
                            : self.context_length
                        ]
                    )
                ]
            else:
                contexts = [
                    self.tokenizer.decode(
                        self.tokenizer.encode(
                            "\n".join(enhanced_contexts), add_special_tokens=False
                        )[: self.context_length]
                    )
                ]
            
            # Use enhanced snippets for return
            retrieved_snippets = enhanced_snippets
            scores = enhanced_scores
        
        # Generate final answer with best context
        answers = []
        for context in contexts:
            prompt_medrag = self.templates["medrag_prompt"].render(
                context=context, question=question, options=options
            )
            messages = [
                {"role": "system", "content": self.templates["medrag_system"]},
                {"role": "user", "content": prompt_medrag},
            ]
            ans = self.generate(messages, **kwargs)
            answers.append(re.sub("\s+", " ", ans))
        
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            with open(os.path.join(save_dir, "snippets.json"), "w") as f:
                json.dump(retrieved_snippets, f, indent=4)
            with open(os.path.join(save_dir, "response.json"), "w") as f:
                json.dump(answers, f, indent=4)
        
        return answers[0] if len(answers) == 1 else answers, retrieved_snippets, scores

    def _flare_with_follow_up(
        self,
        question,
        options,
        k=32,
        rrf_k=100,
        save_path=None,
        n_rounds=4,
        n_queries=3,
        qa_cache_path=None,
        **kwargs,
    ):
        """
        Combines FLARE look ahead with follow-up questions capability
        """
        QUESTION_PROMPT = f"Here is the question:\n{question}\n\n{options}"

        context = ""
        qa_cache = []
        if qa_cache_path is not None and os.path.exists(qa_cache_path):
            qa_cache = eval(open(qa_cache_path, "r").read())[:n_rounds]
            if len(qa_cache) > 0:
                context = qa_cache[-1]
            n_rounds = n_rounds - len(qa_cache)
        last_context = None

        # Run in loop
        max_iterations = n_rounds + 3
        saved_messages = [
            {"role": "system", "content": self.templates["i_medrag_system"]}
        ]

        for i in range(max_iterations):
            if i < n_rounds:
                if context == "":
                    messages = [
                        {
                            "role": "system",
                            "content": self.templates["i_medrag_system"],
                        },
                        {
                            "role": "user",
                            "content": f"{QUESTION_PROMPT}\n\n{self.templates['follow_up_ask'].format(n_queries)}",
                        },
                    ]
                else:
                    messages = [
                        {
                            "role": "system",
                            "content": self.templates["i_medrag_system"],
                        },
                        {
                            "role": "user",
                            "content": f"{context}\n\n{QUESTION_PROMPT}\n\n{self.templates['follow_up_ask'].format(n_queries)}",
                        },
                    ]
            elif context != last_context:
                messages = [
                    {
                        "role": "system",
                        "content": self.templates["i_medrag_system"],
                    },
                    {
                        "role": "user",
                        "content": f"{context}\n\n{QUESTION_PROMPT}\n\n{self.templates['follow_up_answer']}",
                    },
                ]
            elif len(messages) == 1:
                messages = [
                    {
                        "role": "system",
                        "content": self.templates["i_medrag_system"],
                    },
                    {
                        "role": "user",
                        "content": f"{context}\n\n{QUESTION_PROMPT}\n\n{self.templates['follow_up_answer']}",
                    },
                ]
            
            saved_messages.append(messages[-1])
            if save_path:
                with open(save_path, "w") as f:
                    json.dump(
                        [
                            p if type(p) == dict else p.model_dump()
                            for p in saved_messages
                        ],
                        f,
                        indent=4,
                    )
            
            last_context = context
            last_content = self.generate(messages, **kwargs)
            response_message = {"role": "assistant", "content": last_content}
            saved_messages.append(response_message)
            
            if save_path:
                with open(save_path, "w") as f:
                    json.dump(
                        [
                            p if type(p) == dict else p.model_dump()
                            for p in saved_messages
                        ],
                        f,
                        indent=4,
                    )
            
            if i >= n_rounds and (
                "## Answer" in last_content or "answer is" in last_content.lower()
            ):
                messages.append(response_message)
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Output the answer in JSON: {'answer': your_answer (A/B/C/D)}"
                            if options
                            else "Output the answer in JSON: {'answer': your_answer}"
                        ),
                    }
                )
                saved_messages.append(messages[-1])
                answer_content = self.generate(messages, **kwargs)
                answer_message = {"role": "assistant", "content": answer_content}
                messages.append(answer_message)
                saved_messages.append(messages[-1])
                
                if save_path:
                    with open(save_path, "w") as f:
                        json.dump(
                            [
                                p if type(p) == dict else p.model_dump()
                                for p in saved_messages
                            ],
                            f,
                            indent=4,
                        )
                
                return messages[-1]["content"], messages
            
            elif "## Queries" in last_content:
                messages = messages[:-1]
                if last_content.split("## Queries")[-1].strip() == "":
                    print("Empty queries. Continue with next iteration.")
                    continue
                
                try:
                    action_str = self.generate(
                        [
                            {
                                "role": "user",
                                "content": f'Parse the following passage and extract the queries as a list: {last_content}.\n\nPresent the queries as they are. DO NOT merge or break down queries. Output the list of queries in JSON format: {{"output": ["query 1", ..., "query N"]}}',
                            }
                        ],
                        **kwargs,
                    )
                    action_str = re.search(
                        r"output\": (\[.*\])", action_str, re.DOTALL
                    ).group(1)
                    action_list = [
                        re.sub(r"^\d+\.\s*", "", s.strip()) for s in eval(action_str)
                    ]
                except Exception as E:
                    print("Error parsing action list. Continue with next iteration.")
                    error_class = E.__class__.__name__
                    error = f"{error_class}: {str(E)}"
                    print(error)
                    if save_path:
                        with open(save_path + ".error", "a") as f:
                            f.write(f"{error}\n")
                    continue
                
                for question in action_list:
                    if question.strip() == "":
                        continue
                    
                    try:
                        # Use FLARE's look ahead capability for each follow-up question
                        if self.look_ahead_steps > 0 or self.look_ahead_boundary:
                            # First generate a partial answer to the follow-up question
                            look_ahead_messages = [
                                {"role": "system", "content": self.templates["cot_system"]},
                                {"role": "user", "content": f"Question: {question}"},
                            ]
                            
                            if self.look_ahead_steps > 0:
                                look_ahead_params = {'max_tokens': self.look_ahead_steps, 'stop': None}
                                look_ahead_response = self.generate(look_ahead_messages, **look_ahead_params)
                            else:
                                look_ahead_params = {'max_tokens': self.max_length, 'stop': self.look_ahead_boundary}
                                look_ahead_response = self.generate(look_ahead_messages, **look_ahead_params)
                            
                            # Enhance the query with the look ahead content
                            enhanced_question = question if self.only_use_look_ahead else question + " " + look_ahead_response
                            
                            # Limit query length if specified
                            if self.max_query_length and not self.use_full_input_as_query:
                                enhanced_question = enhanced_question[:self.max_query_length]
                            
                            # Use enhanced query for retrieval
                            rag_result = self._flare_without_follow_up(
                                enhanced_question, None, k=k, rrf_k=rrf_k, **kwargs
                            )[0]
                        else:
                            # Use standard retrieval without look ahead
                            rag_result = self.medrag_answer(
                                question, k=k, rrf_k=rrf_k, **kwargs
                            )[0]
                        
                        context += f"\n\nQuery: {question}\nAnswer: {rag_result}"
                        context = context.strip()
                    except Exception as E:
                        error_class = E.__class__.__name__
                        error = f"{error_class}: {str(E)}"
                        print(error)
                        if save_path:
                            with open(save_path + ".error", "a") as f:
                                f.write(f"{error}\n")
                
                qa_cache.append(context)
                if qa_cache_path:
                    with open(qa_cache_path, "w") as f:
                        json.dump(qa_cache, f, indent=4)
            else:
                messages.append(response_message)
                print("No queries or answer. Continue with next iteration.")
                continue
                
        return messages[-1]["content"], messages


class CustomStoppingCriteria(StoppingCriteria):
    def __init__(self, stop_words, tokenizer, input_len=0):
        super().__init__()
        self.tokenizer = tokenizer
        self.stops_words = stop_words
        self.input_len = input_len

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        tokens = self.tokenizer.decode(input_ids[0][self.input_len :])
        return any(stop in tokens for stop in self.stops_words)
