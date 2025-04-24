import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    StoppingCriteria,
    StoppingCriteriaList,
)
from typing import List, Dict, Any, Tuple, Optional
import time
import json  # For potential saving
import os  # For potential saving


class LlamaFlareAgent:
    def __init__(
        self,
        model_name_or_path: str,
        retriever: object,
        tokenizer_name_or_path: Optional[str] = None,
        device_map: str = "auto",
        torch_dtype=torch.float16,
        trust_remote_code: bool = False,
        # --- FLARE Control Parameters ---
        uncertainty_threshold: float = 0.5,  # Probability threshold to trigger retrieval
        lookahead_tokens: int = 5,  # How many tokens to generate tentatively
        max_flare_rounds: int = 3,  # Max retrieval steps per query
        # --- Retriever Parameters ---
        dynamic_k: int = 3,  # How many documents to retrieve in dynamic steps
        # --- LLM Parameters ---
        temperature: float = 0.1,  # Low temp for more factual generation
        top_p: float = 1.0,
        max_total_tokens: int = 512,  # Max length of the final generated answer
        # --- Misc ---
        debug: bool = False,
    ):
        self.retriever = retriever

        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self.trust_remote_code = trust_remote_code

        self.uncertainty_threshold = uncertainty_threshold
        self.lookahead_tokens = lookahead_tokens
        self.max_flare_rounds = max_flare_rounds

        self.dynamic_k = dynamic_k

        self.temperature = temperature
        self.top_p = top_p
        self.max_total_tokens = max_total_tokens

        self.debug = debug

        self._load_model_and_tokenizer(model_name_or_path, tokenizer_name_or_path)

        print("[Init] LlamaFLAREAgent Initialized.")

    def _load_model_and_tokenizer(self, model_path, tokenizer_path):
        _tokenizer_path = tokenizer_path if tokenizer_path else model_path
        self.tokenizer = AutoTokenizer.from_pretrained(
            _tokenizer_path, trust_remote_code=self.trust_remote_code
        )
        print("[Init] Tokenizer loaded.")

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=self.device_map,
            torch_dtype=self.torch_dtype,
            trust_remote_code=self.trust_remote_code,
        )
        self.model.eval()
        print(f"[Init] Model loaded on {self.model.device}")

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            if hasattr(self.model, "config"):
                self.model.config.pad_token_id = self.tokenizer.pad_token_id
            else:
                print("[ERROR] Could not set model.config.pad_token_id")

    def _generate_incremental(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int,
    ):
        _do_sample = self.temperature > 0
        _temperature = (
            self.temperature if _do_sample else 1.0
        )  # why does not take self.temperature
        _top_p = self.top_p if _do_sample else None  # why does not take self.top_p

        eos_token_id_list = [self.tokenizer.eos_token_id]

        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    temperature=_temperature,
                    top_p=_top_p,
                    do_sample=_do_sample,
                    output_scores=True,
                    return_dict_in_generate=True,
                    eos_token_id=list(set(eos_token_id_list)),
                    pad_token_id=self.tokenizer.pad_token_id,
                )
                new_token_ids = outputs.sequences[0][input_ids.shape[1] :].tolist()
                
                generated_probs = []
                if outputs.scores and new_token_ids:
                    num_generated_steps = len(new_token_ids)
                    for step in range(num_generated_steps):
                        if step < len(outputs.scores):
                            # Get the logits for the current step
                            logits = outputs.scores[step][0, :]
                            probs = F.softmax(logits, dim=-1)
                            # Get the token ID of the generated token
                            token_id = new_token_ids[step]
                            if 0 <= token_id < probs.shape[-1]:
                                token_prob = probs[token_id].item()
                                generated_probs.append(token_prob)
                            else:
                                generated_probs.append(0.0)
                        else:
                            generated_probs.append(0.0)

                print(f"[INFO] Generated token IDs: {new_token_ids}")

            return new_token_ids, generated_probs, list(set(eos_token_id_list))

        except Exception as e:
            print(f"[ERROR] During _generate_incremental: {e}")
            return None, None, None

    def _check_uncertainty(self, probs: List[float]) -> Tuple[bool, int]:
        if not probs:
            return False, 0

        for i, p in enumerate(probs):
            if p < self.uncertainty_threshold:
                return True, i

        return False, -1

    def _formulate_dynamic_query(
        self,
        question: str,
        current_answer_text: str,
        uncertain_lookahead_ids: List[int],
    ) -> str:
        uncertain_text = self.tokenizer.decode(
            uncertain_lookahead_ids, skip_special_tokens=True
        )

        context_window = 100
        context_for_query = current_answer_text[-context_window:]

        query = f"Regarding the question '{question}', what specific information is needed to continue or clarify after this: '{context_for_query} {uncertain_text}'?"

        max_q_tokens = 64
        encoded_query = self.tokenizer.encode(query)
        if len(encoded_query) > max_q_tokens:
            query = (
                self.tokenizer.decode(
                    encoded_query[:max_q_tokens], skip_special_tokens=True
                )
                + "..."
            )

        if self.debug:
            print(f"[DEBUG] Dynamic Query: {query}")

        return query

    def _perform_retrieval(self, query: str) -> List[Dict]:
        if self.debug:
            print(f"[DEBUG] Performing retrieval for query: {query}")

        try:
            snippets, scores = self.retriever.retrieve(query, k=self.dynamic_k)

            if self.debug:
                print(f"[DEBUG] Retrieved {len(snippets)} snippets.")
                return snippets
        except Exception as e:
            print(f"[Error] During retrieval: {e}")
            return []

    def _format_context(self, snippets: List[Dict]) -> str:
        contexts = [
            f"Document [{idx + 1}] (Title: {snippet.get('title', 'N/A')})\n{snippet.get('content', '')}"
            for idx, snippet in enumerate(snippets)
        ]
        return "\n\n".join(contexts)

    def _prepare_llm_input(
        self, base_prompt: str, context_str: str, current_gen_text: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        full_prompt_text = f"{base_prompt}\n\nRelevant Documents:\n{context_str}\n\nCurrent Answer Draft:\n{current_gen_text}"
        try:
            # Example: Assuming base_prompt is instruction, rest is user turn
            messages = [
                # {"role": "system", "content": base_prompt}, # If base_prompt is system instruction
                {
                    "role": "user",
                    "content": full_prompt_text,
                }  # Or structure differently
            ]
            final_input_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except:
            print(
                "[Warning] Could not apply chat template. Using combined text directly."
            )
            final_input_text = full_prompt_text  # Fallback

        max_input_len = self.tokenizer.model_max_length - self.max_total_tokens
        inputs = self.tokenizer(
            final_input_text,
            return_tensors="pt",
            padding=False,  # No padding needed for single input generation
            truncation=True,
            max_length=max_input_len,
        ).to(self.model.device)
            
        return inputs["input_ids"], inputs["attention_mask"]

    def generate_answer(
        self,
        question: str,
        initial_context_snippets: Optional[List[Dict]] = None,
        base_prompt: str = "Answer the following question step-by-step.",  # Base instruction
    ) -> Tuple[str, List[Dict]]:
        generated_answer_text = ""
        generated_token_count = 0
        flare_round_count = 0
        retrieval_history = []

        # --- Initial State ---
        current_context_snippets = (
            initial_context_snippets if initial_context_snippets else []
        )
        if not current_context_snippets:
            print("[Info] Performing initial retrieval based on question...")
            current_context_snippets = self._perform_retrieval(question)
            if not current_context_snippets:
                print("[Error] Initial retrieval yielded no documents.")

        retrieval_history.append(
            {"query": question, "step": "initial", "snippets": current_context_snippets}
        )

        if self.debug:
            print(f"[DEBUG] Retrieval history: {retrieval_history[-1]}")
        

        # --- Main Generation Loop ---
        while generated_token_count < self.max_total_tokens:

            if self.debug:
                print(
                    f"\n[DEBUG]--- Flare Step (Round {flare_round_count+1}, Gen Tokens {generated_token_count}) ---"
                )

            # 1. Format current prompt context string
            context_str = self._format_context(current_context_snippets)

            if self.debug:
                print(f"[DEBUG] Context string: {context_str}")

            # 2. Prepare LLM input
            input_ids, attention_mask = self._prepare_llm_input(
                base_prompt, context_str, generated_answer_text
            )

            # 3. Generate Tentative Lookahead
            lookahead_ids, lookahead_probs, eos_ids = self._generate_incremental(
                input_ids, attention_mask, self.lookahead_tokens
            )

            if self.debug:
                print(
                    f"[DEBUG] Lookahead IDs: {lookahead_ids}, Lookahead Probs: {lookahead_probs}, eos_ids: {eos_ids}"
                )

            if (
                lookahead_ids is None or not lookahead_ids
            ):  # Handle generation error or immediate EOS
                print("[Error] Generation yielded no tokens or errored.")
                break

            # 4. Check Uncertainty
            is_uncertain, uncertain_idx = self._check_uncertainty(lookahead_probs)

            if self.debug:
                print(f"[DEBUG] Uncertainty check: {is_uncertain}, Index: {uncertain_idx}")
            
            # 5. Active Retrieval (If Uncertain)
            if is_uncertain and flare_round_count < self.max_flare_rounds:
                if self.debug:
                    print(
                        f"  Uncertainty detected at index {uncertain_idx} (prob={lookahead_probs[uncertain_idx]:.3f})."
                    )
                flare_round_count += 1

                # a. Formulate Query
                query = self._formulate_dynamic_query(
                    question, generated_answer_text, lookahead_ids[: uncertain_idx + 1]
                )

                # b. Perform Retrieval
                new_snippets = self._perform_retrieval(query)

                # c. Update Context (Simple append & dedupe by ID if possible)
                if new_snippets:
                    seen_ids = {
                        s.get("id") for s in current_context_snippets if s.get("id")
                    }
                    added_count = 0
                    for s in new_snippets:
                        s_id = s.get("id")
                        if not s_id or s_id not in seen_ids:
                            current_context_snippets.append(s)
                            if s_id:
                                seen_ids.add(s_id)
                            added_count += 1
                    if self.debug:
                        print(f"  Added {added_count} new unique snippets to context.")

                retrieval_history.append(
                    {
                        "query": query,
                        "step": flare_round_count,
                        "snippets": new_snippets,
                    }
                )

                # d. Continue loop without adding the uncertain lookahead
                continue

            # 6. Finalize Generation (If Certain or Max Rounds Reached)
            else:
                # Max rounds reached but still uncertain
                if is_uncertain:
                    if self.debug:
                        print(
                            "[DEBUG] Max flare rounds reached, proceeding despite uncertainty."
                        )
                else:
                    if self.debug:
                        print("[DEBUG] No uncertainty detected, proceeding with generation.")

                # Certain
                # Decide segment to add (e.g., just the first token)
                segment_to_add_ids = lookahead_ids[:1]

                if (
                    not segment_to_add_ids
                ):  # Should not happen if lookahead_ids is not empty
                    break

                # Append the confirmed segment's text
                segment_text = self.tokenizer.decode(
                    lookahead_ids, skip_special_tokens=True
                )
                generated_answer_text += segment_text
                generated_token_count += len(segment_to_add_ids)

                if self.debug:
                    print(
                        f"[DEBUG] Appending: '{segment_text}' (Total Tokens ~{generated_token_count})"
                    )

                # 7. Check Stop Conditions (EOS token)
                if segment_to_add_ids[0] in eos_ids:
                    if self.debug:
                        print("  EOS token detected. Stopping generation.")
                
                break

        # --- End of Loop ---
        if generated_token_count >= self.max_total_tokens:
            print("Warning: Reached max_total_tokens limit.")

        return generated_answer_text, retrieval_history


# How to formulate the dynamic query
# Why segment_to_add_ids = lookahead_ids[:1]
