import os
import sys
import json
import argparse
from pathlib import Path
from MedRAG.src.medrag import MedRAG

def display_documents(snippets, scores=None):
    """Display retrieved documents with scores if available"""
    print("\n=== External Documents ===")
    for i, doc in enumerate(snippets):
        print(f"\nDocument #{i+1}:")
        if scores is not None and i < len(scores):
            print(f"Relevance Score: {scores[i]:.4f}")
        print(f"Content: {doc}")
        print("-" * 80)

def test_standard_medrag(question, options, verbose=False):
    """Test standard MedRAG without FLARE or follow-up functionality"""
    print("\n=== Testing Standard MedRAG ===")
    model = MedRAG(
        llm_name="OpenAI/gpt-4.1-nano",
        rag=True,
        follow_up=False,
        retriever_name="MedCPT",
        corpus_name="MedText",
        db_dir="./MedRAG/src/data/corpus",
        verbose=verbose
    )
    
    answer, snippets, scores = model.answer(question, options=options, k=10)
    print(f"Question: {question}")
    print(f"Options: {json.dumps(options, indent=2)}")
    print(f"Answer: {answer}")
    print(f"Number of snippets retrieved: {len(snippets)}")
    
    # Display retrieved documents
    display_documents(snippets, scores)
    
    return answer, snippets, scores

def test_flare_medrag(question, options, verbose=False):
    """Test MedRAG with FLARE's look ahead capability"""
    print("\n=== Testing FLARE-enhanced MedRAG ===")
    model = MedRAG(
        llm_name="OpenAI/gpt-4.1-nano",
        rag=True,
        follow_up=False,
        retriever_name="MedCPT",
        corpus_name="MedText",
        db_dir="./MedRAG/src/data/corpus",
        enable_flare=True,
        look_ahead_steps=50,
        look_ahead_truncate_at_boundary=['.', '?', '!'],
        max_query_length=300,
        verbose=verbose
    )
    
    answer, snippets, scores = model.answer(question, options=options, k=10)
    print(f"Question: {question}")
    print(f"Options: {json.dumps(options, indent=2)}")
    print(f"Answer: {answer}")
    print(f"Number of snippets retrieved: {len(snippets)}")
    
    # Display retrieved documents
    display_documents(snippets, scores)
    
    return answer, snippets, scores

def test_followup_medrag(question, options, verbose=False):
    """Test MedRAG with follow-up questions ability"""
    print("\n=== Testing MedRAG with Follow-up Questions ===")
    model = MedRAG(
        llm_name="OpenAI/gpt-4.1-nano",
        rag=True,
        follow_up=True,
        retriever_name="MedCPT",
        corpus_name="MedText",
        db_dir="./MedRAG/src/data/corpus",
        verbose=verbose
    )
    
    answer, messages = model.answer(
        question,
        options=options, 
        k=10,
        n_rounds=3,
        n_queries=2
    )
    
    print(f"Question: {question}")
    print(f"Options: {json.dumps(options, indent=2)}")
    
    print("\n=== Conversation with Follow-up Questions ===")
    all_snippets = []
    
    for i, msg in enumerate(messages):
        if isinstance(msg, dict):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            # Extract snippets if they exist in the message
            if role == "system" and msg.get("snippets"):
                all_snippets.extend(msg.get("snippets", []))
        else:
            role = getattr(msg, "role", "unknown")
            content = getattr(msg, "content", "")
            # Extract snippets if they exist in the message
            if role == "system" and hasattr(msg, "snippets"):
                all_snippets.extend(getattr(msg, "snippets", []))
        
        print(f"[{i+1}] {role.upper()}: {content}")
    
    print(f"\nFinal Answer: {answer}")
    print(f"Total conversation turns: {len(messages)}")
    
    # Display all retrieved documents
    display_documents(all_snippets)
    
    return answer, messages

def test_flare_with_followup(question, options, verbose=False):
    """Test MedRAG with both FLARE's look ahead and follow-up questions"""
    print("\n=== Testing FLARE-enhanced MedRAG with Follow-up Questions ===")
    model = MedRAG(
        llm_name="OpenAI/gpt-4.1-nano",
        rag=True,
        follow_up=True,
        retriever_name="MedCPT",
        corpus_name="MedText",
        db_dir="./MedRAG/src/data/corpus",
        enable_flare=True,
        look_ahead_steps=50,
        look_ahead_truncate_at_boundary=['.', '?', '!'],
        max_query_length=300,
        verbose=verbose
    )
    
    answer, messages = model.answer(
        question, 
        options=options,
        k=10,
        n_rounds=3,
        n_queries=2
    )
    
    print(f"Question: {question}")
    print(f"Options: {json.dumps(options, indent=2)}")
    
    print("\n=== Conversation with FLARE and Follow-up Questions ===")
    all_snippets = []
    
    for i, msg in enumerate(messages):
        if isinstance(msg, dict):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            # Extract snippets if they exist in the message
            if role == "system" and msg.get("snippets"):
                all_snippets.extend(msg.get("snippets", []))
        else:
            role = getattr(msg, "role", "unknown")
            content = getattr(msg, "content", "")
            # Extract snippets if they exist in the message
            if role == "system" and hasattr(msg, "snippets"):
                all_snippets.extend(getattr(msg, "snippets", []))
        
        print(f"[{i+1}] {role.upper()}: {content}")
    
    print(f"\nFinal Answer: {answer}")
    print(f"Total conversation turns: {len(messages)}")
    
    # Display all retrieved documents
    display_documents(all_snippets)
    
    # Save the conversation for analysis
    out_file = Path("Test/flare_with_followup_conversation.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w") as f:
        json.dump([m if isinstance(m, dict) else m.model_dump() for m in messages], f, indent=2)
    
    return answer, messages

def compare_results(question, options, verbose=False):
    """Compare results from all four approaches"""
    print("\n=== Running Comparison Tests ===")
    
    # Run all four test cases
    std_answer, std_snippets, std_scores = test_standard_medrag(question, options, verbose)
    flare_answer, flare_snippets, flare_scores = test_flare_medrag(question, options, verbose)
    followup_answer, followup_messages = test_followup_medrag(question, options, verbose)
    combined_answer, combined_messages = test_flare_with_followup(question, options, verbose)
    
    # Save comparison results
    comparison = {
        "question": question,
        "options": options,
        "standard_medrag": {
            "answer": std_answer,
            "num_snippets": len(std_snippets)
        },
        "flare_medrag": {
            "answer": flare_answer,
            "num_snippets": len(flare_snippets)
        },
        "followup_medrag": {
            "answer": followup_answer,
            "conversation_turns": len(followup_messages)
        },
        "flare_with_followup": {
            "answer": combined_answer,
            "conversation_turns": len(combined_messages)
        }
    }
    
    out_file = Path("Test/comparison_results.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w") as f:
        json.dump(comparison, f, indent=2)
    
    print("\n=== Comparison Saved to comparison_results.json ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test FLARE-Med integration with multiple-choice questions")
    parser.add_argument("--mode", type=str, default="comparison",
                      choices=["standard", "flare", "followup", "combined", "comparison"],
                      help="Test mode: standard, flare, followup, combined, or comparison")
    parser.add_argument("--question", type=str, 
                      default="What is the primary mechanism of SGLT2 inhibitors in the treatment of diabetes?",
                      help="Medical question to test")
    parser.add_argument("--verbose", action="store_true", 
                      help="Display all API calls during execution")
    
    args = parser.parse_args()
    
    # Multiple-choice options example
    options = {
        "A": "Increasing insulin secretion from pancreatic beta cells",
        "B": "Blocking carbohydrate absorption in the intestines",
        "C": "Reducing glucose reabsorption in the kidney proximal tubule",
        "D": "Enhancing peripheral insulin sensitivity in muscle tissue"
    }
    
    if args.mode == "standard":
        test_standard_medrag(args.question, options, args.verbose)
    elif args.mode == "flare":
        test_flare_medrag(args.question, options, args.verbose)
    elif args.mode == "followup":
        test_followup_medrag(args.question, options, args.verbose)
    elif args.mode == "combined":
        test_flare_with_followup(args.question, options, args.verbose)
    else:  # comparison
        compare_results(args.question, options, args.verbose) 