#!/usr/bin/env python3

import json
import os
import re
import time

import pandas as pd

from typing import Any
from vllm import LLM, SamplingParams

from cs336_alignment.baseline import add_system_prompt


def parse_gsm8k_response(
    model_output: str,
) -> str | None:
    numbers = re.findall(r"-?\d+\.?\d*", model_output)
    
    if numbers:
        answer = numbers[-1]
        return answer
    else:
        return None


def extract_number(
    text: str
) -> str | None:
    match = re.search(r"\n#### (\d+(?:\.\d+)?)", text)
    if match:
        return match.group(1)
    else:
        return None


def evaluate_output(generated_answer: str, actual_answer: str) -> bool:
    return generated_answer == actual_answer


def main():
    # Paths to files
    file_path = "../data/gsm8k/test.jsonl"
    gsm8k_output = "llama_gsm8k_results.jsonl"

    # Create a sampling params object, stopping generation on newline.
    sampling_params = SamplingParams(
        temperature=0.0, top_p=1.0, max_tokens=1024, stop=["# Query:"]
    )
    # Create an LLM.
    model_path = "model_checkpoints"
    llama_model = LLM(model=model_path)

    # Tracker for evaluation metrics
    total_examples = 0
    correct_examples = 0
    none_examples = 0

    start_time = time.time()

    with open(file_path, "r") as file:
        for line in file:
            gsm8k_example = json.loads(line)
            question = gsm8k_example["question"]
            answer = gsm8k_example["answer"]
            
            system_prompt = add_system_prompt(question)
            model_output = llama_model.generate(system_prompt, sampling_params)
            generated_text = model_output[0].outputs[0].text
            generated_answer = parse_gsm8k_response(generated_text)
            if generated_answer is None:
                none_examples += 1
                print(generated_text)

            actual_answer = extract_number(answer)
            is_correct = evaluate_output(generated_answer, actual_answer)

            result = {
                "question": question,
                "generated_text": generated_text,
                "actual_answer": actual_answer,
                "generated_answer": generated_answer,
                "is_correct": is_correct
            }

            with open(gsm8k_output, "a") as f:
                json.dump(result, f)
                f.write("\n")

            total_examples += 1
            correct_examples += 1 if is_correct else 0

    end_time = time.time()
    elapsed_time = end_time - start_time

    accuracy = correct_examples / total_examples
    avg_generation_time = elapsed_time / total_examples
    throughput = total_examples / elapsed_time
    
    print(f"Correct examples: {correct_examples}, Total examples: {total_examples}")
    print(f"None examples: {none_examples}")
    print(f"Elapsed time = {elapsed_time}")
    print(f"Results appended to {gsm8k_output}")
    print(f"Accuracy = {accuracy:.4f}")
    print(f"Average Generation Time per Example: {avg_generation_time:.4f} seconds")
    print(f"Throughput: {throughput:.4f} examples/second")


if __name__ == "__main__":
    main()