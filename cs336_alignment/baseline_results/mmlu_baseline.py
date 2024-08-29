#!/usr/bin/env python3

import json
import os
import re
import time

import pandas as pd

from typing import Any
from vllm import LLM, SamplingParams

from cs336_alignment.baseline import add_system_prompt


def parse_mmlu_response(
    mmlu_example: dict[str, Any],
    model_output: str,
) -> str | None:
    match = re.search(r"The correct answer is ([A-Za-z])\.?", model_output)
    
    if match:
        answer = match.group(1)
        return answer.strip().upper()
    else:
        return None
    

def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, header=None)
    return df


def get_subject(path: str) -> str:
    subject = os.path.splitext(os.path.basename(path))[0].replace("_test", "")
    return subject


def row_to_dict(row: pd.Series, subject: str) -> dict:
    entry = {
        "subject": subject,
        "question": row[0],
        "options": tuple(row[1:5]),
        "answer": row[5]
    }
    return entry


def format_string_prompts(df: dict) -> str:
    subject = df["subject"]
    question = df["question"]
    options = df["options"]

    prompt = f"""
    Answer the following multiple choice question about {subject}. Respond with a single sentence of the form "The correct answer is _", filling the blank with the letter corresponding to the correct answer (i.e., A, B, C or D).
    Question: {question}
    A. {options[0]}
    B. {options[1]}
    C. {options[2]}
    D. {options[3]}
    """
    return prompt


def evaluate_output(generated_answer: str, actual_answer: str) -> bool:
    return generated_answer == actual_answer


def example():
    # Sample prompts.
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    prompts = [add_system_prompt(prompt) for prompt in prompts]

    # Create a sampling params object, stopping generation on newline.
    sampling_params = SamplingParams(
        temperature=0.0, top_p=1.0, max_tokens=1024, stop=["\n", "# Query:"]
    )
    # Create an LLM.
    small_path = "/data/Meta-Llama-3-8B"
    large_path = "/home/shared/Meta-Llama-3-70B-Instruct"
    llm = LLM(model=small_path)
    # Generate texts from the prompts. The output is a list of RequestOutput objects # that contain the prompt, generated text, and other information.
    outputs = llm.generate(prompts, sampling_params)
    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


def main():
    # Paths to files
    mmlu_dir = "../data/mmlu/test"
    mmlu_output = "llama_mmlu_results.jsonl"

    # Create a sampling params object, stopping generation on newline.
    sampling_params = SamplingParams(
        temperature=0.0, top_p=1.0, max_tokens=1024, stop=["# Query:"]
    )
    # Create an LLM.
    model_path = "/data/Meta-Llama-3-8B"
    llama_model = LLM(model=model_path)

    # Tracker for evaluation metrics
    total_examples = 0
    correct_examples = 0
    none_examples = 0

    start_time = time.time()

    for file in os.listdir(mmlu_dir):
        if file.endswith(".csv"):
            file_path = os.path.join(mmlu_dir, file)

            df = load_csv(file_path)
            subject = get_subject(file_path)

            for _, row in df.iterrows():
                mmlu_example = row_to_dict(row, subject)
                prompt = format_string_prompts(mmlu_example)
                system_prompt = add_system_prompt(prompt)

                model_output = llama_model.generate(system_prompt, sampling_params)
                generated_text = model_output[0].outputs[0].text
                generated_answer = parse_mmlu_response(mmlu_example, generated_text)
                if generated_answer is None:
                    none_examples += 1
                    print(model_output)

                actual_answer = mmlu_example["answer"]
                is_correct = evaluate_output(generated_answer, actual_answer)

                result = {
                    "subject": subject,
                    "question": mmlu_example["question"],
                    "options": mmlu_example["options"],
                    "generated_text": generated_text,
                    "actual_answer": actual_answer,
                    "generated_answer": generated_answer,
                    "is_correct": is_correct
                }
                
                with open(mmlu_output, "a") as f:
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
    print(f"Results appended to {mmlu_output}")
    print(f"Accuracy = {accuracy:.4f}")
    print(f"Average Generation Time per Example: {avg_generation_time:.4f} seconds")
    print(f"Throughput: {throughput:.4f} examples/second")


if __name__ == "__main__":
    main()