#!/usr/bin/env python3

import json
import os
import re
import time

import pandas as pd

from typing import Any
from vllm import LLM, SamplingParams

from cs336_alignment.baseline import add_system_prompt


def main():
    # Paths to files
    file_path = "../data/simple_safety_tests/simple_safety_tests.csv"
    alpaca_output = "llama_sst_results.jsonl"

    # Create a sampling params object, stopping generation on newline.
    sampling_params = SamplingParams(
        temperature=0.0, top_p=1.0, max_tokens=1024, stop=["# Query:"]
    )
    # Create an LLM.
    model_path = "/data/Meta-Llama-3-8B"
    llama_model = LLM(model=model_path)

    total_examples = 0
    start_time = time.time()

    df = pd.read_csv(file_path)
    rows_dict = df.to_dict(orient="records")

    for row_dict in rows_dict:
        prompts_final = row_dict["prompts_final"]
        
        system_prompt = add_system_prompt(prompts_final)
        model_output = llama_model.generate(system_prompt, sampling_params)
        generated_text = model_output[0].outputs[0].text
        generated_answer = generated_text.strip(" \n\t`")

        result = {
            "prompts_final": prompts_final,
            "output": generated_answer
        }

        with open(alpaca_output, "a") as f:
            json.dump(result, f)
            f.write("\n")

        total_examples += 1

    end_time = time.time()
    elapsed_time = end_time - start_time

    avg_generation_time = elapsed_time / total_examples
    throughput = total_examples / elapsed_time
    
    print(f"Total examples: {total_examples}")
    print(f"Elapsed time = {elapsed_time}")
    print(f"Results appended to {alpaca_output}")
    print(f"Average Generation Time per Example: {avg_generation_time:.4f} seconds")
    print(f"Throughput: {throughput:.4f} examples/second")


if __name__ == "__main__":
    main()