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
    file_path = "../data/alpaca_eval/alpaca_eval.jsonl"
    alpaca_output = "llama_alpaca_results.jsonl"

    # Create a sampling params object, stopping generation on newline.
    sampling_params = SamplingParams(
        temperature=0.0, top_p=1.0, max_tokens=1024, stop=["# Query:"]
    )
    # Create an LLM.
    model_path = "model_checkpoints"
    llama_model = LLM(model=model_path)

    total_examples = 0
    start_time = time.time()

    with open(file_path, "r") as file:
        for line in file:
            alpaca_example = json.loads(line)
            instruction = alpaca_example["instruction"]
            generator = "llama-3-8b-base"
            dataset = alpaca_example["dataset"]
            
            system_prompt = add_system_prompt(instruction)
            model_output = llama_model.generate(system_prompt, sampling_params)
            generated_text = model_output[0].outputs[0].text
            generated_answer = generated_text.strip(" \n\t`")

            result = {
                "instruction": instruction,
                "output": generated_answer,
                "generator": generator,
                "dataset": dataset
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