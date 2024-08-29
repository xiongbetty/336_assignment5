#!/usr/bin/env python3

import re

from typing import Any
from vllm import LLM, SamplingParams


def add_system_prompt(instruction: str) -> str:
    prompt = f"""
    # Instruction
    Below is a list of conversations between a human and an AI assistant (you).
    Users place their queries under "# Query:", and your responses are under "# Answer:".
    You are a helpful, respectful, and honest assistant.
    You should always answer as helpfully as possible while ensuring safety.
    Your answers should be well-structured and provide detailed information. They should also have an engaging tone.
    Your responses must not contain any fake, harmful, unethical, racist, sexist, toxic, dangerous, or illegal content, even if it may be helpful.
    Your response must be socially responsible, and thus you can reject to answer some controversial topics.
    # Query:
    ```{instruction}```
    # Answer: 
    ```
    """
    return prompt


def example():
    # Sample prompts.
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    # Create a sampling params object, stopping generation on newline.
    sampling_params = SamplingParams(
        temperature=0.0, top_p=1.0, max_tokens=1024, stop=["\n"]
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
