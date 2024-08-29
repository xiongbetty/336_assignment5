#!/usr/bin/env python3

import gzip
import json
import os
import torch

import torch.nn.functional as F
from transformers import PreTrainedTokenizerBase


def load_hh(folder_path):
    output_file = "hh_dataset.jsonl"

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".jsonl.gz"):
            file_path = os.path.join(folder_path, file_name)
            with gzip.open(file_path, "rt", encoding="utf-8") as file:
                for line in file:
                    data = json.loads(line)
                    chosen_conversation = data["chosen"].strip("\n\n").split("\n\n")
                    rejected_conversation = data["rejected"].strip("\n\n").split("\n\n")

                    if len(chosen_conversation) == 2 and len(rejected_conversation) == 2:
                        instruction = chosen_conversation[0].replace("Human: ", "")
                        chosen_response = chosen_conversation[1].replace("Assistant", "")
                        rejected_response = rejected_conversation[1].replace("Assistant", "")

                        example = {
                            "file": file_name,
                            "instruction": instruction,
                            "chosen": chosen_response,
                            "rejected": rejected_response
                        }

                        with open(output_file, "a", encoding="utf-8") as out_file:
                            out_file.write(json.dumps(example) + "\n")

    print(f"Processing completed. Processed examples appended to '{output_file}'.")


def get_logps(logits, labels):
    labels = labels[..., 1:].clone()
    logits = logits[..., :-1, :]
    per_token_logps = torch.gather(logits.log_softmax(-1), dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    return per_token_logps.sum(dim=-1)


def dpo_loss(
    lm: torch.nn.Module,
    lm_ref: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    beta: float,
    prompt: str,
    response_chosen: str,
    response_rejected: str,
):
    with open("/home/c-xiongb/spring2024-assignment5-alignment/cs336_alignment/prompts/alpaca_sft.prompt", "r") as file:
        template = file.read()

    # Tokenize.
    input_chosen = template.format(instruction=prompt, response=response_chosen) + tokenizer.eos_token
    tokens_chosen = tokenizer.encode(input_chosen, add_special_tokens=False)
    token_ids_chosen = torch.tensor(tokens_chosen, dtype=torch.long)
    
    input_rejected = template.format(instruction=prompt, response=response_rejected) + tokenizer.eos_token
    tokens_rejected = tokenizer.encode(input_rejected, add_special_tokens=False)
    tokens_id_rejected = torch.tensor(tokens_rejected, dtype=torch.long)

    # Get the log-probabilities.
    with torch.no_grad():
        token_ids_chosen = token_ids_chosen.to(lm.device)
        tokens_id_rejected = tokens_id_rejected.to(lm.device)
        logp_chosen = get_logps(lm(token_ids_chosen).logits, token_ids_chosen)
        logp_rejected = get_logps(lm(tokens_id_rejected).logits, tokens_id_rejected)
        difference = logp_chosen - logp_rejected

        token_ids_chosen_ref = token_ids_chosen.to(lm_ref.device)
        tokens_id_rejected_ref = tokens_id_rejected.to(lm_ref.device)
        logp_chosen_ref = get_logps(lm(token_ids_chosen_ref).logits, token_ids_chosen_ref)
        logp_rejected_ref = get_logps(lm(tokens_id_rejected_ref).logits, tokens_id_rejected_ref)
        difference_ref = logp_chosen_ref - logp_rejected_ref

    dpo_loss = -F.logsigmoid(beta * (difference - difference_ref))
    return dpo_loss


if __name__ == "__main__":
    folder_path = "/home/shared/hh"
    load_hh(folder_path)