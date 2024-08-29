#!/usr/bin/env python3

import gzip
import json
import random

import torch
from torch.utils.data import Dataset, DataLoader


class InstructionTuning(Dataset):
    def __init__(self, tokenizer, dataset_path, seq_length, shuffle):
        self.tokenizer = tokenizer
        self.seq_length = seq_length

        self.data = []
        with open(dataset_path, "r") as file:
            for line in file:
                example = json.loads(line)
                self.data.append(example)
        
        if shuffle:
            random.shuffle(self.data)
        
        with open("/home/c-xiongb/spring2024-assignment5-alignment/cs336_alignment/prompts/alpaca_sft.prompt", "r") as file:
            template = file.read().strip()
        self.token_ids = []
        print(f"tokens = {len(self.token_ids)}")
        for example in self.data:
            prompt = example["prompt"]
            response = example["response"]
            
            input_text = template.format(instruction=prompt, response=response)
            tokens = self.tokenizer.encode(input_text, add_special_tokens=False, padding=False, max_length = 1024, truncation=True)
            self.token_ids.append(self.tokenizer.bos_token_id)
            self.token_ids.extend(tokens)
            self.token_ids.append(self.tokenizer.eos_token_id)

    def __len__(self):
        return (len(self.token_ids)) // (self.seq_length)

    def __getitem__(self, i):
        if i >= len(self):
            raise IndexError("Index out of range")

        start_idx = i * self.seq_length
        end_idx = start_idx + self.seq_length
        
        input_ids = self.token_ids[start_idx:end_idx]
        labels = self.token_ids[start_idx+1:end_idx+1]
        
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long)
        }
    

def iterate_batches(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)