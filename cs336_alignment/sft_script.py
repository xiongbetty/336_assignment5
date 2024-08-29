#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import pathlib
import sys
from contextlib import nullcontext

import torch
import torch.nn.functional as F
import wandb

from tqdm import tqdm
from cs336_alignment.data_loading import InstructionTuning, iterate_batches

from transformers import AutoModelForCausalLM, AutoTokenizer


logger = logging.getLogger(__name__)


# context_length = 512
# batch_size = 2
# gradient_accumulation_steps = 4
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# train_path = "/home/shared/safety_augmented_ultrachat_200k_single_turn/train.jsonl.gz"
# dev_path = "/home/shared/safety_augmented_ultrachat_200k_single_turn/test.jsonl.gz"
# output_dir = "outputs"

def train(
    train_path,
    dev_path,
    output_dir,
    # vocab_size,
    context_length,
    # d_model,
    # num_layers,
    # num_heads,
    # d_ff,
    # attn_pdrop,
    # residual_pdrop,
    batch_size,
    # train_steps,
    gradient_accumulation_steps,
    # eval_iters,
    eval_interval,
    learning_rate,
    lr_scheduler,
    warmup_ratio,
    weight_decay,
    adam_beta1,
    adam_beta2,
    adam_eps,
    grad_clip,
    device,
    # compile,
    # dtype,
    wandb_project,
):

    # Load the model for fine-tuning.
    tokenizer = AutoTokenizer.from_pretrained("/data/Meta-Llama-3-8B")
    model = AutoModelForCausalLM.from_pretrained(
        "/data/Meta-Llama-3-8B",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    # Load data.
    train_data = InstructionTuning(
        tokenizer=tokenizer,
        dataset_path=train_path,
        seq_length=context_length,
        shuffle=True
    )
    dev_data = InstructionTuning(
        tokenizer=tokenizer,
        dataset_path=dev_path,
        seq_length=context_length,
        shuffle=True
    )

    # Move model to the device.
    model = model.to(device)

    # Set up the AdamW optimizer.
    # We do not apply decay on 1D parameters (e.g., biases and RMSNorms)
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    params_to_decay = [p for _, p in param_dict.items() if p.dim() >= 2]
    params_to_not_decay = [p for _, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {"params": params_to_decay, "weight_decay": weight_decay},
        {"params": params_to_not_decay, "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(
        optim_groups,
        lr=learning_rate,
        betas=(adam_beta1, adam_beta2),
        eps=adam_eps,
    )

    train_data = iterate_batches(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=True
    )
    train_steps = len(train_data)

    # Gradient accumulation.
    for i, train_batch in enumerate(tqdm(train_data)):
        if lr_scheduler.lower() == "cosine":
            lr = get_cosine_lr(
                i,
                max_learning_rate=learning_rate,
                min_learning_rate=learning_rate * 0.1,
                warmup_iters=int(train_steps * warmup_ratio),
                cosine_cycle_iters=train_steps,
            )
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
        else:
            lr = learning_rate

        # Forward pass.
        input_ids = train_batch["input_ids"].to(device)
        labels = train_batch["labels"].to(device)
        logits = model(input_ids).logits

        # Compute language modeling loss.
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
        loss = loss / gradient_accumulation_steps
        
        # Backward pass.
        loss.backward()
        if (i + 1) % gradient_accumulation_steps == 0:
            # Gradient clipping.
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            # Update weights every `gradient_accumulation_steps` batches. 
            torch.cuda.empty_cache()
            optimizer.step()
            # Zero gradients every `gradient_accumulation_steps` batches. 
            optimizer.zero_grad()
        
            # Logging.
            loss_float = loss.item() * gradient_accumulation_steps
            logger.info(f"Train step {i}, Loss: {loss_float}")
            if wandb_project:
                wandb.log({"train_loss": loss_float, "lr": lr}, step=i)

        if i != 0 and i % eval_interval == 0:
            dev_loss = estimate_dev_loss(
                model=model,
                dev_dataset=dev_data,
                batch_size=batch_size,
                device=device,
            )
            logger.info(f"Estimated validation loss: {dev_loss}")
            if wandb_project:
                wandb.log({"eval_loss": dev_loss}, step=i)

            if i % (eval_interval * 10) == 0:
                logger.info(f"Saving model weights to {output_dir}")
                model.save_pretrained(save_directory=output_dir)
                tokenizer.save_pretrained(save_directory=output_dir)
       
    # Calculate final dev loss.
    dev_loss = estimate_dev_loss(
        model=model,
        dev_dataset=dev_data,
        batch_size=batch_size,
        device=device,
    )
    logger.info(f"Final estimated validation loss: {dev_loss}")
    if wandb_project:
        wandb.log({"eval_loss": dev_loss}, step=train_steps)
    # Save the model weights.
    logger.info(f"Saving model weights to {output_dir}")
    model.save_pretrained(save_directory=output_dir)
    tokenizer.save_pretrained(save_directory=output_dir)


@torch.no_grad()
def estimate_dev_loss(
    model,
    dev_dataset,
    batch_size,
    device,
):
    model.eval()
    torch.cuda.empty_cache()
    dev_data_loader = iterate_batches(
        dataset=dev_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    dev_steps = len(dev_data_loader) // 10 + 1
    losses = torch.zeros(dev_steps)
    for k, dev_batch in enumerate(tqdm(dev_data_loader)):
        if k != 0 and k % 10 == 0:
            input_ids = dev_batch["input_ids"].to(device)
            labels = dev_batch["labels"].to(device)
            logits = model(input_ids).logits
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
            losses[k // 10] = loss.item()
    model.train()
    return losses.mean()


def get_cosine_lr(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """Cosine with warmup learning rate scheduler."""
    # First, we linearly warmup for warmup_iters steps.
    if it < warmup_iters:
        return max_learning_rate * it / warmup_iters
    # Then, if it > cosine_cycle_iters, we return min learning rate.
    if it > cosine_cycle_iters:
        return min_learning_rate
    # Else, we use cosine decay down to min learning rate.
    decay_ratio = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_learning_rate + coeff * (max_learning_rate - min_learning_rate)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(module)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--train-path",
        required=True,
        help="Path to input IDs to train with.",
    )
    parser.add_argument(
        "--dev-path",
        required=True,
        help="Path to input IDs to use for measuring validation performance.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Path to folder to write model configuration and trained model checkpoint",
    )
    # parser.add_argument(
    #     "--vocab-size",
    #     type=int,
    #     required=True,
    #     help="Path to file with mapping from token to BPE index",
    # )
    parser.add_argument(
        "--context-length",
        type=int,
        required=True,
        help="Context length to use when training language model",
    )
    # parser.add_argument(
    #     "--d-model",
    #     type=int,
    #     required=True,
    #     help="The dimensionality of the model embeddings and sublayer outputs.",
    # )
    # parser.add_argument(
    #     "--num-layers",
    #     type=int,
    #     required=True,
    #     help=(
    #         "The number of Transformer layers to use. "
    #         "`d_model` must be evenly divisible by `num_heads`."
    #     ),
    # )
    # parser.add_argument(
    #     "--num-heads",
    #     type=int,
    #     required=True,
    #     help=(
    #         "Number of heads to use in multi-headed attention. "
    #         "`d_model` must be evenly divisible by `num_heads`."
    #     ),
    # )
    # parser.add_argument(
    #     "--d-ff",
    #     type=int,
    #     required=True,
    #     help=("Dimensionality of the feed-forward inner layer (section 3.3)."),
    # )
    # parser.add_argument(
    #     "--attn-pdrop",
    #     type=float,
    #     help=("If given, drop-out the attention probabilities with this rate."),
    # )
    # parser.add_argument(
    #     "--residual-pdrop",
    #     type=float,
    #     help=(
    #         "If given, apply dropout to output of each sub-layer, before it is "
    #         "added to the sub-layer input and normalized (section 5.4)."
    #     ),
    # )
    parser.add_argument(
        "--batch-size",
        type=int,
        required=True,
        help=("Batch size to use during training."),
    )
    # parser.add_argument(
    #     "--train-steps",
    #     type=int,
    #     required=True,
    #     help="Number of training steps to perform",
    # )
    parser.add_argument(
        "--gradient-accumulation-steps",
        default=1,
        type=int,
        help=(
            "Number of forward+backward passes to do with given "
            "batch size for each single train step"
        ),
    )
    # parser.add_argument(
    #     "--eval-iters",
    #     type=int,
    #     default=200,
    #     help="Number of evaluation batches to use for calculating validation loss",
    # )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=2000,
        help="Measure validation loss every `eval-interval` trainig steps",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        required=True,
        help=("Learning rate to use during training."),
    )
    parser.add_argument(
        "--lr-scheduler",
        type=str,
        choices=["constant", "cosine"],
        default="cosine",
        help=("Learning rate scheduler to use during training."),
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.01,
        help=("Ratio of total steps to use for LR warmup"),
    )
    parser.add_argument(
        "--weight-decay", type=float, default=1e-1, help="AdamW weight decay"
    )
    parser.add_argument(
        "--adam-beta1",
        type=float,
        default=0.9,
        help=("Value to use for Adam beta_1"),
    )
    parser.add_argument(
        "--adam-beta2",
        type=float,
        default=0.98,
        help=("Value to use for Adam beta_2"),
    )
    parser.add_argument(
        "--adam-eps",
        type=float,
        default=1e-9,
        help=("Value to use for Adam epsilon"),
    )
    parser.add_argument(
        "--grad-clip",
        type=float,
        help=("If set, clip gradient norms to this value"),
    )
    parser.add_argument(
        "--device",
        required=True,
        help="Device to use for training (e.g., 'cpu', 'cuda', 'cuda:0', etc.)",
    )
    # parser.add_argument(
    #     "--compile",
    #     action="store_true",
    #     help="If true, compile the model with torch.compile",
    # )
    # parser.add_argument(
    #     "--dtype",
    #     type=str,
    #     choices=["float32", "float16", "bfloat16"],
    #     default="bfloat16"
    #     if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    #     else "float16",
    #     help="dtype to use when training",
    # )
    parser.add_argument(
        "--wandb-project",
        type=str,
        help="If set, log results to the specified wandb project",
    )
    args = parser.parse_args()

    logger.info("running %s", " ".join(sys.argv))

    # Make the directory for output if it doesn't already exist
    if os.path.exists(os.path.join(args.output_dir, "model.pt")):
        raise ValueError(
            f"output directory {args.output_dir} already exists and contains model.pt"
        )
    else:
        os.makedirs(args.output_dir, exist_ok=True)

    if args.wandb_project:
        wandb.login()
        wandb.init(
            # Set the project where this run will be logged
            project=args.wandb_project,
            config=vars(args),
            name=pathlib.Path(args.output_dir).name,
        )

    train(
        args.train_path,
        args.dev_path,
        args.output_dir,
        # args.vocab_size,
        args.context_length,
        # args.d_model,
        # args.num_layers,
        # args.num_heads,
        # args.d_ff,
        # args.attn_pdrop,
        # args.residual_pdrop,
        args.batch_size,
        # args.train_steps,
        args.gradient_accumulation_steps,
        # args.eval_iters,
        args.eval_interval,
        args.learning_rate,
        args.lr_scheduler,
        args.warmup_ratio,
        args.weight_decay,
        args.adam_beta1,
        args.adam_beta2,
        args.adam_eps,
        args.grad_clip,
        args.device,
        # args.compile,
        # args.dtype,
        args.wandb_project,
    )
    logger.info("finished running %s", sys.argv[0])