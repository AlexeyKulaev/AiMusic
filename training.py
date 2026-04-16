"""Training pipeline for the MIDI transformer.

This module contains all logic required to train the model - data loading,
batch sampling, loss estimation, the training loop and checkpoint saving.
It purposefully contains **no** inference or generation code so that it can be
imported without sideeffects.
"""

import os
import argparse
import torch
import torch.nn as nn
from torch.nn import functional as F

from model_def import create_model
import config

# -----------------------------------------------------------------------------
# Hyper‑parameters are now sourced from the central config module.
batch_size = config.BATCH_SIZE
block_size = config.BLOCK_SIZE
max_iters = config.MAX_ITERS
eval_interval = config.EVAL_INTERVAL
learning_rate = config.LEARNING_RATE
device = config.DEVICE
eval_iters = config.EVAL_ITERS
n_embed = config.N_EMBED
n_head = config.N_HEAD
n_layer = config.N_LAYER
dropout = config.DROPOUT


def _load_tokens(data_dir: str, filename: str) -> list[int]:
    """Read a comma-separated token file.
    """
    path = os.path.join(data_dir, filename)
    with open(path, "r") as f:
        return [int(x) for x in f.read().split(",")]


def _prepare_data(project_dir: str):
    data_dir = os.path.join(project_dir, "data")
    train_tokens = _load_tokens(data_dir, "train.txt")
    val_tokens = _load_tokens(data_dir, "validate.txt")
    train_data = torch.tensor(train_tokens, dtype=torch.int64)
    val_data = torch.tensor(val_tokens, dtype=torch.int64)
    vocab_size = max(int(train_data.max()), int(val_data.max())) + 1
    return train_data, val_data, vocab_size, data_dir


def get_batch(split: str, train_data: torch.Tensor, val_data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a batch of block_size tokens.

    split must be either "train" or "val".
    """
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model: nn.Module, train_data: torch.Tensor, val_data: torch.Tensor) -> dict:
    """Compute average loss over eval_iters batches for train and validation."""
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split, train_data, val_data)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def train(project_dir: str, resume_path: str | None = None, max_iters_override: int | None = None, samples: int = 1):
    """Execute the full training pipeline.

    Parameters
    ----------
    project_dir:
        Root directory of the repository - used to locate the data folder and to
        store the final checkpoint.
    resume_path:
        Optional path to a .pth checkpoint to resume training from.
    max_iters_override:
        If supplied, overrides the default max_iters hyper‑parameter.
    samples:
        Number of samples to generate after training (delegated to the inference
        module). Generation is performed by calling :func:`inference.generate_and_save`.
    """
    # ---------------------------------------------------------------------
    # Data preparation
    # ---------------------------------------------------------------------
    train_data, val_data, vocab_size, _ = _prepare_data(project_dir)

    # ---------------------------------------------------------------------
    # Model / optimizer
    # ---------------------------------------------------------------------
    model = create_model(vocab_size, device)
    start_step = 0
    if resume_path:
        checkpoint = torch.load(resume_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint)
        print(f"Resumed training from {resume_path}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # ---------------------------------------------------------------------
    # Training loop
    # ---------------------------------------------------------------------
    total_iters = max_iters_override if max_iters_override is not None else max_iters
    print(f"Training on {device} for {total_iters} iterations…")
    for step in range(start_step, total_iters):
        if step % eval_interval == 0 or step == total_iters - 1:
            losses = estimate_loss(model, train_data, val_data)
            print(f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        xb, yb = get_batch("train", train_data, val_data)
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # ---------------------------------------------------------------------
    # Save checkpoint
    # ---------------------------------------------------------------------
    ckpt_path = os.path.join(project_dir, "model.pth")
    torch.save(model.state_dict(), ckpt_path)
    print(f"Model saved to {ckpt_path}")

    # ---------------------------------------------------------------------
    # Post‑training generation (optional)
    # ---------------------------------------------------------------------
    if samples > 0:
        # Import lazily to avoid circular imports when the module is used only for training.
        from inference import generate_and_save

        for i in range(samples):
            generate_and_save(i + 1, model_path=ckpt_path, project_dir=project_dir)
