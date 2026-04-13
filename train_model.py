# -*- coding: utf-8 -*-
"""
Train a transformer model on MIDI token sequences.

Model architecture copied from model.py (karpathy-style decoder-only transformer).
Data is loaded from train.txt and validate.txt (comma-separated integer tokens).

Usage:
    python train_model.py                          # Train from scratch
    python train_model.py --resume model.pth       # Resume from checkpoint
    python train_model.py --generate model.pth     # Generate MIDI from checkpoint only
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
import os
import argparse

# -----------------------------------------------------------------------------
# Hyperparameters
# -----------------------------------------------------------------------------
batch_size = 16
block_size = 128
max_iters = 500
eval_interval = 300
learning_rate = 1e-3
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
n_embed = 384       # sz_token = 64 * 6
n_head = 6
n_layer = 3
dropout = 0.2

# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------
project_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(project_dir, "data")

def load_tokens(filename):
    path = os.path.join(data_dir, filename)
    print(f"Loading {filename}...")
    with open(path, "r") as f:
        return [int(x) for x in f.read().split(",")]

train_tokens = load_tokens("train.txt")
val_tokens = load_tokens("validate.txt")

train_data = torch.tensor(train_tokens, dtype=torch.int64)
val_data = torch.tensor(val_tokens, dtype=torch.int64)

# Determine vocab size from data
vocab_size = max(int(train_data.max()), int(val_data.max())) + 1
print(f"Vocab size: {vocab_size}")
print(f"Train tokens: {len(train_data)}, Val tokens: {len(val_data)}")

torch.manual_seed(5)

# -----------------------------------------------------------------------------
# Data sampling
# -----------------------------------------------------------------------------
def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1: i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# print(get_batch('train'))

# -----------------------------------------------------------------------------
# Loss estimation
# -----------------------------------------------------------------------------
@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# -----------------------------------------------------------------------------
# Model definition
# -----------------------------------------------------------------------------
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * C ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        head_size = n_embed // num_heads
        self.sa = MultiHeadAttention(num_heads, head_size)
        self.ffwd = FeedForward()
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[TransformerBlock(n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size, bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Train or generate with MIDI transformer model")
parser.add_argument("--resume", type=str, default=None, help="Path to model.pth to resume training")
parser.add_argument("--generate", type=str, default=None, help="Path to model.pth to generate only (no training)")
parser.add_argument("--iters", type=int, default=None, help="Override max_iters")
args = parser.parse_args()

if args.iters is not None:
    max_iters = args.iters

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
model = BigramLanguageModel(vocab_size).to(device)

# Load weights if resuming or generating
start_step = 0
if args.generate:
    model.load_state_dict(torch.load(args.generate, map_location=device, weights_only=True))
    print(f"Loaded model from {args.generate}")
elif args.resume:
    checkpoint = torch.load(args.resume, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)
    print(f"Resumed training from {args.resume}")

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# -----------------------------------------------------------------------------
# Generate-only mode
# -----------------------------------------------------------------------------
if args.generate:
    model.eval()
    print("Generating 5000 tokens...")
    context = torch.zeros((1, 1), dtype=torch.int64, device=device)
    generated = model.generate(context, max_new_tokens=5000)
    tokens_out = generated[0].tolist()

    out_path = os.path.join(project_dir, "generated.txt")
    with open(out_path, "w") as f:
        f.write(",".join(str(t) for t in tokens_out))
    print(f"Saved generated.txt ({len(tokens_out)} tokens)")

    # Also decode to MIDI
    import midi_decoder
    midi_decoder.tokens_to_midi(tokens_out, os.path.join(project_dir, "generated.midi"))
    print("Saved generated.midi")
    exit()

# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
print(f"Training on {device} for {max_iters} iterations...")
print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

for step in range(start_step, max_iters):
    if step % eval_interval == 0 or step == max_iters - 1:
        losses = estimate_loss(model)
        print(f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch("train")
    logits, loss = model(xb, yb)
    model.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# -----------------------------------------------------------------------------
# Save model
# -----------------------------------------------------------------------------
torch.save(model.state_dict(), os.path.join(project_dir, "model.pth"))
print(f"Model saved to model.pth")

# -----------------------------------------------------------------------------
# Generate sample
# -----------------------------------------------------------------------------
model.eval()
context = torch.zeros((1, 1), dtype=torch.int64, device=device)
generated = model.generate(context, max_new_tokens=5000)
tokens_out = generated[0].tolist()

# Save as comma-separated
out_path = os.path.join(project_dir, "generated.txt")
with open(out_path, "w") as f:
    f.write(",".join(str(t) for t in tokens_out))
print(f"Generated 5000 tokens saved to {out_path}")

# Decode to MIDI
import midi_decoder
midi_decoder.tokens_to_midi(tokens_out, os.path.join(project_dir, "generated.midi"))
print("Saved generated.midi")
