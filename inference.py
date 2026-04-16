"""Generation module"""

import os
import torch
from torch.nn import functional as F

from model_def import create_model
import config

# The project stores token‑to‑MIDI conversion in midi_decoder.
import midi_decoder


def _load_tokens(data_dir: str, filename: str) -> list[int]:
    """Utility to read comma-separated token files (same format as training)."""
    path = os.path.join(data_dir, filename)
    with open(path, "r") as f:
        return [int(x) for x in f.read().split(",")]


def load_model(model_path: str, project_dir: str) -> torch.nn.Module:
    """Load a checkpoint and return a ready-to-use model."""
    data_dir = os.path.join(project_dir, "data")
    train_tokens = _load_tokens(data_dir, "train.txt")
    val_tokens = _load_tokens(data_dir, "validate.txt")
    train_data = torch.tensor(train_tokens, dtype=torch.int64)
    val_data = torch.tensor(val_tokens, dtype=torch.int64)
    vocab_size = max(int(train_data.max()), int(val_data.max())) + 1

    model = create_model(vocab_size, device=config.DEVICE).to(config.DEVICE)
    state = torch.load(model_path, map_location=config.DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def generate_and_save(sample_idx: int, max_tokens: int = config.MAX_TOKENS, *, model_path: str, project_dir: str):
    """Generate a MIDI sample and write both token and .midi files.

    The output files are placed in <project_dir>/sample and follow the
    naming convention sample_<idx>.txt and sample_<idx>.midi as required
    by the original code base.
    """
    model = load_model(model_path, project_dir)
    device = config.DEVICE
    context = torch.zeros((1, 1), dtype=torch.int64, device=device)
    generated = model.generate(context, max_new_tokens=max_tokens)
    tokens_out = generated[0].tolist()

    sample_dir = os.path.join(project_dir, "sample")
    os.makedirs(sample_dir, exist_ok=True)

    txt_path = os.path.join(sample_dir, f"sample_{sample_idx}.txt")
    with open(txt_path, "w") as f:
        f.write(",".join(str(t) for t in tokens_out))
    print(f"Saved token file {txt_path}")

    midi_path = os.path.join(sample_dir, f"sample_{sample_idx}.midi")
    midi_decoder.tokens_to_midi(tokens_out, midi_path)
    print(f"Saved MIDI file {midi_path}")
