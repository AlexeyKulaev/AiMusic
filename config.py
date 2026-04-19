"""Centralized configuration for the AiMusic project.

All hyperparameters, model architecture settings, device selection and data
paths are defined here.  Import this module wherever a constant is required to
avoid hardcoded values scattered across the codebase.
"""

import os
import torch

# -----------------------------------------------------------------------------
# Device configuration
# -----------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------------------------------------------------------
# Training hyperparameters
# -----------------------------------------------------------------------------
BATCH_SIZE = 64
BLOCK_SIZE = 256
MAX_ITERS = 500
EVAL_INTERVAL = 300
LEARNING_RATE = 1e-3
EVAL_ITERS = 100
DROPOUT = 0.2

# -----------------------------------------------------------------------------
# Model architecture parameters
# -----------------------------------------------------------------------------
N_EMBED = 64 * 6
N_HEAD = 6
N_LAYER = 6

# -----------------------------------------------------------------------------
# Data configuration (paths are resolved relative to this file's directory)
# -----------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
SAMPLE_DIR = os.path.join(PROJECT_ROOT, "sample")

# -----------------------------------------------------------------------------
# Generation defaults
# -----------------------------------------------------------------------------
MAX_TOKENS = 5000

