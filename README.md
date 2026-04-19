# AiMusic

**AiMusic** is a full-stack AI music generation system built around a **self-designed and self-trained decoder-only transformer model**.  
The system generates original musical compositions in MIDI format using autoregressive sequence modeling and converts them into playable audio using TiMidity++.

This project was developed end-to-end, including:
- custom transformer architecture design
- training pipeline implementation
- MIDI tokenization / detokenization system
- web-based inference and playback interface

The model was trained on the **MAESTRO dataset**, a high-quality collection of aligned piano performances, used for symbolic music learning and sequence modeling.

---

## Key Idea

The core of AiMusic is a **decoder-only Transformer model trained from scratch on symbolic music data**.

Unlike rule-based or pre-trained music systems, this model learns musical structure directly from data by predicting the next token in a MIDI sequence.

The model operates on a **tokenized representation of MIDI files (REMI format)** and generates coherent musical sequences autoregressively.

---

## Model Overview (Custom Sequence Modeling Architecture)

The neural network is a **custom-built decoder-only Transformer**, implemented from first principles and adapted specifically for symbolic music generation.

### Core Design Principles

- Causal self-attention (autoregressive modeling)
- Multi-head attention mechanism
- Multi-layer Transformer blocks
- Residual connections for stable gradient flow
- Layer normalization
- Feed-forward MLP blocks (position-wise networks)
- Dropout regularization for generalization

---

### Transformer Block Structure

Each block consists of:

1. **Masked Multi-Head Self-Attention**
   - Ensures causal generation (no future token leakage)
   - Multiple attention heads operate in parallel
   - Each head learns different musical dependencies

2. **Feed-Forward Network (MLP)**
   - Expands representation space (typically 4× embedding size)
   - Non-linear transformation of features

3. **Residual Connections (Critical Design Choice)**
   - x = x + Attention(LN(x))
   - x = x + MLP(LN(x))

   This design ensures stable gradient propagation during deep training, preventing vanishing gradients as model depth increases.

4. **Layer Normalization**
   - Stabilizes training dynamics
   - Improves convergence speed

## Models Directory

The project now includes a **`Models/`** directory that stores trained model checkpoints, training logs, and example generated music samples. This directory is organized by model version:

- `model0/` – Baseline model trained for 5,000 iterations with a block size of 128.
- `model1/` – Best model trained for 50,000 iterations with a larger block size (256) and deeper training.
- `model2/` – Latest model trained for 20,000 iterations using a higher learning rate schedule and larger embedding size.

Each sub‑folder contains:

- `*.pth` – PyTorch checkpoint of the trained model.
- `logs*.txt` – Training hyper‑parameters and loss curves.
- `*_sample_*.txt` – Tokenized representations of generated music samples (one per line).
- `*_sample_*.midi` – Generated music sample files (one sample per file).

### Purpose

The `Models/` directory allows you to:

1. **Re‑use** a pre‑trained model without retraining from scratch.
2. **Compare** performance across different model configurations.
3. **Inspect** generated music samples directly from the token files.

## Using Your Own Model

To run this project with another trained music generation model, follow these steps:

1. Select a model
- Go to the `Models` folder in the repository
- Choose a compatible model file (`.pth` format)

2. Move the model
- Copy the selected model file into the root directory of the project
- Rename the file to: `model.pth`

⚠️ Important: Model requirements
- The application expects the model file to be named exactly `model.pth`
- Make sure the model architecture matches the backend inference code
- Incompatible models may cause errors or fail during generation

3. Update model configuration
- Inside the selected model folder locate the `logs.txt` file
- At the top of this file, you will find the hyperparameters used during training
- Open `config.py` inside the same model folder
- Update the hyperparameters in `config.py` to match the values from `logs.txt`

This step is required to ensure the model runs correctly and produces consistent results.

### Model Variants

#### `model0`

- **Training**: 5,000 iterations, block size 128, learning rate 1e‑3.
- **Loss progression** (from [`models/model0/logs0.txt`](models/model0/logs0.txt:1)) shows a steady decrease from ~5.5 to ~2.12 validation loss.
- **Checkpoint**: [`models/model0/model0.pth`](models/model0/model0.pth:1).

#### `model1`

- **Training**: 50,000 iterations, block size 256, learning rate 1e‑3.
- **Loss progression** (from [`models/model1/logs1.txt`](models/model1/logs1.txt:1)) improves from ~5.63 to ~1.80 validation loss, indicating better convergence.
- **Checkpoint**: [`models/model1/model_1.pth`](models/model1/model_1.pth:1).

#### `model2`

- **Training**: 20,000 iterations, block size 256, learning rate 4e‑4, larger embedding (`N_EMBED = 256 * 6`).
- **Loss progression** (from [`models/model2/logs2.txt`](models/model2/logs2.txt:1)) reaches ~1.80 validation loss and ~1.62 train loss because of overfitting
- **Checkpoint**: (not provided in repository – you can generate one by running `training.py` with the `model2` configuration).

### Training Observations

- All models use the same architecture (6 layers, 6 heads) but differ in block size and embedding dimensions.
- Increasing `BLOCK_SIZE` and `N_EMBED` (as in `model2`) leads to faster loss reduction.
- `model1` benefits from a much longer training schedule, achieving lower loss than `model0` despite the same embedding size.

### Recommended Model

Based on validation loss and training duration, **`model1`** is the recommended version for most experiments, offering the best trade‑off between model capacity and convergence speed.

## Results / Experiments (updated)

The training curves for each model are summarized below (extracted from the log files):

- **model0** – final val loss ≈ 2.17 after 5 k iters.
- **model1** – final val loss ≈ 1.80 after 50 k iters.
- **model2** – final val loss ≈ 1.80 after 20 k iters.

These results demonstrate that larger embeddings and longer training improve generation quality, as reflected in lower validation loss and more coherent sample outputs.

---

### Why This Architecture Works

The model captures:

- long-range musical structure (rhythm, repetition)
- local note dependencies
- temporal consistency

The combination of attention + residual learning enables deep scaling without loss of gradient signal, allowing stable training of a large autoregressive model.

---

## Data Pipeline

### Dataset

The model is trained on the **MAESTRO dataset**, which contains high-quality aligned piano performances.  
It is widely used for symbolic music generation research and provides rich temporal and expressive musical structure.

### MIDI Tokenization

- `midi_encoder.py` converts MIDI files into token sequences using the REMI representation (via miditok).
- Each musical event (note, duration, velocity, timing) is encoded as discrete tokens.

### MIDI Reconstruction

- `midi_decoder.py` converts generated tokens back into valid MIDI files.
- Ensures consistency between encoding and decoding formats.

---

## Training Pipeline

The model is trained using a fully custom training loop:

- Optimizer: AdamW
- Loss: cross-entropy next-token prediction
- Batch sampling: random sequence chunks from dataset
- Evaluation: periodic validation loss monitoring
- Checkpointing: model saved as `model.pth`

Training data consists of tokenized MIDI sequences stored in plain text format derived from the MAESTRO dataset.

---

## Inference System

The inference engine:

1. Loads trained model weights
2. Generates token sequences autoregressively
3. Decodes tokens into MIDI files
4. Optionally integrates with web API for real-time generation

All outputs are reproducible via deterministic sample IDs.

---

## Web Application

AiMusic includes a lightweight Flask-based web interface.

### Features

- Generate new musical compositions
- Play generated music directly in browser
- Download MIDI files

### Audio System

Generated MIDI files are converted into WAV audio using TiMidity++, ensuring consistent playback across systems.

---

## Tech Stack

- Python 3.10+
- PyTorch
- Flask
- miditok (REMI encoding)
- MAESTRO dataset
- TiMidity++ (MIDI → WAV rendering)
- HTML / CSS / JavaScript

---

## Quick Start

```bash
git clone https://github.com/yourusername/AiMusic.git
cd AiMusic

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt

python app.py

Then open:

http://127.0.0.1:5000
