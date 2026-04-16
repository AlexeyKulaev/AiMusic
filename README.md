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

---

### Why This Architecture Works

The model captures:

- long-range musical structure (rhythm, repetition)
- harmonic progression
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