# AiMusic

A transformer-based model for generating MIDI music. Trained on the [MAESTRO dataset](https://magenta.tensorflow.org/datasets/maestro) using REMI tokenization via [miditok](https://github.com/Natooz/MidiTok).

## Project Structure

| File | Description |
|---|---|
| `midi_encoder.py` | Encode `.midi` files into integer token sequences (REMI) |
| `midi_decoder.py` | Decode token sequences back into `.midi` files |
| `train_model.py` | Train the transformer model and generate samples |
| `model.py` | Original Colab notebook reference implementation |
| `model.pth` | Trained model checkpoint |

## Requirements

- Python 3.10+
- PyTorch 2.0+
- miditok 3.0+

```bash
pip install -r requirements.txt
```

## Usage

### Encode MIDI to tokens

```bash
python midi_encoder.py input.midi output.txt
```

### Decode tokens back to MIDI

```bash
python midi_decoder.py tokens.txt output.midi
```

### Train the model

```bash
python train_model.py                              # Train from scratch
python train_model.py --resume model.pth           # Resume from checkpoint
python train_model.py --iters 10000                # Custom training steps
```

### Generate music

```bash
python train_model.py --generate model.pth          # Generate 1 sample
python train_model.py --generate model.pth --samples 5  # Generate 5 samples
```

Generated files are saved to the `sample/` folder as `generated_N.txt` and `generated_N.midi`.

## Model Architecture

Decoder-only transformer (karpathy-style):
- Embedding dimension: 384
- Heads: 6
- Layers: 3
- Block size: 128
- Dropout: 0.2

## Dataset

Training data is prepared from the MAESTRO v3.0.0 dataset. MIDI files are tokenized using REMI encoding and split into `data/train.txt` and `data/validate.txt` (comma-separated integer tokens).

## License

MIT — see [LICENSE](LICENSE)
