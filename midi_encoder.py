"""
MIDI Encoder - Convert .midi files to list of numbers using miditok (REMI)
"""

from miditok import REMI


def midi_to_tokens(midi_path: str) -> list[int]:
    """Encode a MIDI file to a flat list of integer tokens using REMI encoding."""
    tokenizer = REMI()
    sequences = tokenizer.encode(midi_path)
    # Flatten all track ids into one list
    flat = []
    for seq in sequences:
        flat.extend(seq.ids)
    return flat


def tokens_to_file(token_ids: list[int], output_path: str) -> None:
    """Save a flat list of token integers to a comma-separated text file."""
    with open(output_path, "w") as f:
        f.write(",".join(str(x) for x in token_ids))


def load_tokens(input_path: str) -> list[int]:
    """Load a flat list of token integers from a comma-separated text file."""
    with open(input_path, "r") as f:
        return [int(x) for x in f.read().split(",")]


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python midi_encoder.py <midi_file> [output_tokens.txt]")
        sys.exit(1)
    midi_file = sys.argv[1]
    output = sys.argv[2] if len(sys.argv) > 2 else midi_file.replace(".midi", ".txt").replace(".mid", ".txt")
    tokens = midi_to_tokens(midi_file)
    print(f"Encoded {midi_file} -> {len(tokens)} tokens")
    tokens_to_file(tokens, output)
    print(f"Saved to {output}")
