# -*- coding: utf-8 -*-
"""
MIDI Decoder - Convert list of numbers back to .midi file using miditok (REMI)
"""

from miditok import REMI


def tokens_to_midi(token_ids: list[int], output_path: str) -> None:
    """
    Decode a flat list of token integers back into a MIDI file.

    Args:
        token_ids: Flat list of integer tokens
        output_path: Path to save the output MIDI file
    """
    tokenizer = REMI()
    # decode expects 2D: [[track_tokens, ...]]. Returns a Score object.
    score = tokenizer.decode([token_ids])
    score.dump_midi(output_path)


def tokens_to_midi_from_file(input_path: str, output_path: str) -> None:
    """
    Load token integers from a text file and decode to MIDI.

    Args:
        input_path: Path to comma-separated token file
        output_path: Path to save the output MIDI file
    """
    with open(input_path, "r") as f:
        tokens = [int(x) for x in f.read().split(",")]
    tokens_to_midi(tokens, output_path)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python midi_decoder.py <tokens.txt> <output.midi>")
        sys.exit(1)
    tokens_to_midi_from_file(sys.argv[1], sys.argv[2])
    print(f"Decoded {sys.argv[1]} -> {sys.argv[2]}")
