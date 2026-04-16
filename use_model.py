"""Entry point for training or inference.

This thin wrapper parses command-line arguments and delegates to the dedicated
modules:

* :pymod: training - contains the full training pipeline.
* :pymod:inference - stateless generation utilities.

Importing this file has no side effects; the heavy work only runs when the
module is executed as a script (python train_model.py).
"""

import argparse
import os

from training import train
from inference import generate_and_save


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train or generate MIDI with the transformer model")
    parser.add_argument("--resume", type=str, default=None, help="Path to a checkpoint (.pth) to resume training")
    parser.add_argument("--generate", type=str, default=None, help="Path to a checkpoint (.pth) for generation‑only mode")
    parser.add_argument("--iters", type=int, default=None, help="Override the default number of training iterations")
    parser.add_argument("--samples", type=int, default=1, help="Number of samples to generate after training or in generate‑only mode")
    return parser.parse_args()


def _cli_train(args: argparse.Namespace, project_dir: str) -> None:
    """Run the training pipeline.

    args.resume may be None - in that case training starts from scratch.
    args.iters overrides the default max_iters defined in
    :pymod:training.
    """
    train(
        project_dir=project_dir,
        resume_path=args.resume,
        max_iters_override=args.iters,
        samples=args.samples,
    )


def _cli_generate(args: argparse.Namespace, project_dir: str) -> None:
    """Generate args.samples MIDI files using the provided checkpoint.
    """
    for i in range(args.samples):
        generate_and_save(i + 1, model_path=args.generate, project_dir=project_dir)


if __name__ == "__main__":
    # Resolve the project root – this file lives directly inside the repository.
    project_dir = os.path.abspath(os.path.dirname(__file__))
    args = _parse_args()

    if args.generate:
        _cli_generate(args, project_dir)
    else:
        _cli_train(args, project_dir)

