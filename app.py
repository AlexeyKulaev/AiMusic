"""Local web server for AI music generation.

Provides a minimal UI (served from templates) and two API endpoints:

* POST /generate – triggers generate_and_save from inference.py
  using a randomly generated sample identifier and a token count that maps
  to the requested composition length.
* GET /download/<sample_idx> – streams the generated .midi file.

The server runs on localhost only (127.0.0.1) and listens on port
5000 by default.  All generated files are stored in the sample
directory inside the project root.
"""

import os
import random
import string
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory, render_template
import subprocess

from inference import generate_and_save
import config

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "model.pth"
SAMPLE_DIR = PROJECT_ROOT / "sample"


LENGTH_TO_TOKENS = {
    "short": 256,
    "medium": 512,
    "long": 1024,
}


def _random_id(min_len: int = 7, max_len: int = 10) -> str:
    """Return a random alphanumeric identifier."""
    length = random.randint(min_len, max_len)
    return "".join(random.choices(string.ascii_letters + string.digits, k=length))


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route('/')
def index():
    """Render the minimal UI."""
    return render_template('index.html')


@app.route('/generate', methods=['POST'])
def generate():
    """Trigger music generation"""
    data = request.get_json(silent=True) or {}
    length = data.get('length', 'short')
    max_tokens = LENGTH_TO_TOKENS.get(length, LENGTH_TO_TOKENS['short'])

    sample_idx = _random_id()

    generate_and_save(
        sample_idx=sample_idx,
        max_tokens=max_tokens,
        model_path=str(MODEL_PATH),
        project_dir=str(PROJECT_ROOT),
    )

    # Convert the generated MIDI to WAV using TiMidity++ for accurate playback.
    midi_path = SAMPLE_DIR / f"sample_{sample_idx}.midi"
    wav_path = SAMPLE_DIR / f"sample_{sample_idx}.wav"
    try:
        subprocess.run(
            ["timidity", str(midi_path), "-Ow", "-o", str(wav_path)],
            check=True,
        )
    except Exception as e:
        # If conversion fails, log.
        print(f"TiMidity conversion failed for {midi_path}: {e}")

    return jsonify({"sample_idx": sample_idx})


@app.route('/download/<sample_idx>', methods=['GET'])
def download(sample_idx: str):
    """Serve the generated .midi file."""
    filename = f"sample_{sample_idx}.midi"
    return send_from_directory(
        directory=SAMPLE_DIR,
        path=filename,
        as_attachment=True,
        mimetype='audio/midi'
    )


@app.route('/audio/<sample_idx>', methods=['GET'])
def audio(sample_idx: str):
    """Serve the WAV audio rendered by TiMidity++."""
    filename = f"sample_{sample_idx}.wav"
    return send_from_directory(
        directory=SAMPLE_DIR,
        path=filename,
        as_attachment=True,
        mimetype='audio/wav'
    )


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False)
