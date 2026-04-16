"""Local web server for AI music generation.

Provides a minimal UI (served from ``templates``) and two API endpoints:

* ``POST /generate`` – triggers ``generate_and_save`` from ``inference.py``
  using a randomly generated sample identifier and a token count that maps
  to the requested composition length.
* ``GET /download/<sample_idx>`` – streams the generated ``.midi`` file.

The server runs on ``localhost`` only (``127.0.0.1``) and listens on port
``5000`` by default.  All generated files are stored in the ``sample``
directory inside the project root.
"""

import os
import random
import string
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory, render_template
import subprocess

# Import the generation helper – do **not** modify ``inference.py``.
from inference import generate_and_save
import config

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "model.pth"
SAMPLE_DIR = PROJECT_ROOT / "sample"

# Mapping from UI length selector to token count.  These values are chosen to
# be reasonable defaults; they can be tweaked without touching the core
# generation logic.
LENGTH_TO_TOKENS = {
    "short": 256,
    "medium": 512,
    "long": 1024,
}


def _random_id(min_len: int = 7, max_len: int = 10) -> str:
    """Return a random alphanumeric identifier.

    The identifier is used as ``sample_idx`` for ``generate_and_save``.  The
    function returns a *string* even though the original signature annotates
    ``sample_idx`` as ``int`` – Python does not enforce the type hint, so the
    string works with the existing filename formatting.
    """
    length = random.randint(min_len, max_len)
    return "".join(random.choices(string.ascii_letters + string.digits, k=length))


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route('/')
def index():
    """Render the minimal UI.

    The template lives in ``templates/index.html`` and contains a selector for
    composition length, a *Generate* button and a placeholder for the download
    link that appears after generation.
    """
    return render_template('index.html')


@app.route('/generate', methods=['POST'])
def generate():
    """Trigger music generation.

    Expected JSON payload::

        {"length": "short"}

    Returns a JSON object with the generated ``sample_idx`` which the frontend
    can use to build the download URL.
    """
    data = request.get_json(silent=True) or {}
    length = data.get('length', 'short')
    max_tokens = LENGTH_TO_TOKENS.get(length, LENGTH_TO_TOKENS['short'])

    sample_idx = _random_id()

    # Call the core generation routine – this writes both ``.txt`` and ``.midi``
    # files to ``sample/``.
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
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception as e:
        # If conversion fails, log but do not prevent the API response.
        print(f"TiMidity conversion failed for {midi_path}: {e}")

    return jsonify({"sample_idx": sample_idx})


@app.route('/download/<sample_idx>', methods=['GET'])
def download(sample_idx: str):
    """Serve the generated ``.midi`` file.

    The ``sample_idx`` is the identifier returned by ``/generate``.  The file
    is located in the ``sample`` directory and is sent with the correct MIME
    type (``audio/midi``).
    """
    filename = f"sample_{sample_idx}.midi"
    return send_from_directory(
        directory=SAMPLE_DIR,
        path=filename,
        as_attachment=True,
        mimetype='audio/midi'
    )


@app.route('/audio/<sample_idx>', methods=['GET'])
def audio(sample_idx: str):
    """Serve the WAV audio rendered by TiMidity++.

    The WAV file is generated during the ``/generate`` call and stored alongside
    the MIDI file.  It is streamed with the correct ``audio/wav`` MIME type.
    """
    filename = f"sample_{sample_idx}.wav"
    return send_from_directory(
        directory=SAMPLE_DIR,
        path=filename,
        as_attachment=True,
        mimetype='audio/wav'
    )


if __name__ == '__main__':
    # Bind to localhost only – this satisfies the strict local‑only requirement.
    app.run(host='127.0.0.1', port=5000, debug=False)
