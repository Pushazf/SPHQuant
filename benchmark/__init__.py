"""In-repo MMMU-Pro evaluation for BF16 and SPHQuant W2 weights served by SGLang."""
import os
from pathlib import Path


def atomic_write_text(path, text):
    """Write text to path via a temporary file and os.replace."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
