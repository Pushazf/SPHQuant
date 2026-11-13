"""Helpers shared by the dataset modules: pinned parquet files, seeded subsets and chat messages."""
import base64
import hashlib
import io
import random
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

LETTER_INSTRUCTION = "Answer with the option's letter from the given choices directly."


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(data_dir, files, columns, revision, verify=True):
    """Rows of the parquet FILES ({relative path: sha256}, read in sorted order) under DATA_DIR; verify=True checks
    each file's sha256 at REVISION."""
    rows = []
    for name in sorted(files):
        path = Path(data_dir) / name
        if verify and sha256(path) != files[name]:
            raise ValueError(f"{path}: sha256 does not match revision {revision}")
        rows += pq.read_table(path, columns=columns).to_pylist()
    return rows


def stratified_subset(samples, k, seed):
    """k samples, allocated to subjects proportionally (largest remainder, seeded ties), in dataset order."""
    if k >= len(samples):
        return list(samples)
    rng = random.Random(seed)
    by_subject = defaultdict(list)
    for i, s in enumerate(samples):
        by_subject[s.subject].append(i)
    subjects = sorted(by_subject)
    quota = {s: k * len(by_subject[s]) / len(samples) for s in subjects}
    alloc = {s: int(quota[s]) for s in subjects}
    tie = {s: rng.random() for s in subjects}
    for s in sorted(subjects, key=lambda s: (alloc[s] - quota[s], tie[s]))[: k - sum(alloc.values())]:
        alloc[s] += 1
    chosen = [i for s in subjects for i in rng.sample(by_subject[s], alloc[s])]
    return [samples[i] for i in sorted(chosen)]


def rgb_images(sample):
    return [Image.open(io.BytesIO(b)).convert("RGB") for b in sample.images]


def png_data_url(image):
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def user_turn(sample, text):
    """OpenAI chat messages: one user turn, the sample's images first (PNG data URLs), then TEXT."""
    content = [{"type": "image_url", "image_url": {"url": png_data_url(im)}} for im in rgb_images(sample)]
    return [{"role": "user", "content": content + [{"type": "text", "text": text}]}]
