"""ScienceQA image subset, test split (lmms-lab-encoder/ScienceQA config ScienceQA-IMG, formerly lmms-lab/ScienceQA;
2017 questions): offline loading, prompts and scoring.

The prompt is "Context: <hint>" (when the question has one), the question, "A. ..." option lines and the letter
instruction, after the image. Unparsable answers are incorrect; there is no random fallback.
"""
from dataclasses import dataclass

from benchmark import common, scoring

DATASET = "lmms-lab-encoder/ScienceQA"
CONFIG = "ScienceQA-IMG"
SPLIT = "test"
REVISION = "69dd4d6b67373d38f96a5badd5d24d0eb5bcdc50"
DATASET_ID = f"{DATASET}/{CONFIG}/{SPLIT}"
# Parquet files at REVISION -> LFS sha256 (from the Hub tree API).
FILES = {f"{CONFIG}/test-00000-of-00001.parquet": "1f9729818b577d70325a0e59f488f6eeda74c827caddd56b75ccf3e36ed45064"}
N_ROWS = 2017
INSTRUCTION = common.LETTER_INSTRUCTION  # part of the request profile hash
COLUMNS = ["image", "question", "choices", "answer", "hint", "subject"]


@dataclass(frozen=True)
class Sample:
    doc_id: str
    subject: str  # natural, social or language science; subsets are stratified by it
    question: str
    options: tuple
    answer: str  # gold letter
    images: tuple  # encoded image bytes (one image)
    hint: str


def parse_row(index, row):
    """Parquet row dict at dataset position INDEX -> Sample (the dataset has no ids)."""
    doc_id = f"sqa_{index:04d}"
    choices = row["choices"] or []
    if not (len(choices) >= 2 and all(isinstance(c, str) and c.strip() for c in choices)):
        raise ValueError(f"{doc_id}: bad choices {choices!r}")
    if not 0 <= row["answer"] < len(choices):
        raise ValueError(f"{doc_id}: answer {row['answer']} outside {len(choices)} choices")
    image = row.get("image")
    if not image or not image.get("bytes"):
        raise ValueError(f"{doc_id}: no image")
    return Sample(doc_id, row["subject"], row["question"], tuple(choices), chr(ord("A") + row["answer"]),
                  (image["bytes"],), (row["hint"] or "").strip())


def load(data_dir, verify=True):
    """All samples in dataset order; verify=True checks the file hashes at REVISION and the row count."""
    rows = common.read_rows(data_dir, FILES, COLUMNS, REVISION, verify)
    samples = [parse_row(i, r) for i, r in enumerate(rows)]
    if verify and len(samples) != N_ROWS:
        raise ValueError(f"expected {N_ROWS} rows, found {len(samples)}")
    return samples


subset = common.stratified_subset


def prompt_text(sample):
    context = f"Context: {sample.hint}\n" if sample.hint else ""
    options = "\n".join(f"{chr(ord('A') + i)}. {o}" for i, o in enumerate(sample.options))
    return f"{context}{sample.question}\n{options}\n{INSTRUCTION}"


def chat_messages(sample):
    return common.user_turn(sample, prompt_text(sample))


def score(content, finish_reason, think, sample):
    return scoring.score(content, finish_reason, think, len(sample.options), sample.answer)
