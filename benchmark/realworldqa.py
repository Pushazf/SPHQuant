"""RealWorldQA (xai-org/RealworldQA) test split: offline loading, prompts and scoring.

Every question already ends with its answer instruction: multiple-choice questions ask for the option letter
only, the others for a single word or number. The question is sent unchanged after its image. Unparsable answers
are incorrect; there is no random fallback.
"""
import re
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq

from benchmark import common, scoring

DATASET = "xai-org/RealworldQA"
SPLIT = "test"
REVISION = "17e7f75e092e47169732462ea3cdfebe911105dd"
DATASET_ID = f"{DATASET}/{SPLIT}"
# Parquet files at REVISION -> LFS sha256 (from the Hub tree API).
FILES = {
    "data/test-00000-of-00002.parquet": "2cf29e308a34e0255f23520796fc04445a01b8baab18686f1a4b3c3943f695b2",
    "data/test-00001-of-00002.parquet": "dac5b9f7cfb947a4865ea3424abfbdb3449d09e0a0b911a787cdc930cefefdc0",
}
N_ROWS = 765
MC_SUFFIX = "Please answer directly with only the letter of the correct option and nothing else."
SHORT_SUFFIX = "Please answer directly with a single word or number."
INSTRUCTION = "the dataset's own instruction at the end of each question"  # part of the request profile hash
MULTIPLE_CHOICE = "multiple-choice"
SHORT_ANSWER = "short-answer"
COLUMNS = ["image", "question", "answer"]
_OPTION = re.compile(r"\s*([A-Z])(?:\s*[.:]|\s+\S)")  # "A. Red", "A: Red", "C.Red", "B Only cars"
_NUMBER_WORDS = {w: str(i) for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
    "seventeen eighteen nineteen twenty".split())}
_LEAD = re.compile(r"^(?:final answer|the answer is|answer)\s*[:\-]?\s*")
_WORD = re.compile(r"[\s,.;:!?()\[\]\"']+")


@dataclass(frozen=True)
class Sample:
    doc_id: str
    subject: str  # MULTIPLE_CHOICE or SHORT_ANSWER; subsets are stratified by it
    question: str
    options: tuple  # the option letters of a multiple-choice question, () for a short answer
    answer: str
    images: tuple  # encoded image bytes (one image)


def parse_row(index, row):
    """Parquet row dict (image struct, question, answer) at dataset position INDEX -> Sample."""
    question, answer = row["question"], row["answer"].strip()
    body = question.rstrip()
    doc_id = f"rwqa_{index:04d}"
    if body.endswith(MC_SUFFIX):
        lines = body[: -len(MC_SUFFIX)].splitlines()[1:]  # options follow the first line of the question
        letters = [m.group(1) for m in map(_OPTION.match, lines) if m]
        if not letters or letters != [chr(ord("A") + i) for i in range(len(letters))] or answer not in letters:
            raise ValueError(f"{doc_id}: options {letters} do not fit answer {answer!r}")
        kind, options = MULTIPLE_CHOICE, tuple(letters)
    elif body.endswith(SHORT_SUFFIX):
        if not answer:
            raise ValueError(f"{doc_id}: empty answer")
        kind, options = SHORT_ANSWER, ()
    else:
        raise ValueError(f"{doc_id}: question ends with no known instruction")
    image = row.get("image")
    if not image or not image.get("bytes"):
        raise ValueError(f"{doc_id}: no image")
    return Sample(doc_id, kind, question, options, answer, (image["bytes"],))


def load(data_dir, verify=True):
    """All samples in dataset order; verify=True checks the file hashes at REVISION and the row count."""
    root = Path(data_dir)
    paths = sorted((root / "data").glob(f"{SPLIT}-*.parquet"))
    if verify:
        names = [p.relative_to(root).as_posix() for p in paths]
        if names != sorted(FILES):
            raise ValueError(f"expected {sorted(FILES)} under {root}, found {names}")
        for p, name in zip(paths, names):
            if common.sha256(p) != FILES[name]:
                raise ValueError(f"{p}: sha256 does not match revision {REVISION}")
    rows = [row for p in paths for row in pq.read_table(p, columns=COLUMNS).to_pylist()]
    samples = [parse_row(i, row) for i, row in enumerate(rows)]
    if verify and len(samples) != N_ROWS:
        raise ValueError(f"expected {N_ROWS} rows, found {len(samples)}")
    return samples


subset = common.stratified_subset  # seeded, proportional to question type, in dataset order


def chat_messages(sample):
    """OpenAI chat messages: one user turn, the image first (PNG data URL), then the question unchanged."""
    return common.user_turn(sample, sample.question)


def _norm(text):
    text = re.sub(r"[*_`]", "", text).strip().lower()
    text = _LEAD.sub("", text).strip(" \t.,;:!?\"'()[]")
    return _NUMBER_WORDS.get(text, text)


def short_answer(answer):
    """Normalized first non-empty line of a final answer (lower case, no markdown or edge punctuation, number words
    as digits), or None."""
    lines = [line for line in answer.splitlines() if line.strip()]
    return (_norm(lines[0]) or None) if lines else None


def short_correct(pred, gold):
    """The whole normalized answer, or its first word, equals the normalized gold answer."""
    if pred is None:
        return False
    gold = _norm(gold)
    first = _WORD.split(pred, maxsplit=1)[0]
    return pred == gold or _NUMBER_WORDS.get(first, first) == gold


def score(content, finish_reason, think, sample):
    """Score one completion. A truncated thinking output without </think> has no answer and is incorrect."""
    reasoning, answer, closed = scoring.split_thinking(content, think)
    if sample.options:
        pred = scoring.extract_letter(answer, len(sample.options))
        correct = pred == sample.answer
    else:
        pred = short_answer(answer)
        correct = short_correct(pred, sample.answer)
    return {"reasoning": reasoning, "answer": answer, "closed": closed, "letter": pred, "correct": correct,
            "truncated": finish_reason == "length"}
