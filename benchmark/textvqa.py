"""TextVQA validation split (lmms-lab-encoder/textvqa, formerly lmms-lab/textvqa; 5000 questions, 10 human answers
each): offline loading, prompts and VQA-accuracy scoring.

The prompt is the question (first letter capitalized) and the single-word-or-phrase instruction; no OCR tokens are
given. The answer's first non-empty line is normalized as the VQA evaluation normalizes answers (lower case,
punctuation, periods, number words, articles), and a question scores the mean over its 10 answers of
min(1, matches among the other 9 / 3). Records therefore carry a fractional "correct" in [0, 1].
"""
import re
from dataclasses import dataclass

from benchmark import common, scoring

DATASET = "lmms-lab-encoder/textvqa"
SPLIT = "validation"
REVISION = "9c0699cd19768ac5ab97568f6b3cbac4c0062884"
DATASET_ID = f"{DATASET}/{SPLIT}"
# Parquet files at REVISION -> LFS sha256 (from the Hub tree API).
FILES = {
    "data/validation-00000-of-00003.parquet": "491649b5b8add2c5bde13a1e36707002fcd11898699255d65e2e568ac4fbddff",
    "data/validation-00001-of-00003.parquet": "909ad12a378217431548a7f4b66fda163a6def9d21e70844c96449e64206c2dd",
    "data/validation-00002-of-00003.parquet": "a770d03ec95fef58def04cda949d97669b7666caaf174af927856d75684e6fe6",
}
N_ROWS = 5000
N_ANSWERS = 10
INSTRUCTION = "Answer the question using a single word or phrase."  # part of the request profile hash
SUBJECT = "textvqa"  # one stratum: the subset is a plain seeded sample
COLUMNS = ["question_id", "question", "image", "answers"]
_PUNCT = ";/[]\"{}()=+\\_-><@`,?!"
_PERIOD = re.compile(r"\.(?!\d)")  # a period not followed by a digit
_NUMBER_WORDS = {"none": "0", "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6",
                 "seven": "7", "eight": "8", "nine": "9", "ten": "10"}
_ARTICLES = {"a", "an", "the"}
_LEAD = re.compile(r"^(?:final answer|the answer is|answer)\s*[:\-]?\s*")


@dataclass(frozen=True)
class Sample:
    doc_id: str
    subject: str
    question: str
    options: tuple  # always (): open answers
    answer: tuple  # the 10 human answers
    images: tuple  # encoded image bytes (one image)


def parse_row(row):
    doc_id = f"textvqa_{row['question_id']}"
    answers = tuple(row["answers"] or ())
    if len(answers) != N_ANSWERS or not all(isinstance(a, str) for a in answers):
        raise ValueError(f"{doc_id}: expected {N_ANSWERS} answers, found {answers!r}")
    image = row.get("image")
    if not image or not image.get("bytes"):
        raise ValueError(f"{doc_id}: no image")
    if not row["question"].strip():
        raise ValueError(f"{doc_id}: empty question")
    return Sample(doc_id, SUBJECT, row["question"], (), answers, (image["bytes"],))


def load(data_dir, verify=True):
    """All samples in dataset order; verify=True checks the file hashes at REVISION and the row count."""
    samples = [parse_row(r) for r in common.read_rows(data_dir, FILES, COLUMNS, REVISION, verify)]
    if len({s.doc_id for s in samples}) != len(samples):
        raise ValueError("duplicate doc ids")
    if verify and len(samples) != N_ROWS:
        raise ValueError(f"expected {N_ROWS} rows, found {len(samples)}")
    return samples


subset = common.stratified_subset


def prompt_text(sample):
    q = sample.question.strip()
    return q[:1].upper() + q[1:] + "\n" + INSTRUCTION


def chat_messages(sample):
    return common.user_turn(sample, prompt_text(sample))


def normalize(text):
    """VQA answer normalization: lower case, no commas or question marks, other punctuation removed where it touches
    a space in the answer (else a space), periods not followed by a digit dropped, number words as digits, no
    articles."""
    text = text.lower().replace(",", "").replace("?", "").replace("'s", " 's")
    text = text.replace("\n", " ").replace("\t", " ").strip()
    spaced = text
    for p in _PUNCT:
        text = text.replace(p, "" if (p + " " in spaced or " " + p in spaced) else " ")
    text = _PERIOD.sub("", text)
    words = [_NUMBER_WORDS.get(w, w) for w in text.split()]
    return " ".join(w for w in words if w not in _ARTICLES)


def prediction(answer):
    """Normalized first non-empty line of a final answer (markdown and a leading "answer:" removed), or None."""
    lines = [line for line in answer.splitlines() if line.strip()]
    if not lines:
        return None
    line = _LEAD.sub("", re.sub(r"[*`]", "", lines[0]).strip().lower())
    return normalize(line) or None


def vqa_accuracy(pred, answers):
    """Mean over the answers of min(1, matches among the other answers / 3)."""
    if pred is None:
        return 0.0
    golds = [normalize(a) for a in answers]
    hits = [g == pred for g in golds]
    return sum(min(1.0, (sum(hits) - hit) / 3) for hit in hits) / len(golds)


def score(content, finish_reason, think, sample):
    """Score one completion; "correct" is the VQA accuracy. A truncated thinking output has no answer."""
    reasoning, answer, closed = scoring.split_thinking(content, think)
    pred = prediction(answer)
    return {"reasoning": reasoning, "answer": answer, "closed": closed, "letter": pred,
            "correct": vqa_accuracy(pred, sample.answer), "truncated": finish_reason == "length"}
