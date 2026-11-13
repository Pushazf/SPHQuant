"""Answer extraction and scoring for multiple-choice outputs. Unparsable answers are incorrect; no random fallback."""
import re

THINK_END = "</think>"
LOOP_WINDOW = 512  # tokens at the end of a truncated output that are checked for a loop
LOOP_MAX_PERIOD = 128  # longest repeated n-gram considered
LOOP_MIN_SHARE = 0.5  # share of the window that must repeat the token one period earlier

# "The answer is B", "Answer: (B)", "**Answer:** B", "the correct option is B", "answer should be **B**".
_PHRASE = re.compile(
    r"\b(?i:answer|option|choice)[\s*]*(?:(?i:is|should be|would be|must be|will be)\b|:)"
    r"[\s*:\-(\[{$]*([A-Z])(?![A-Za-z0-9])"
)
_BOXED = re.compile(r"\\boxed\{[\s(]*(?:\\text(?:bf)?\{)?[\s(]*([A-Z])(?![A-Za-z0-9])")
# A line that is just a letter, or starts with one followed by a delimiter: "B", "(B)", "B.", "**B**", "B. text".
_BARE = re.compile(r"^[\s*(\[{$]*([A-Z])(?:\s*$|[)\]}.:*$]+(?:\s|$))")


def split_thinking(text, think=True):
    """Split at the last </think> -> (reasoning, answer, closed).

    Without </think>, a thinking-mode output is all (unclosed) reasoning and a non-thinking output is all answer.
    """
    reasoning, sep, answer = text.rpartition(THINK_END)
    if sep:
        return reasoning, answer, True
    return (text, "", False) if think else ("", text, False)


def extract_letter(answer, n_options):
    """Option letter from a final answer, or None. Explicit answer phrases win (last one), then the first and last lines."""
    found = [(m.start(), m.group(1)) for pattern in (_PHRASE, _BOXED) for m in pattern.finditer(answer)]
    if found:
        letter = max(found)[1]
    else:
        lines = [line for line in answer.strip().splitlines() if line.strip()]
        matches = [m for m in (_BARE.match(line) for line in lines[:1] + lines[-1:]) if m]
        if not matches:
            return None
        letter = matches[0].group(1)
    return letter if ord(letter) - ord("A") < n_options else None


def score(content, finish_reason, think, n_options, gold):
    """Score one completion. A truncated thinking output without </think> has no answer and is incorrect."""
    reasoning, answer, closed = split_thinking(content, think)
    letter = extract_letter(answer, n_options)
    return {
        "reasoning": reasoning,
        "answer": answer,
        "closed": closed,
        "letter": letter,
        "correct": letter == gold,
        "truncated": finish_reason == "length",
    }


def loop_flag(token_ids):
    """True if one repeated n-gram (n <= LOOP_MAX_PERIOD) dominates the final LOOP_WINDOW tokens."""
    tail = list(token_ids[-LOOP_WINDOW:])
    for period in range(1, min(LOOP_MAX_PERIOD, len(tail) // 2) + 1):
        same = sum(a == b for a, b in zip(tail, tail[period:]))
        if same >= LOOP_MIN_SHARE * len(tail):
            return True
    return False
