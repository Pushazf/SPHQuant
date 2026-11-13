"""Benchmark client (RealWorldQA, ScienceQA, TextVQA) for an SGLang OpenAI-compatible
server: bounded concurrency, append-only fsync'd JSONL, resumable. The server is started separately
(scripts/benchmark.sh runs SGLang's own launch_server)."""
import argparse
import asyncio
import hashlib
import json
import os
import time
from pathlib import Path

import httpx

from benchmark import realworldqa, scienceqa, scoring, textvqa

STOP_TOKEN_IDS = [248046, 248044]  # <|im_end|>, <|endoftext|>
# Each module provides DATASET_ID, REVISION, INSTRUCTION, load, subset, chat_messages and score.
DATASETS = {"realworldqa": realworldqa, "scienceqa": scienceqa, "textvqa": textvqa}


def request_profile(think, max_tokens):
    """Every request field except the messages."""
    return {
        "model": "default",
        "temperature": 0,
        "max_tokens": max_tokens,
        "stop_token_ids": STOP_TOKEN_IDS,
        "chat_template_kwargs": {"enable_thinking": think},
    }


def profile_hash(profile, dataset):
    blob = json.dumps({"request": profile, "instruction": DATASETS[dataset].INSTRUCTION, "images": "png-rgb"},
                      sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def record_key(rec):
    return (rec["dataset"], rec["revision"], rec["doc_id"], rec["weights_id"], rec["think"], rec["profile_hash"])


def read_records(path):
    """All records of a JSONL run file (strict: every line must parse)."""
    path = Path(path)
    if not path.exists():
        return []
    lines = path.read_bytes().split(b"\n")
    if lines[-1]:
        raise ValueError(f"{path}: last line is not newline-terminated")
    return [json.loads(line) for line in lines[:-1]]


def repair_torn_tail(path):
    """Cut a final line without newline (a crash mid-append); complete lines are never touched."""
    path = Path(path)
    if not path.exists():
        return
    data = path.read_bytes()
    keep = data.rfind(b"\n") + 1
    if keep < len(data):
        with open(path, "r+b") as f:
            f.truncate(keep)
            os.fsync(f.fileno())


def append_record(f, rec):
    f.write((json.dumps(rec, ensure_ascii=False) + "\n").encode("utf-8"))
    f.flush()
    os.fsync(f.fileno())


def load_encoder(tokenizer_dir):
    """Token-id encoder from tokenizer.json (no special tokens added)."""
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(Path(tokenizer_dir) / "tokenizer.json"))
    return lambda text: tok.encode(text, add_special_tokens=False).ids


async def _post_with_retry(http, body, retries, backoff):
    """POST; only transport errors (connect, timeout, protocol) are retried, with exponential backoff."""
    for attempt in range(1, retries + 1):
        try:
            return await http.post("/v1/chat/completions", json=body), attempt
        except httpx.TransportError:
            if attempt == retries:
                raise
            await asyncio.sleep(backoff * 2 ** (attempt - 1))


async def evaluate_one(http, sample, *, profile, phash, weights_id, encode, retries, backoff, dataset):
    ds = DATASETS[dataset]
    think = profile["chat_template_kwargs"]["enable_thinking"]
    body = dict(profile, messages=await asyncio.to_thread(ds.chat_messages, sample))
    start = time.monotonic()
    resp, attempts = await _post_with_retry(http, body, retries, backoff)
    resp.raise_for_status()
    latency = time.monotonic() - start
    out = resp.json()
    choice = out["choices"][0]
    if choice["message"].get("reasoning_content"):
        raise RuntimeError("server separated the reasoning; launch it without --reasoning-parser")
    content = choice["message"].get("content") or ""
    finish = choice["finish_reason"]
    if finish not in ("stop", "length"):
        raise RuntimeError(f"{sample.doc_id}: finish_reason {finish!r}")
    s = ds.score(content, finish, think, sample)
    return {
        "dataset": ds.DATASET_ID,
        "revision": ds.REVISION,
        "doc_id": sample.doc_id,
        "weights_id": weights_id,
        "think": think,
        "profile_hash": phash,
        "subject": sample.subject,
        "gold": sample.answer,
        "n_options": len(sample.options),
        "content": content,
        "answer": s["answer"],
        "letter": s["letter"],
        "correct": s["correct"],
        "finish_reason": finish,
        "usage": out.get("usage"),
        "think_tokens": len(encode(s["reasoning"])),
        "answer_tokens": len(encode(s["answer"])),
        "closed": s["closed"],
        "truncated": s["truncated"],
        "loop": s["truncated"] and scoring.loop_flag(encode(content)),
        "latency_s": round(latency, 3),
        "attempts": attempts,
    }


async def run_samples(samples, out_path, *, base_url, weights_id, think, max_tokens, concurrency, encode,
                      dataset, transport=None, retries=5, backoff=2.0, timeout=7200.0):
    """Evaluate samples not yet in out_path; returns the records of all samples in sample order."""
    ds = DATASETS[dataset]
    profile = request_profile(think, max_tokens)
    phash = profile_hash(profile, dataset)

    def key(sample):
        return (ds.DATASET_ID, ds.REVISION, sample.doc_id, weights_id, think, phash)

    repair_torn_tail(out_path)
    done = {}
    for rec in read_records(out_path):
        if record_key(rec) in done:
            raise ValueError(f"{out_path}: duplicate record {record_key(rec)}")
        done[record_key(rec)] = rec
    pending = iter([s for s in samples if key(s) not in done])
    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    async with httpx.AsyncClient(base_url=base_url, transport=transport, timeout=httpx.Timeout(timeout, connect=60.0),
                                 limits=limits, trust_env=False) as http:
        with open(out_path, "ab") as f:

            async def worker():
                for sample in pending:  # shared iterator: dispatch follows sample order
                    rec = await evaluate_one(http, sample, profile=profile, phash=phash, weights_id=weights_id,
                                             encode=encode, retries=retries, backoff=backoff, dataset=dataset)
                    if record_key(rec) in done:
                        raise RuntimeError(f"double count of {record_key(rec)}")
                    append_record(f, rec)
                    done[record_key(rec)] = rec

            async with asyncio.TaskGroup() as tg:
                for _ in range(concurrency):
                    tg.create_task(worker())
    return [done[key(s)] for s in samples]


def summarize(records):
    """Rates over the given records (fractions; TextVQA's "correct" is itself a fraction)."""
    n = max(len(records), 1)
    return {
        "n": len(records),
        "accuracy": sum(r["correct"] for r in records) / n,
        "truncated": sum(r["truncated"] for r in records) / n,
        "loop": sum(r["loop"] for r in records) / n,
        "no_letter": sum(r["letter"] is None for r in records) / n,
    }


def served_model(base_url):
    """The model_path the server at base_url reports (/model_info, or /get_model_info in older SGLang)."""
    for path in ("/model_info", "/get_model_info"):
        r = httpx.get(base_url + path, timeout=30, trust_env=False)
        if r.status_code == 200:
            return r.json().get("model_path")
    raise RuntimeError(f"{base_url} reports no model info")


def check_served_model(base_url, model):
    """Refuse unless the server at base_url serves exactly MODEL (a directory, compared by real path)."""
    served = served_model(base_url)
    if not served or os.path.realpath(served) != os.path.realpath(model):
        raise RuntimeError(f"{base_url} serves {served!r}, not {str(model)!r}; refusing to evaluate another server")


def template_ok(rec):
    """Thinking off: no </think> in the output. Thinking on: </think> present unless truncated."""
    closed = scoring.THINK_END in rec["content"]
    if rec["think"]:
        return closed or rec["finish_reason"] == "length"
    return not closed


def select_samples(data_dir, limit, sample_seed, dataset):
    """The dataset's samples, or its seeded subset of LIMIT (stratified by subject or question type)."""
    ds = DATASETS[dataset]
    samples = ds.load(data_dir)
    return ds.subset(samples, limit, sample_seed) if limit else samples


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", required=True)
    p.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    p.add_argument("--data", required=True, help="snapshot directory of the dataset")
    p.add_argument("--weights-id", required=True)
    p.add_argument("--think", required=True, choices=["on", "off"])
    p.add_argument("--out", required=True)
    p.add_argument("--concurrency", type=int, default=64)
    p.add_argument("--max-tokens", type=int, required=True)
    p.add_argument("--tokenizer", required=True, help="directory with tokenizer.json")
    p.add_argument("--expect-model", help="refuse unless the server serves this model directory")
    p.add_argument("--limit", type=int, default=0, help="seeded stratified subset size (0 = all)")
    p.add_argument("--sample-seed", type=int, default=0)
    args = p.parse_args(argv)
    think = args.think == "on"
    samples = select_samples(args.data, args.limit, args.sample_seed, args.dataset)
    if args.expect_model:
        check_served_model(args.base_url, args.expect_model)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    kwargs = dict(base_url=args.base_url, weights_id=args.weights_id, think=think, max_tokens=args.max_tokens,
                  concurrency=args.concurrency, encode=load_encoder(args.tokenizer), dataset=args.dataset)
    first = asyncio.run(run_samples(samples[:1], args.out, **kwargs))[0]
    if not template_ok(first):
        raise SystemExit(f"chat template check failed for {first['doc_id']} with thinking {args.think}")
    records = asyncio.run(run_samples(samples, args.out, **kwargs))
    print(json.dumps({"out": args.out, "dataset": DATASETS[args.dataset].DATASET_ID, "weights_id": args.weights_id,
                      "think": think, "max_tokens": args.max_tokens, "limit": args.limit,
                      "sample_seed": args.sample_seed, "profile_hash": first["profile_hash"], **summarize(records)}))


if __name__ == "__main__":
    main()
