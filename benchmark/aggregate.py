"""Summarize benchmark runs of one dataset: accuracy per (weights, think), paired bootstrap deltas,
truncation/loop/parse rates, think tokens."""
import argparse
import json
from collections import defaultdict

import numpy as np

from benchmark import atomic_write_text, client, realworldqa, scienceqa, textvqa

# Qwen3.5-9B in the paper; reference only (different harness): dataset id -> (name, thinking mode, numbers).
PAPER = {
    realworldqa.DATASET_ID: ("RealWorldQA", "thinking off", {"bf16": 67.1, "w2": 61.3}),  # paper Table 1
    textvqa.DATASET_ID: ("TextVQA", "thinking off", {"bf16": 69.1, "w2": 63.7}),  # paper Table 1
    scienceqa.DATASET_ID: ("ScienceQA", "thinking off", {"bf16": 92.5, "w2": 83.0}),  # paper Table 1
}


def dataset_of(groups):
    """The one dataset id of all records; refuses runs of different datasets."""
    ids = {r["dataset"] for recs in groups.values() for r in recs.values()}
    if len(ids) != 1:
        raise ValueError(f"runs mix datasets: {sorted(ids)}")
    return ids.pop()


def load_groups(paths):
    """Records grouped by (weights_id, think) -> {doc_id: record}; refuses duplicates and mixed profiles."""
    groups = defaultdict(dict)
    profiles = defaultdict(set)
    for path in paths:
        for rec in client.read_records(path):
            group = groups[(rec["weights_id"], rec["think"])]
            if rec["doc_id"] in group:
                raise ValueError(f"{path}: {rec['doc_id']} appears twice for {rec['weights_id']} think={rec['think']}")
            group[rec["doc_id"]] = rec
            profiles[(rec["weights_id"], rec["think"])].add((rec["dataset"], rec["revision"], rec["profile_hash"]))
    for g, ps in profiles.items():
        if len(ps) > 1:
            raise ValueError(f"{g} mixes dataset revisions or request profiles: {sorted(ps)}")
    return dict(groups)


def _pct(flags):
    return 100.0 * float(np.mean(flags))


def _mode(think):
    return "on" if think else "off"


def group_stats(recs):
    """Percent rates and thinking-token statistics for one group of records."""
    recs = list(recs)
    tokens = np.array([r["think_tokens"] for r in recs], dtype=float)
    p50, p90 = np.percentile(tokens, [50, 90])
    return {
        "n": len(recs),
        "accuracy": _pct([r["correct"] for r in recs]),
        "truncated": _pct([r["truncated"] for r in recs]),
        "loop": _pct([r["loop"] for r in recs]),
        "parse_fail": _pct([r["letter"] is None and not r["truncated"] for r in recs]),
        "think_mean": float(tokens.mean()),
        "think_p50": float(p50),
        "think_p90": float(p90),
    }


def paired_delta(base, other, n_boot, seed, chunk=1000):
    """Accuracy delta other - base (percent points) on shared doc ids, with a paired bootstrap 95% CI."""
    ids = sorted(base.keys() & other.keys())
    if not ids:
        return None
    d = np.array([float(other[i]["correct"]) - float(base[i]["correct"]) for i in ids])
    rng = np.random.default_rng(seed)
    means = np.concatenate([
        d[rng.integers(0, len(d), size=(min(chunk, n_boot - start), len(d)))].mean(axis=1)
        for start in range(0, n_boot, chunk)
    ])
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"n": len(ids), "delta": 100.0 * float(d.mean()), "lo": 100.0 * float(lo), "hi": 100.0 * float(hi)}


def summarize(groups, base, n_boot, seed):
    stats = {g: group_stats(recs.values()) for g, recs in sorted(groups.items())}
    weights = sorted({w for w, _ in groups})
    vs_base = {}
    for think in (False, True):
        for w in weights:
            if w != base and (base, think) in groups and (w, think) in groups:
                vs_base[(w, think)] = paired_delta(groups[(base, think)], groups[(w, think)], n_boot, seed)
    on_off = {w: paired_delta(groups[(w, False)], groups[(w, True)], n_boot, seed)
              for w in weights if (w, False) in groups and (w, True) in groups}
    return stats, vs_base, on_off


def _ci(d):
    return "0 | n/a | n/a" if d is None else f"{d['n']} | {d['delta']:+.2f} | [{d['lo']:+.2f}, {d['hi']:+.2f}]"


def render(stats, vs_base, on_off, base, n_boot, seed, dataset):
    name, paper_mode, paper = PAPER.get(dataset, (dataset, None, None))
    lines = [
        f"# {name} summary",
        "",
        f"{dataset}; percentages; think tokens from the local tokenizer.",
        "",
        "| weights | think | n | accuracy | truncated | loop | parse fail | think tok mean | P50 | P90 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for (w, think), s in stats.items():
        lines.append(f"| {w} | {_mode(think)} | {s['n']} | {s['accuracy']:.2f} | {s['truncated']:.2f} | {s['loop']:.2f} | "
                     f"{s['parse_fail']:.2f} | {s['think_mean']:.0f} | {s['think_p50']:.0f} | {s['think_p90']:.0f} |")
    lines += ["", f"## Delta vs {base} (paired bootstrap 95% CI, {n_boot} resamples, seed {seed})", "",
              "| weights | think | n paired | delta | 95% CI |", "|---|---|---|---|---|"]
    lines += [f"| {w} | {_mode(think)} | {_ci(d)} |" for (w, think), d in vs_base.items()]
    lines += ["", "## Thinking on - off (paired, same bootstrap)", "",
              "| weights | n paired | delta | 95% CI |", "|---|---|---|---|"]
    lines += [f"| {w} | {_ci(d)} |" for w, d in on_off.items()]
    if paper:
        lines += ["", "## Paper reference (not a gate; different harness)", "",
                  f"{name}, {paper_mode}: " + ", ".join(f"{k} {v}" for k, v in paper.items())
                  + f" (delta {paper['w2'] - paper['bf16']:+.1f})."]
    return "\n".join(lines + [""])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--base", default="bf16")
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    groups = load_groups(args.runs)
    dataset = dataset_of(groups)
    stats, vs_base, on_off = summarize(groups, args.base, args.bootstrap, args.seed)
    atomic_write_text(args.out, render(stats, vs_base, on_off, args.base, args.bootstrap, args.seed, dataset))
    print(json.dumps({
        "out": args.out,
        "dataset": dataset,
        "accuracy": {f"{w}/think-{_mode(t)}": round(s["accuracy"], 2) for (w, t), s in stats.items()},
        "delta_vs_base": {f"{w}/think-{_mode(t)}": d for (w, t), d in vs_base.items()},
        "on_minus_off": on_off,
    }))


if __name__ == "__main__":
    main()
