"""Tables of decode_bench (and gemv_bench) results: tokens/s per method, mode and attention, the W2 speedup over
dense with the same mode and attention, where the time of a decode step goes, and per-projection GEMV timings.

python -m throughput.summarize --runs DIR/*_compiled.json [--gemv DIR/gemv.json] --out DIR/summary.md
"""
import argparse
import json
from pathlib import Path


ATTENTION = {"kernel": "split-KV kernel", "sdpa": "SDPA"}


def load(paths):
    """{(method, mode, attention): result}."""
    runs = {}
    for path in paths:
        r = json.loads(Path(path).read_text())
        key = (r["method"], r["mode"], r["attention"])
        if key in runs:
            raise ValueError(f"{path}: a second result for {key}")
        runs[key] = r
    return runs


def model_name(r):
    return f"{Path(r['config']).stem} ({r['architecture']}, random weights" + (
        ", q/k/v and gate/up fused)" if r["fused"] else ")")


def render(runs, gemv=None):
    first = next(iter(runs.values()))
    lines = [
        "# Decode throughput",
        "",
        f"{model_name(first)}; {first['hardware']['name']}; batch 1; {first['prefill']} prompt tokens; "
        f"{first['decode']} generated tokens; {first['dtype']} activations; median of {first['samples']} windows; "
        f"torch {first['versions']['torch']}, transformers {first['versions']['transformers']}.",
        "",
        "| method | mode | decode attention | decode tokens/s | ms/token | speedup | end-to-end tokens/s | speedup "
        "| TTFT ms | prefill ms | model GB | peak GB |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    order = lambda kv: (kv[0][2] != "kernel", kv[0][1], kv[0][0] != "dense", kv[0][0])  # noqa: E731
    for (method, mode, attention), r in sorted(runs.items(), key=order):
        ph = r["phases"]
        dense = runs.get(("dense", mode, attention))
        speed = {p: (f"{ph[p]['tokens_per_s'] / dense['phases'][p]['tokens_per_s']:.2f}x" if dense else "n/a")
                 for p in ("decode", "end_to_end")}
        lines.append(
            f"| {method} | {mode} | {ATTENTION[attention]} | {ph['decode']['tokens_per_s']:.1f} | "
            f"{ph['decode']['ms_per_token']:.3f} | "
            f"{speed['decode']} | {ph['end_to_end']['tokens_per_s']:.1f} | {speed['end_to_end']} | "
            f"{ph['end_to_end']['ttft_ms']:.1f} | {ph['prefill']['median_ms']:.1f} | {r['model_bytes'] / 1e9:.2f} | "
            f"{r['peak_allocated_bytes'] / 1e9:.2f} |")
    checks = [r["kernel_check"] for r in runs.values() if "kernel_check" in r]
    if checks:
        c = max(checks, key=lambda c: c["max_rel_error"])
        glu = (f" GLU outputs silu(gate) * up of the fused gate/up modules: {c['glu_max_rel_error']:.2e} "
               f"(tolerance {2 * c['tolerance']:.0e})." if "glu_max_rel_error" in c else "")
        lines += ["", f"Kernel check before timing: {c['modules']} packed modules, max relative error "
                      f"{c['max_rel_error']:.2e} (tolerance {c['tolerance']:.0e}; the same product with the FP16 "
                      f"weight: {c['fp16_weight_max_rel_error']:.2e})." + glu]
    skipped = ["/".join(key) for key, r in runs.items() if r.get("cudagraph_skips")]
    if skipped:
        lines += ["", f"WARNING: torch.compile skipped CUDA graphs in {', '.join(skipped)}; those decode steps did not "
                      "replay a graph."]
    profiled = {key: r["profile"] for key, r in runs.items() if "profile" in r}
    if profiled:
        kinds = sorted({k for p in profiled.values() for k in p["kernel_us_per_step"]},
                       key=lambda k: -max(p["kernel_us_per_step"].get(k, 0) for p in profiled.values()))
        lines += ["", "## GPU time of one decode step (us)", "",
                  "| method | mode | decode attention | wall | kernels | count | " + " | ".join(kinds) + " |",
                  "|---|---|---|---|---|---|" + "---|" * len(kinds)]
        for (method, mode, attention), p in sorted(profiled.items(), key=order):
            lines.append(f"| {method} | {mode} | {ATTENTION[attention]} | {p['wall_us_per_step']:.0f} | "
                         f"{p['kernel_us_total']:.0f} | "
                         f"{p['kernels_per_step']:.0f} | " +
                         " | ".join(f"{p['kernel_us_per_step'].get(k, 0):.0f}" for k in kinds) + " |")
        lines += ["", "Wall: time per decode step while profiling; kernels: GPU time of its kernels, by kind; the "
                      "difference is time the GPU waits between kernels."]
    if gemv:
        lines += ["", "## One projection at batch 1 (us per call, weights read from memory)", "",
                  "| projection | out x in | dense us | dense GB/s | W2 us | W2 GB/s | speedup |",
                  "|---|---|---|---|---|---|---|"]
        for name, row in gemv["projections"].items():
            w2 = min(row["w2"].values(), key=lambda r: r["us"]) if "w2" in row else None
            lines.append(f"| {name} | {row['out_features']} x {row['in_features']} | {row['dense']['us']:.1f} | "
                         f"{row['dense']['GBps']:.0f} | " +
                         (f"{w2['us']:.1f} | {w2['GBps']:.0f} | {row['speedup']:.2f}x |" if w2 else "- | - | - |"))
            g = row.get("glu", {})
            if "w2" in g:
                lines.append(f"| {name} + silu(gate) * up | {row['out_features']} x {row['in_features']} | "
                             f"{g['dense']['us']:.1f} | {g['dense']['GBps']:.0f} | {g['w2']['us']:.1f} | "
                             f"{g['w2']['GBps']:.0f} | {g['speedup']:.2f}x |")
    lines += ["", "Decode: single-token forwards after the prompt. End-to-end: prompt forward plus greedy generation. "
              "Decode attention: the attention of the single-token forwards (split-KV kernel of "
              "cuda_kernel/decode_attention.cu, or PyTorch's SDPA as transformers calls it); the prompt always uses "
              "SDPA. Model GB counts every parameter and buffer, including the unquantized embeddings and LM head."]
    return "\n".join(lines) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--gemv")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    gemv = json.loads(Path(args.gemv).read_text()) if args.gemv else None
    text = render(load(args.runs), gemv)
    Path(args.out).write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
