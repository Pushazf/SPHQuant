"""Batch-1 timings of each decoder projection: dense FP16 (torch matmul) vs SPHQuant W2 (fused GEMV kernel).

python -m throughput.gemv_bench --config throughput/configs/qwen3-8b.json --kernel SO --out FILE.json

Shapes come from the config, with q/k/v and gate/up fused as in decode_bench. Each projection is timed as in a
decode step: a CUDA graph of back-to-back calls that cycle through enough random copies of the weight to exceed L2
four times, so every call reads its weight from memory. W2 is timed for each rows-per-block setting of the kernel
(--blocks-per-sm) with and without programmatic dependent launch (--pdl), dense once. The gate/up projection is
also timed with its activation, silu(gate) * up: dense as the matmul followed by PyTorch's SiLU and product, W2 as
the GLU mode of the kernel. Reports microseconds per call and the bytes of weight read per second.
"""
import argparse
import json
from pathlib import Path

import torch


def shapes(config):
    t = config.get("text_config", config)
    h, inter = t["hidden_size"], t["intermediate_size"]
    hd = t.get("head_dim") or h // t["num_attention_heads"]
    q, kv = t["num_attention_heads"] * hd, t["num_key_value_heads"] * hd
    return {"qkv_proj": ([q, kv, kv], h), "o_proj": ([h], q), "gate_up_proj": ([inter, inter], h),
            "down_proj": ([h], inter), "lm_head": ([t["vocab_size"]], h)}


def dense_glu(x, w):
    gate, up = (x @ w.T).chunk(2, dim=-1)
    return torch.nn.functional.silu(gate) * up


def time_calls(fns, reps, iters):
    """Microseconds per call of the functions FNS, cycled REPS times inside one CUDA graph."""
    for f in fns:
        f()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for i in range(reps):
            fns[i % len(fns)]()
    graph.replay()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(iters):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / (iters * reps)


@torch.inference_mode()
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--kernel", required=True, help="sph_gemv.so")
    p.add_argument("--blocks-per-sm", default="2", help="GEMV work splits to time for W2")
    p.add_argument("--pdl", default="1,0", help="W2 launches with (1) and without (0) programmatic dependent launch")
    p.add_argument("--reps", type=int, default=64)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    from cuda_kernel import packed_linear as pl
    from throughput.random_weights import random_export

    pl.load_library(args.kernel)
    defaults = pl.BLOCKS_PER_SM, pl.PDL
    torch.cuda.set_device(0)
    props = torch.cuda.get_device_properties(0)
    gen = torch.Generator(device="cuda").manual_seed(args.seed)
    config = json.loads(Path(args.config).read_text())
    results = {"hardware": {"name": props.name, "sm_count": props.multi_processor_count,
                            "l2_bytes": props.L2_cache_size}, "torch": torch.__version__, "projections": {}}
    for name, (parts, k) in shapes(config).items():
        n = sum(parts)
        dense_bytes = n * k * 2
        copies = max(2, -(-4 * props.L2_cache_size // dense_bytes))
        x = torch.randn(1, k, generator=gen, device="cuda", dtype=torch.float16)
        dense = [torch.empty(n, k, device="cuda", dtype=torch.float16).normal_(0, 0.02, generator=gen)
                 for _ in range(copies)]
        us = time_calls([lambda w=w: x @ w.T for w in dense], args.reps, args.iters)
        row = {"out_features": n, "in_features": k, "parts": parts,
               "dense": {"us": us, "GBps": dense_bytes / us / 1e3, "copies": copies}}
        glu = name == "gate_up_proj"
        if glu:
            us = time_calls([lambda w=w: dense_glu(x, w) for w in dense], args.reps, args.iters)
            row["glu"] = {"dense": {"us": us, "GBps": dense_bytes / us / 1e3}}
        del dense
        if name != "lm_head":
            w2_bytes = n * k * 2.5 / 8
            copies = max(2, -(-4 * props.L2_cache_size // int(w2_bytes)))
            mods = []
            for _ in range(copies):
                ms = [pl.PackedSPHLinear.from_export(random_export(m, k, gen, "cuda")) for m in parts]
                mods.append(ms[0] if len(ms) == 1 else pl.fuse(ms))
            row["w2"] = {}
            for bps in (int(b) for b in args.blocks_per_sm.split(",")):
                for pdl in (int(v) for v in args.pdl.split(",")):
                    pl.BLOCKS_PER_SM, pl.PDL = bps, bool(pdl)
                    us = time_calls([lambda m=m: m(x) for m in mods], args.reps, args.iters)
                    row["w2"][f"bps{bps}" + ("" if pdl else "_nopdl")] = {
                        "us": us, "GBps": w2_bytes / us / 1e3, "blocks_per_sm": bps, "pdl": bool(pdl),
                        "rows_per_block": pl.rows_per_block(n, k, props.multi_processor_count), "copies": copies}
            best = min(row["w2"].values(), key=lambda r: r["us"])
            row["speedup"] = row["dense"]["us"] / best["us"]
            if glu:
                pl.BLOCKS_PER_SM, pl.PDL = defaults
                us = time_calls([lambda m=m: m.forward_glu(x) for m in mods], args.reps, args.iters)
                row["glu"]["w2"] = {"us": us, "GBps": w2_bytes / us / 1e3}
                row["glu"]["speedup"] = row["glu"]["dense"]["us"] / us
            pl.BLOCKS_PER_SM, pl.PDL = defaults
            del mods
        torch.cuda.empty_cache()
        results["projections"][name] = row
        print(name, json.dumps({k_: v for k_, v in row.items() if k_ != "parts"}), flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
