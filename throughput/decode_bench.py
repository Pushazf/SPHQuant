"""Batch-1 decode throughput of one configuration: dense weights, or SPHQuant W2 through the fused GEMV kernel.

python -m throughput.decode_bench --config throughput/configs/qwen3-8b.json --method dense|w2 --kernel SO \
    --mode compiled|eager [--attention kernel|sdpa] --out FILE.json

--config builds the architecture with random weights (throughput/random_weights.py); no checkpoint is needed, since
only the shapes matter for speed. With --method w2 every decoder projection is a PackedSPHLinear with random codes.
By default q/k/v and gate/up are fused into one projection each, for dense and W2 alike.
Attention (both methods): with --attention kernel (default) every decode step runs the split-KV kernel of
cuda_kernel/decode_attention.cu and the prompt runs PyTorch's SDPA; --attention sdpa uses SDPA throughout, as
transformers does by default (at batch 1 its memory-efficient kernel, after copying K and V once per query head).
Inputs are random token ids (seeded), batch 1. Every window starts from an empty static KV cache and is timed on its
own; three phases:
  prefill     one forward of the --prefill prompt tokens
  decode      after an untimed prompt forward, --decode - 1 single-token forwards on fixed tokens (decode speed)
  end_to_end  the prompt forward, then greedy generation on the GPU up to --decode tokens in total
Compiled mode compiles the single-token forward with torch.compile(fullgraph=True, dynamic=False,
mode="reduce-overhead"), so every decode step replays a CUDA graph; the prompt forward stays eager, as in
transformers' own generation with a static cache. Compilation, graph capture and --warmup windows are untimed, and
measured windows refuse to recompile. Eager mode runs the same loops without compilation. --profile N records the
GPU kernels of N compiled decode steps after timing and adds their time per step, by kind, to the result.
"""
import argparse
import json
import platform
import re
import time
from collections import defaultdict
from importlib.metadata import version
from pathlib import Path
from statistics import median

import torch

PHASES = ("prefill", "decode", "end_to_end")
REL_TOL = 4e-3  # kernel check: max |y - ref| / max |ref| per row, against the FP32 product with the decoded weight
CHECK_CHUNK = 4096  # rows of a weight decoded at a time for the check
KINDS = (  # kernel name patterns, first match wins
    ("w2_gemv", r"sph_gemv"),
    ("w2_dequant", r"sph_dequant"),
    ("fused_elementwise", r"^triton"),  # inductor kernels (norms, rotary, residual adds, cache writes, ...)
    ("attention", r"flash|fmha|AttentionKernel|attention"),  # includes sph_decode_attention_run
    ("dense_matmul", r"gemv|gemm|cutlass|cublas|nvjet|xmma|splitK|matmul"),
)


ATTENTION = "sph_decode"  # name of the decode attention in transformers' attention and mask interfaces


def use_decode_attention(model):
    """Single-token forwards of MODEL use the split-KV decode kernel, longer ones (the prompt) transformers' SDPA;
    the mask is SDPA's."""
    from transformers import AttentionInterface
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    from transformers.masking_utils import AttentionMaskInterface, sdpa_mask

    from cuda_kernel import decode_attention as da

    def forward(module, query, key, value, attention_mask, dropout=0.0, scaling=None, sliding_window=None, **kwargs):
        if dropout == 0.0 and sliding_window is None and da.supported(query, key, value, attention_mask):
            scale = scaling if scaling is not None else query.shape[-1] ** -0.5
            return da.decode_attention(query, key, value, attention_mask, scale), None
        return sdpa_attention_forward(module, query, key, value, attention_mask, dropout=dropout, scaling=scaling,
                                      sliding_window=sliding_window, **kwargs)

    AttentionInterface.register(ATTENTION, forward)
    AttentionMaskInterface.register(ATTENTION, sdpa_mask)
    da.prepare(next(model.parameters()).device)
    model.set_attn_implementation(ATTENTION)


def quantized_modules(model):
    from cuda_kernel.packed_linear import PackedSPHLinear

    return {name: m for name, m in model.named_modules() if isinstance(m, PackedSPHLinear)}


def _rel_error(y, ref):
    return ((y - ref).abs().amax(1) / ref.abs().amax(1).clamp_min(1e-30)).max().item()


@torch.inference_mode()
def check_kernel(modules, seed):
    """Every packed module against its weight decoded in PyTorch (FP32), on random FP16 inputs of 1 row (GEMV
    kernel) and 16 rows (dequantize + GEMM); fused gate/up modules also in GLU mode, against silu(gate) * up of the
    reference (tolerance doubled: the product adds the errors of both factors). Also reports the error of the same
    product with the FP16 weight."""
    from cuda_kernel import packed_linear as pl

    gen = torch.Generator(device="cuda").manual_seed(seed)
    worst, worst_fp16, worst_glu = 0.0, 0.0, None
    for name, m in modules.items():
        glu = pl.GLU and name.endswith("gate_up_proj")
        for rows in (1, 16):
            x = torch.randn(rows, m.in_features, generator=gen, device="cuda", dtype=torch.float16)
            ref, ref16 = [], []
            for r0 in range(0, m.out_features, CHECK_CHUNK):
                w = pl.decode_run(m.run(), m.ends, range(r0, min(r0 + CHECK_CHUNK, m.out_features)))
                ref.append(x.float() @ w.T)
                ref16.append((x @ w.half().T).float())
            ref, ref16 = torch.cat(ref, 1), torch.cat(ref16, 1)
            e = _rel_error(m(x).float(), ref)
            worst = max(worst, e)
            worst_fp16 = max(worst_fp16, _rel_error(ref16, ref))
            if not e <= REL_TOL:
                raise RuntimeError(f"kernel check failed for {name} ({rows} rows): relative error {e:.2e}")
            if glu:
                half = m.out_features // 2
                e = _rel_error(m.forward_glu(x).float(), torch.nn.functional.silu(ref[:, :half]) * ref[:, half:])
                worst_glu = max(worst_glu or 0.0, e)
                if not e <= 2 * REL_TOL:
                    raise RuntimeError(f"GLU check failed for {name} ({rows} rows): relative error {e:.2e}")
    out = {"modules": len(modules), "max_rel_error": worst, "fp16_weight_max_rel_error": worst_fp16,
           "tolerance": REL_TOL}
    if worst_glu is not None:
        out["glu_max_rel_error"] = worst_glu
    return out


def forward(model, cache, ids):
    return model(input_ids=ids, past_key_values=cache, use_cache=True, logits_to_keep=1)


@torch.inference_mode()
def window(model, step, cache, prompt, tokens, phase, decode):
    """One timed window from an empty cache -> {"ms": wall time} (+ "ttft_ms" for end_to_end). MODEL runs the
    prompt, STEP (the compiled model in compiled mode) every single-token forward."""
    cache.reset()
    if phase == "decode":
        forward(model, cache, prompt)
    start, first, end = (torch.cuda.Event(enable_timing=True) for _ in range(3))
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    start.record()
    if phase == "prefill":
        forward(model, cache, prompt)
    elif phase == "decode":
        for ids in tokens:
            forward(step, cache, ids)
    else:
        out = forward(model, cache, prompt)
        token = out.logits[:, -1].argmax(-1, keepdim=True)
        del out
        first.record()
        generated = [token]
        for _ in range(decode - 1):
            out = forward(step, cache, token)
            token = out.logits[:, -1].argmax(-1, keepdim=True)
            del out
            generated.append(token)
        torch.cat(generated, dim=1)
    end.record()
    end.synchronize()
    result = {"ms": (time.perf_counter() - t0) * 1000, "cuda_ms": start.elapsed_time(end)}
    if phase == "end_to_end":
        result["ttft_ms"] = start.elapsed_time(first)
    return result


def summarize(phase, runs, prefill, decode):
    ms = [r["ms"] for r in runs]
    out = {"runs": runs, "median_ms": median(ms)}
    if phase == "prefill":
        out["tokens_per_s"] = prefill * 1000 / median(ms)
    elif phase == "decode":
        out["tokens_per_s"] = (decode - 1) * 1000 / median(ms)
        out["ms_per_token"] = median(ms) / (decode - 1)
    else:
        ttft = median(r["ttft_ms"] for r in runs)
        out.update({"tokens_per_s": decode * 1000 / median(ms), "ttft_ms": ttft,
                    "tpot_ms": (median(ms) - ttft) / max(decode - 1, 1)})
    return out


def kind_of(name):
    for kind, pattern in KINDS:
        if re.search(pattern, name, re.IGNORECASE):
            return kind
    return "other"


@torch.inference_mode()
def profile_decode(model, step, cache, prompt, tokens, steps):
    """GPU kernels of STEPS decode steps (CUDA graph replays in compiled mode): time per step by kind, the busiest
    kernels, and the wall time per step for comparison."""
    from torch.profiler import ProfilerActivity, profile

    cache.reset()
    forward(model, cache, prompt)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        t0 = time.perf_counter()
        for ids in tokens[:steps]:
            forward(step, cache, ids)
        torch.cuda.synchronize()
        wall_us = (time.perf_counter() - t0) * 1e6 / steps
    by_kind, by_name, count = defaultdict(float), defaultdict(lambda: [0.0, 0]), 0
    for e in prof.events():
        if e.device_type != torch.autograd.DeviceType.CUDA or e.device_time <= 0:
            continue
        name = e.name
        if name.lower().startswith(("memcpy", "memset")):
            kind = "memcpy_memset"
        else:
            kind = kind_of(name)
        by_kind[kind] += e.device_time / steps
        by_name[name][0] += e.device_time / steps
        by_name[name][1] += 1
        count += 1
    top = sorted(by_name.items(), key=lambda kv: -kv[1][0])[:20]
    return {"steps": steps, "wall_us_per_step": wall_us, "kernels_per_step": count / steps,
            "kernel_us_per_step": dict(sorted(by_kind.items(), key=lambda kv: -kv[1])),
            "kernel_us_total": sum(by_kind.values()),
            "top": [{"name": n[:160], "us_per_step": t, "calls_per_step": c / steps, "kind": kind_of(n)}
                    for n, (t, c) in top]}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="config.json of the architecture, random weights (e.g. "
                                                   "throughput/configs/qwen3-8b.json)")
    p.add_argument("--method", choices=("dense", "w2"), default="dense")
    p.add_argument("--kernel", help="sph_gemv.so (python -m cuda_kernel.build); required for w2 and for "
                                    "--attention kernel")
    p.add_argument("--no-fuse", action="store_true", help="keep q/k/v and gate/up separate")
    p.add_argument("--mode", choices=("compiled", "eager"), default="compiled")
    p.add_argument("--attention", choices=("kernel", "sdpa"), default="kernel",
                   help="decode steps: split-KV kernel (default) or SDPA")
    p.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    p.add_argument("--prefill", type=int, default=256)
    p.add_argument("--decode", type=int, default=512, help="generated tokens, including the first one")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--samples", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--profile", type=int, default=0, metavar="N", help="profile N decode steps after timing")
    p.add_argument("--no-fullgraph", action="store_true", help="allow graph breaks in compiled mode")
    p.add_argument("--out", required=True, help="result JSON file")
    args = p.parse_args(argv)
    if (args.method == "w2" or args.attention == "kernel") and not args.kernel:
        p.error("--method w2 and --attention kernel need --kernel")
    from transformers import StaticCache

    from throughput.random_weights import build

    torch.cuda.set_device(0)
    torch.manual_seed(args.seed)
    dtype = getattr(torch, args.dtype)
    if args.kernel:
        from cuda_kernel.packed_linear import load_library

        load_library(args.kernel)
    model, _ = build(args.config, args.method, dtype, args.seed, fuse=not args.no_fuse)
    if args.attention == "kernel":
        use_decode_attention(model)
    torch.cuda.empty_cache()
    result = {"method": args.method, "mode": args.mode, "attention": args.attention, "dtype": args.dtype,
              "batch_size": 1,
              "prefill": args.prefill, "decode": args.decode, "warmup": args.warmup, "samples": args.samples,
              "seed": args.seed, "config": args.config, "fused": not args.no_fuse,
              "architecture": type(model).__name__}
    if args.method == "w2":
        from cuda_kernel import packed_linear

        result["pdl"], result["glu"] = packed_linear.PDL, packed_linear.GLU
        modules = quantized_modules(model)
        result["packed_modules"] = len(modules)
        result["kernel_check"] = check_kernel(modules, args.seed)
    result["model_bytes"] = sum(t.numel() * t.element_size()
                                for t in list(model.parameters()) + list(model.buffers()))

    vocab = model.config.get_text_config().vocab_size
    with torch.inference_mode():  # inputs like the generated tokens, so one compiled specialization serves all
        ids = torch.randint(3, vocab, (1, args.prefill + args.decode - 1),
                            generator=torch.Generator().manual_seed(args.seed)).cuda()
        prompt = ids[:, : args.prefill].clone(memory_format=torch.contiguous_format)
        tokens = [ids[:, i: i + 1].clone(memory_format=torch.contiguous_format)
                  for i in range(args.prefill, ids.shape[1])]
    # cache length rounded up to a multiple of 64, so the attention mask needs no padding for the kernels
    cache = StaticCache(config=model.config, max_cache_len=-(-ids.shape[1] // 64) * 64)
    with torch.inference_mode():
        forward(model, cache, prompt)  # allocate the cache storage (static addresses) outside compilation
    step = model
    if args.mode == "compiled":
        step = torch.compile(model, fullgraph=not args.no_fullgraph, dynamic=False, mode="reduce-overhead")
    torch.cuda.reset_peak_memory_stats()
    result["phases"] = {}
    for phase in PHASES:
        runs = []
        for i in range(-2 - args.warmup, args.samples):  # 2 windows to compile and capture, then warmup
            stance = "fail_on_recompile" if i >= 0 and args.mode == "compiled" else "default"
            with torch.compiler.set_stance(stance):
                r = window(model, step, cache, prompt, tokens, phase, args.decode)
            if i >= 0:
                runs.append(r)
        result["phases"][phase] = summarize(phase, runs, args.prefill, args.decode)
        print(f"{result['method']} {args.mode} {args.attention} {phase}: "
              f"{result['phases'][phase]['tokens_per_s']:.2f} tokens/s", flush=True)
    if args.mode == "compiled":  # a skipped CUDA graph would leave the decode steps without graph replay
        from torch._dynamo.utils import counters

        result["cudagraph_skips"] = counters["inductor"]["cudagraph_skips"]
    if args.profile:
        stance = "fail_on_recompile" if args.mode == "compiled" else "default"
        with torch.compiler.set_stance(stance):
            result["profile"] = profile_decode(model, step, cache, prompt, tokens, min(args.profile, len(tokens)))
        prof = result["profile"]
        print(f"profile: {prof['wall_us_per_step']:.1f} us/step wall, {prof['kernel_us_total']:.1f} us in "
              f"{prof['kernels_per_step']:.0f} kernels: " +
              ", ".join(f"{k} {v:.1f}" for k, v in prof["kernel_us_per_step"].items()), flush=True)
    props = torch.cuda.get_device_properties(0)
    result.update({
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "hardware": {"name": props.name, "compute_capability": [props.major, props.minor],
                     "sm_count": props.multi_processor_count, "memory_bytes": props.total_memory},
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "transformers": version("transformers"), "cuda": torch.version.cuda},
    })
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
