"""Layer-wise SPHQuant calibration of the Qwen3.5 text decoder.

Per layer: the teacher (original layer, FP16) maps the teacher stream to the targets; the student (the layer in FP32
with its targets replaced by SPHLinear) is trained with Adam on the student stream against those targets; the best
validation state is exported, and the student stream is propagated through the FP16 layer rebuilt from the export form.
"""
import argparse
import copy
import hashlib
import json
import math
import os
import sys
import time
import traceback
from collections import defaultdict
from contextlib import contextmanager

import torch
import torch.nn.functional as F
import transformers
from torch import nn
from transformers import AutoConfig, Qwen3_5ForConditionalGeneration

from sphquant.coco_data import load_split
from sphquant.fileio import atomic_write, config_hash, log, read_json, receipt, sha256_file, write_json
from sphquant.qwen35_targets import layer_call, layer_targets, layer_types, module_name, rotary, text_config, text_layers
from sphquant.sph_linear import SPHLinear, decode_export

FORMAT = "sphquant-calib-v1"
MIN_RHO_GRAD_FRACTION = 0.9
PILOT_MAX_HOURS = 24.0
STEP_PHASES = ("materialize", "forward", "backward", "optimizer")


def parse_layers(text):
    """'A-B' or 'A' -> (A, B)."""
    a, _, b = text.partition("-")
    first, last = int(a), int(b or a)
    if not 0 <= first <= last:
        raise ValueError(f"bad layer range {text!r}")
    return first, last


def derived_seed(*parts):
    return int.from_bytes(hashlib.sha256("/".join(map(str, parts)).encode()).digest()[:8], "little")


class Timer:
    """Accumulated wall time per phase (synchronizing CUDA around each phase)."""

    def __init__(self, device):
        self.cuda = device.type == "cuda"
        self.times = defaultdict(float)

    @contextmanager
    def __call__(self, phase):
        if self.cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        yield
        if self.cuda:
            torch.cuda.synchronize()
        self.times[phase] += time.perf_counter() - t0


class StepLinear(nn.Module):
    """Stands in for a target nn.Linear of the student: F.linear with a weight materialized once per step."""

    def __init__(self, sph):
        super().__init__()
        self.sph = sph
        self.weight = None

    def forward(self, x):
        return F.linear(x, self.weight, self.sph.bias)


def set_weights(wrappers, grad):
    """Materialize every SPHLinear once; the wrappers get detached leaves. Returns the graph-carrying weights."""
    full = []
    for w in wrappers.values():
        weight = w.sph.materialize()
        w.weight = weight.detach().requires_grad_(grad)
        full.append(weight)
    return full


def length_groups(lengths, idx, size):
    """Indices idx split into same-length chunks of at most `size`, by ascending length, order kept within a length."""
    by_len = defaultdict(list)
    for i in idx:
        by_len[lengths[i]].append(i)
    return [same[k:k + size] for n in sorted(by_len) for same in [by_len[n]] for k in range(0, len(same), size)]


def epoch_steps(lengths, batch, seed):
    """Optimizer steps of one epoch: samples shuffled within each length, laid out by length, cut into steps of
    `batch` samples, then the list of steps shuffled."""
    g = torch.Generator().manual_seed(seed)
    order = []
    for n in sorted(set(lengths)):
        same = [i for i, m in enumerate(lengths) if m == n]
        order += [same[j] for j in torch.randperm(len(same), generator=g).tolist()]
    steps = [order[k:k + batch] for k in range(0, len(order), batch)]
    return [steps[j] for j in torch.randperm(len(steps), generator=g).tolist()]


def stack(xs, idx, dtype):
    return torch.stack([xs[i] for i in idx]).to(dtype)


@torch.no_grad()
def run_layer(model, layer, layer_type, xs, pos, groups, dtype):
    """Every sample of a stream through `layer` in `dtype`; returns the new stream (FP16 tensors)."""
    out = [None] * len(xs)
    for g in groups:
        x = stack(xs, g, dtype)
        y = layer_call(layer, layer_type, x, rotary(model, x, torch.stack([pos[i] for i in g], 1)))
        if not torch.isfinite(y).all():
            raise RuntimeError(f"non-finite layer output ({dtype})")
        for k, i in enumerate(g):
            out[i] = y[k].to(torch.float16)
    return out


def build_student(layer, layer_type, layer_idx, args, device):
    """FP32 copy of the layer (frozen) whose targets are StepLinear(SPHLinear.from_dense(original weight))."""
    gen = torch.Generator().manual_seed(derived_seed(args.seed, layer_idx, "kmeans"))
    student = copy.deepcopy(layer).to(device=device, dtype=torch.float32).requires_grad_(False)
    wrappers = {}
    for rel, lin in layer_targets(student, layer_type).items():
        sph = SPHLinear.from_dense(lin.weight, lin.bias, gen, args.kmeans_samples, args.kmeans_iters, args.grid)
        wrappers[rel] = StepLinear(sph)
        student.set_submodule(rel, wrappers[rel])
    return student, wrappers


def rebuild_layer(layer, layer_type, layer_idx, states, device):
    """FP16 copy of the layer whose target weights are the decoded export form."""
    out = copy.deepcopy(layer).to(device=device, dtype=torch.float16)
    with torch.no_grad():
        for rel, lin in layer_targets(out, layer_type).items():
            state = {k: v.to(device) for k, v in states[module_name(layer_idx, rel)].items()}
            lin.weight.copy_(decode_export(state).to(torch.float16))
    return out


@torch.no_grad()
def validate(model, student, layer_type, wrappers, xs, targets, pos, groups):
    """Sample-weighted mean MSE of the FP32 student against the targets."""
    set_weights(wrappers, grad=False)
    total = 0.0
    for g in groups:
        x = stack(xs, g, torch.float32)
        y = layer_call(student, layer_type, x, rotary(model, x, torch.stack([pos[i] for i in g], 1)))
        total += (y - stack(targets, g, torch.float32)).square().mean(dim=(1, 2)).double().sum().item()
    if not math.isfinite(total):
        raise RuntimeError("non-finite validation loss")
    return total / len(xs)


def calibrate_layer(args, model, layer, layer_type, layer_idx, x_q, targets, pos, lengths, groups, timer, device):
    """Train one layer; returns (export states by full module name, metrics) for the best validation state."""
    with timer("init"):
        student, wrappers = build_student(layer, layer_type, layer_idx, args, device)
    sphs = [w.sph for w in wrappers.values()]
    opt = torch.optim.Adam([{"params": [m.rho for m in sphs]}, {"params": [m.phi for m in sphs]}], lr=args.lr)

    def val():
        with timer("validation"):
            return validate(model, student, layer_type, wrappers, x_q["val"], targets["val"], pos["val"], groups["val"])

    def snapshot():
        return [(m.rho.detach().clone(), m.phi.detach().clone()) for m in sphs]

    with timer("validation"):  # unquantized FP32 layer on the same quantized inputs: the inherited error floor
        ref = copy.deepcopy(layer).to(device=device, dtype=torch.float32)
        ref_val = validate(model, ref, layer_type, {}, x_q["val"], targets["val"], pos["val"], groups["val"])
        del ref
    init_val = val()
    best_val, best_epoch, best = init_val, 0, snapshot()
    val_history, train_history, grad_fraction, steps = [init_val], [], None, 0
    for epoch in range(1, args.epochs + 1):
        plan, epoch_loss = epoch_steps(lengths["train"], args.batch, derived_seed(args.seed, layer_idx, epoch)), 0.0
        for step in plan:
            with timer("materialize"):
                full = set_weights(wrappers, grad=True)
            step_loss = 0.0
            for sub in length_groups(lengths["train"], step, args.batch):
                x = stack(x_q["train"], sub, torch.float32)
                p = torch.stack([pos["train"][i] for i in sub], 1)
                with timer("forward"):
                    y = layer_call(student, layer_type, x, rotary(model, x, p))
                    loss = F.mse_loss(y, stack(targets["train"], sub, torch.float32)) * (len(sub) / len(step))
                with timer("backward"):
                    loss.backward()
                step_loss += loss.item()
            with timer("materialize"):
                torch.autograd.backward(full, [w.weight.grad for w in wrappers.values()])
            for w in wrappers.values():
                w.weight = None
            if grad_fraction is None:
                nonzero = sum(int(torch.count_nonzero(m.rho.grad)) for m in sphs)
                grad_fraction = nonzero / sum(m.rho.numel() for m in sphs)
                if grad_fraction <= MIN_RHO_GRAD_FRACTION:
                    raise RuntimeError(f"layer {layer_idx}: only {grad_fraction:.3f} of rho gradients are nonzero")
            if not math.isfinite(step_loss) or not all(
                    bool(torch.isfinite(m.rho.grad).all() and torch.isfinite(m.phi.grad).all()) for m in sphs):
                raise RuntimeError(f"layer {layer_idx} epoch {epoch}: non-finite loss or gradient")
            with timer("optimizer"):
                opt.step()
                opt.zero_grad(set_to_none=True)
            epoch_loss += step_loss
            steps += 1
        train_history.append(epoch_loss / len(plan))
        val_history.append(val())
        if val_history[-1] < best_val:
            best_val, best_epoch, best = val_history[-1], epoch, snapshot()
        log(f"layer {layer_idx} epoch {epoch}: train {train_history[-1]:.4e} val {val_history[-1]:.4e} "
            f"(init {init_val:.4e}, best epoch {best_epoch})")
    with torch.no_grad():
        for m, (rho, phi) in zip(sphs, best):
            m.rho.copy_(rho)
            m.phi.copy_(phi)
    states = {module_name(layer_idx, rel): w.sph.export_state() for rel, w in wrappers.items()}
    metrics = {
        "layer": layer_idx, "layer_type": layer_type, "init_val": init_val, "best_val": best_val,
        "ref_val": ref_val, "best_epoch": best_epoch, "val_history": val_history, "train_loss": train_history,
        "rho_grad_nonzero": grad_fraction, "epochs": args.epochs, "steps": steps,
        "n_train": len(x_q["train"]), "n_val": len(x_q["val"]),
        "kmeans": {rel: w.sph.kmeans_stats for rel, w in wrappers.items()},
    }
    return states, metrics


def layer_paths(out, i):
    return os.path.join(out, f"layer_{i:02d}.pt"), os.path.join(out, f"layer_{i:02d}.done")


def read_layer(out, i):
    """Completed layer record (checksum verified against its done marker), or None without a done marker."""
    pt, done = layer_paths(out, i)
    if not os.path.exists(done):
        return None
    if sha256_file(pt) != read_json(done)["sha256"]:
        raise RuntimeError(f"{pt} does not match its done marker")
    return torch.load(pt, map_location="cpu", weights_only=True)


def save_layer(out, i, record):
    """layer_XX.pt (atomic), one metrics.jsonl line, then the done marker."""
    pt, done = layer_paths(out, i)
    atomic_write(pt, lambda tmp: torch.save(record, tmp))
    with open(os.path.join(out, "metrics.jsonl"), "ab+") as f:
        f.seek(0, os.SEEK_END)
        if f.tell():  # start on a fresh line even if a previous append was cut off
            f.seek(-1, os.SEEK_END)
            if f.read(1) != b"\n":
                f.write(b"\n")
        f.write((json.dumps(record["metrics"], sort_keys=True) + "\n").encode("utf-8"))
        f.flush()
        os.fsync(f.fileno())
    write_json(done, {"sha256": sha256_file(pt), "config_hash": record["config_hash"],
                      "manifest_hash": record["manifest_hash"]})


def gcal_ratio(m):
    """(best - ref) / (init - ref): share of the layer's own quantization error left after training. ref_val is the
    unquantized layer's error on the quantized inputs, i.e. what earlier layers pass on and this layer cannot fix."""
    ref = m.get("ref_val", 0.0)
    own = m["init_val"] - ref
    return (m["best_val"] - ref) / own if own > 0 else 0.0


def check_history(history, gcal_max_ratio, full_run):
    m = history[-1]
    if gcal_max_ratio is not None and gcal_ratio(m) > gcal_max_ratio:
        raise RuntimeError(f"layer {m['layer']}: (best-ref)/(init-ref) validation {gcal_ratio(m):.3f} "
                           f"> --gcal-max-ratio {gcal_max_ratio}")
    if full_run and len(history) > 1 and history[-1]["best_epoch"] == 0 and history[-2]["best_epoch"] == 0:
        raise RuntimeError(f"layers {m['layer'] - 1} and {m['layer']} both kept their initial state")


def run_config(args):
    return {
        "format": FORMAT, "epochs": args.epochs, "batch": args.batch, "lr": args.lr, "seed": args.seed,
        "grid": args.grid, "lr_schedule": args.lr_schedule, "kmeans_samples": args.kmeans_samples,
        "kmeans_iters": args.kmeans_iters,
        "cache_manifest_sha256": sha256_file(os.path.join(args.cache, "manifest.json")),
        "model_config_sha256": sha256_file(os.path.join(args.model, "config.json")),
    }


def calibrate(args):
    """Run (or resume) the calibration; returns (receipt, final streams {'x_fp', 'x_q'})."""
    t_start = time.time()
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.fp32_precision = "ieee"  # FP32 training: no TF32
        torch.backends.cudnn.conv.fp32_precision = "ieee"
    manifest_hash = os.environ.get("SPHQ_MANIFEST_HASH", "")  # optional code-version tag: layers saved under
    # another tag (or another config) are never reused by a resumed run
    config = run_config(args)
    chash = config_hash(config)
    os.makedirs(args.out, exist_ok=True)
    for i in range(parse_layers(args.layers)[1] + 1):  # refuse before RUN.json is rewritten
        done = layer_paths(args.out, i)[1]
        if os.path.exists(done):
            mark = read_json(done)
            if (mark["config_hash"], mark["manifest_hash"]) != (chash, manifest_hash):
                raise RuntimeError(f"layer {i} in {args.out} has config {mark['config_hash'][:12]} / manifest "
                                   f"{mark['manifest_hash']!r}; refusing to reuse it under {chash[:12]} / "
                                   f"{manifest_hash!r}")
    write_json(os.path.join(args.out, "RUN.json"), {
        "args": vars(args), "config": config, "config_hash": chash, "SPHQ_MANIFEST_HASH": manifest_hash,
        "torch": torch.__version__, "transformers": transformers.__version__, "device": str(device),
        "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })

    model = Qwen3_5ForConditionalGeneration.from_pretrained(args.model, dtype=torch.float32).eval()
    model.model.language_model.rotary_emb.to(device)
    layers, types = text_layers(model), layer_types(model)
    first, last = parse_layers(args.layers)
    if last >= len(layers):
        raise ValueError(f"--layers {args.layers}: the model has {len(layers)} layers")
    full_run = first == 0 and last == len(layers) - 1

    data = {s: load_split(args.cache, s, device) for s in ("train", "val")}
    pos = {s: [p for _, p in data[s]] for s in data}
    lengths = {s: [e.shape[0] for e, _ in data[s]] for s in data}
    groups = {s: length_groups(lengths[s], range(len(lengths[s])), args.batch) for s in data}
    x_fp = {s: [e for e, _ in data[s]] for s in data}
    x_q = {s: list(x_fp[s]) for s in data}
    del data

    history, calibrated, reused, stopped = [], [], [], None
    for i in range(last + 1):
        t = types[i]
        rec = read_layer(args.out, i)
        if rec is not None and (rec["config_hash"], rec["manifest_hash"]) != (chash, manifest_hash):
            raise RuntimeError(f"layer {i} in {args.out} has config {rec['config_hash'][:12]} / manifest "
                               f"{rec['manifest_hash']!r}; refusing to reuse it under {chash[:12]} / {manifest_hash!r}")
        if rec is not None and calibrated:
            raise RuntimeError(f"layer {i} has a result although layer {calibrated[0]} was just recalibrated")
        if rec is None and i < first:
            raise RuntimeError(f"layer {i} has no completed result in {args.out} (--layers starts at {first})")
        timer = Timer(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        t_layer = time.time()
        with timer("teacher"):
            teacher = copy.deepcopy(layers[i]).to(device=device, dtype=torch.float16)
            targets = {s: run_layer(model, teacher, t, x_fp[s], pos[s], groups[s], torch.float16) for s in x_fp}
            del teacher
        if rec is None:
            states, metrics = calibrate_layer(args, model, layers[i], t, i, x_q, targets, pos, lengths, groups,
                                              timer, device)
        else:
            states, metrics = rec["modules"], rec["metrics"]
        with timer("propagation"):
            rebuilt = rebuild_layer(layers[i], t, i, states, device)
            x_q = {s: run_layer(model, rebuilt, t, x_q[s], pos[s], groups[s], torch.float16) for s in x_q}
            del rebuilt
        x_fp = targets
        if rec is None:
            metrics["times"] = {**timer.times, "total": time.time() - t_layer}
            metrics["max_memory"] = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
            save_layer(args.out, i, {"layer": i, "layer_type": t, "config_hash": chash,
                                     "manifest_hash": manifest_hash, "modules": states, "metrics": metrics})
            calibrated.append(i)
            log(f"layer {i} done: init {metrics['init_val']:.4e} best {metrics['best_val']:.4e} "
                f"(epoch {metrics['best_epoch']}), {metrics['times']['total']:.1f}s")
        else:
            reused.append(i)
        history.append(metrics)
        check_history(history, args.gcal_max_ratio, full_run)
        if args.stop_after_layer == i:
            stopped = i
            break
    out = {"ok": True, "out": args.out, "config_hash": chash, "calibrated": calibrated, "reused": reused,
           "stopped_after": stopped, "seconds": round(time.time() - t_start, 1)}
    return out, {"x_fp": x_fp, "x_q": x_q}


def project(args):
    """Projected full-run hours from the per-step and per-layer timings recorded in metrics.jsonl, plus pilot checks."""
    latest = {}
    with open(os.path.join(args.out, "metrics.jsonl"), encoding="utf-8") as f:
        for line in f:
            try:
                m = json.loads(line)
            except json.JSONDecodeError:
                log(f"skipping an unparsable metrics line: {line[:80]!r}")
                continue
            latest[m["layer"]] = m
    types = list(text_config(AutoConfig.from_pretrained(args.model)).layer_types)
    n_train = len(read_json(os.path.join(args.cache, "manifest.json"))["splits"]["train"]["lengths"])
    steps = math.ceil(n_train / args.batch)
    per_type = {}
    for t in sorted(set(types)):
        secs = []
        for m in (m for m in latest.values() if m["layer_type"] == t):
            tm = m["times"]
            step = sum(tm[k] for k in STEP_PHASES) / m["steps"]
            fixed = tm["init"] + tm["teacher"] + tm["propagation"]
            secs.append(fixed + args.epochs * steps * step + (args.epochs + 1) * tm["validation"] / (m["epochs"] + 1))
        if not secs:
            raise RuntimeError(f"no measured {t} layer in {args.out}")
        per_type[t] = sum(secs) / len(secs)
    hours = sum(per_type[t] for t in types) / 3600
    zero_grad_ok = all((m["rho_grad_nonzero"] or 0) > MIN_RHO_GRAD_FRACTION for m in latest.values())
    gcal_ok = None if args.gcal_max_ratio is None else all(
        gcal_ratio(m) <= args.gcal_max_ratio for m in latest.values())
    ok = bool(gcal_ok) and zero_grad_ok and hours <= PILOT_MAX_HOURS
    return {
        "ok": ok, "pass": ok, "projected_hours": hours, "max_hours": PILOT_MAX_HOURS, "per_layer_seconds": per_type,
        "epochs": args.epochs, "n_train": n_train, "steps_per_epoch": steps,
        "layer_types": {t: types.count(t) for t in per_type}, "measured_layers": sorted(latest),
        "gcal_max_ratio": args.gcal_max_ratio, "gcal_ok": gcal_ok, "zero_grad_ok": zero_grad_ok,
        "val_ratio": {m["layer"]: m["best_val"] / m["init_val"] for m in latest.values()},
        "gcal_ratio": {m["layer"]: gcal_ratio(m) for m in latest.values()},
    }


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--layers", default="0-31")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--grid", choices=("recompute", "frozen"), default="recompute")
    p.add_argument("--lr-schedule", choices=("constant",), default="constant")
    p.add_argument("--kmeans-samples", type=int, default=65536)
    p.add_argument("--kmeans-iters", type=int, default=20)
    p.add_argument("--stop-after-layer", type=int)
    p.add_argument("--gcal-max-ratio", type=float)
    p.add_argument("--project", action="store_true", help="only print the projected full-run hours")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        out = project(args) if args.project else calibrate(args)[0]
    except Exception as e:
        traceback.print_exc()
        receipt({"ok": False, "error": f"{type(e).__name__}: {e}"})
        return 1
    receipt(out)
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
