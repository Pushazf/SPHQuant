"""Export calibrated layers as a dense BF16 checkpoint directory (decoded targets) and as the packed W2 file."""
import argparse
import os
import shutil
import sys
import traceback

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from transformers import AutoConfig

from sphquant.calibrate import parse_layers, read_layer
from sphquant.fileio import atomic_write, log, receipt, write_json
from sphquant.packing import pack, unpack
from sphquant.qwen35_targets import MODEL_REV, text_config
from sphquant.sph_linear import decode_export
from sphquant.spherical import VEC

PACKED_FORMAT = "sphquant-w2-v1"
DONE = "EXPORT_DONE.json"  # in both output dirs, written last: consumers require it


def collect(calib, first, last):
    """{full module name: export state} of the completed layers first..last, and their common config hash."""
    states, hashes = {}, set()
    for i in range(first, last + 1):
        rec = read_layer(calib, i)
        if rec is None:
            raise RuntimeError(f"layer {i} is not complete in {calib}")
        states.update(rec["modules"])
        hashes.add((rec["config_hash"], rec["manifest_hash"]))
    if len(hashes) != 1:
        raise RuntimeError(f"layers {first}-{last} come from different calibration configs or code: {sorted(hashes)}")
    return states, hashes.pop()[0]


def as_bytes(t):
    return t.contiguous().reshape(-1).view(torch.uint8)


def export_dense(model_dir, out_dir, states):
    """Copy the non-safetensors files and rewrite every shard, replacing only the exported target weights."""
    os.makedirs(out_dir, exist_ok=True)
    shards = []
    for f in sorted(os.listdir(model_dir)):
        src = os.path.join(model_dir, f)
        if not os.path.isfile(src):
            continue
        if f.endswith(".safetensors"):
            shards.append(f)
        else:
            atomic_write(os.path.join(out_dir, f), lambda tmp: shutil.copyfile(src, tmp))
    replaced = {}
    for f in shards:
        with safe_open(os.path.join(model_dir, f), "pt") as s:
            meta = s.metadata()
            tensors = {k: s.get_tensor(k) for k in s.keys()}
        for key, old in tensors.items():
            name = key.removesuffix(".weight")
            if key.endswith(".weight") and name in states:
                new = decode_export(states[name], torch.bfloat16)
                if old.dtype != torch.bfloat16 or new.shape != old.shape:
                    raise RuntimeError(f"{key}: checkpoint {old.dtype} {tuple(old.shape)}, export {tuple(new.shape)}")
                tensors[key] = new
                replaced[name] = f
        dst = os.path.join(out_dir, f)
        atomic_write(dst, lambda tmp: save_file(tensors, tmp, metadata=meta))
        with safe_open(dst, "pt") as s:
            if set(s.keys()) != set(tensors):
                raise RuntimeError(f"{dst}: key set changed")
            for key, t in tensors.items():
                back = s.get_tensor(key)
                if back.dtype != t.dtype or back.shape != t.shape or not torch.equal(as_bytes(back), as_bytes(t)):
                    raise RuntimeError(f"{dst}: {key} does not read back")
        log(f"{f}: {sum(v == f for v in replaced.values())} replaced, {len(tensors)} tensors")
    missing = sorted(set(states) - set(replaced))
    if missing:
        raise RuntimeError(f"{len(missing)} exported modules not found in the checkpoint, e.g. {missing[0]}")
    return replaced


def export_packed(out_dir, states, calib_hash):
    """packed.safetensors (<module>.<field>) and packed_manifest.json; verified by unpacking."""
    os.makedirs(out_dir, exist_ok=True)
    tensors, modules = {}, {}
    for name in sorted(states):
        tensors.update({f"{name}.{field}": t for field, t in pack(states[name]).items()})
        n, kv = states[name]["codes"].shape
        modules[name] = [n, kv * VEC]
    path = os.path.join(out_dir, "packed.safetensors")
    atomic_write(path, lambda tmp: save_file(tensors, tmp, metadata={"format": PACKED_FORMAT}))
    write_json(os.path.join(out_dir, "packed_manifest.json"), {
        "format": PACKED_FORMAT, "modules": modules, "source_model_rev": MODEL_REV, "calib_config_hash": calib_hash})
    back = load_file(path)
    fields = ("packed_codes", "radius_extra", "radius_scales", "radius_mins", "codebook")
    for name, state in states.items():
        got = unpack({field: back[f"{name}.{field}"] for field in fields})
        if any(got[k].dtype != v.dtype or not torch.equal(got[k], v) for k, v in state.items()):
            raise RuntimeError(f"{name}: packed round trip mismatch")
    return modules


def run(args):
    if os.path.realpath(args.dense_out) == os.path.realpath(args.model):
        raise ValueError("--dense-out must differ from --model")
    n_layers = len(text_config(AutoConfig.from_pretrained(args.model)).layer_types)
    first, last = parse_layers(args.layers or f"0-{n_layers - 1}")
    states, calib_hash = collect(args.calib, first, last)
    for d in (args.dense_out, args.packed_out):  # a stale marker must not vouch for a rewrite that gets cut off
        if os.path.exists(os.path.join(d, DONE)):
            os.remove(os.path.join(d, DONE))
    replaced = export_dense(args.model, args.dense_out, states)
    modules = export_packed(args.packed_out, states, calib_hash)
    out = {"ok": True, "layers": [first, last], "modules": len(modules), "replaced": len(replaced),
           "weights": sum(n * k for n, k in modules.values()), "calib_config_hash": calib_hash,
           "dense_out": args.dense_out, "packed_out": args.packed_out}
    for d in (args.dense_out, args.packed_out):  # both outputs were read back and verified above
        write_json(os.path.join(d, DONE), out)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--calib", required=True)
    p.add_argument("--dense-out", required=True)
    p.add_argument("--packed-out", required=True)
    p.add_argument("--layers", help="A-B (default: every layer of the model)")
    args = p.parse_args(argv)
    try:
        out = run(args)
    except Exception as e:
        traceback.print_exc()
        receipt({"ok": False, "error": f"{type(e).__name__}: {e}"})
        return 1
    receipt(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
