"""COCO val2017 calibration cache: chat-templated (image, instruction, caption) samples -> FP16 input embeds + 3-axis positions.

Cache layout (in --out): `<split>-NNNNN.safetensors` shards with keys `<i>.embeds` (FP16 [S, H]) and `<i>.pos`
(int64 [3, S]), i = sample index within the split, plus `manifest.json`.
"""
import argparse
import hashlib
import itertools
import json
import os
import sys
import time
import traceback

import torch
import transformers
from PIL import Image
from safetensors.torch import load_file, save_file
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

from sphquant.fileio import atomic_write, log, read_json, receipt, sha256_file, write_json
from sphquant.qwen35_targets import MODEL_REV

FORMAT = "sphquant-calib-cache-v1"
SHARD_SAMPLES = 256
EMPTY_THINK = "<think>\n\n</think>\n\n"
TURN_END = "<|im_end|>\n"


def select(captions, n_train, n_val, seed):
    """Seeded distinct image ids -> {split: [(image_id, file_name, caption_id, caption)]}.

    The caption of an image is its annotation with the lowest id (surrounding whitespace stripped).
    """
    files = {im["id"]: im["file_name"] for im in captions["images"]}
    first = {}
    for ann in captions["annotations"]:
        if ann["image_id"] not in first or ann["id"] < first[ann["image_id"]]["id"]:
            first[ann["image_id"]] = ann
    ids = sorted(first)
    if len(ids) < n_train + n_val:
        raise ValueError(f"{len(ids)} captioned images < {n_train} + {n_val}")
    perm = torch.randperm(len(ids), generator=torch.Generator().manual_seed(seed))[: n_train + n_val]
    rows = [(ids[i], files[ids[i]], first[ids[i]]["id"], first[ids[i]]["caption"].strip()) for i in perm.tolist()]
    return {"train": rows[:n_train], "val": rows[n_train:]}


def messages(image, instruction, caption):
    return [
        {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": instruction}]},
        {"role": "assistant", "content": [{"type": "text", "text": caption}]},
    ]


def tokenize(processor, msgs):
    """(rendered text, processor outputs); the text must carry the empty think block and end the assistant turn."""
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    if EMPTY_THINK not in text or not text.endswith(TURN_END):
        raise ValueError(f"unexpected chat template rendering: {text[-200:]!r}")
    inputs = processor.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False, return_dict=True,
                                           return_tensors="pt")
    return text, inputs


@torch.no_grad()
def encode(model, inputs, device):
    """One sample -> (embeds FP16 [S, H], pos int64 [3, S]) on CPU; image features scattered into the token embeds."""
    mm = model.model
    ids = inputs["input_ids"].to(device)
    grid = inputs["image_grid_thw"].to(device)
    feats = torch.cat(mm.get_image_features(inputs["pixel_values"].to(device), grid, return_dict=True).pooler_output)
    embeds = mm.get_input_embeddings()(ids)
    mask = ids == model.config.image_token_id
    if int(mask.sum()) != feats.shape[0]:
        raise ValueError(f"{int(mask.sum())} image tokens but {feats.shape[0]} image features")
    embeds = embeds.masked_scatter(mask.unsqueeze(-1), feats.to(embeds.dtype))
    pos, _ = mm.get_rope_index(ids, inputs["mm_token_type_ids"].to(device), image_grid_thw=grid)
    if not torch.isfinite(embeds).all():
        raise ValueError("non-finite input embeds")
    return embeds[0].cpu(), pos[:, 0].cpu()


def write_shard(path, samples, first):
    tensors = {}
    for i, (embeds, pos) in enumerate(samples, first):
        if embeds.dtype != torch.float16 or pos.dtype != torch.int64 or pos.shape != (3, embeds.shape[0]):
            raise ValueError(f"bad sample {i}: {embeds.dtype} {tuple(embeds.shape)} {pos.dtype} {tuple(pos.shape)}")
        tensors[f"{i}.embeds"] = embeds.contiguous()
        tensors[f"{i}.pos"] = pos.contiguous()
    atomic_write(path, lambda tmp: save_file(tensors, tmp, metadata={"format": "pt"}))
    return sha256_file(path)


def write_cache(out_dir, splits, meta):
    """Write {split: iterable of (embeds, pos)} as shards plus manifest.json (meta + lengths + shard checksums)."""
    os.makedirs(out_dir, exist_ok=True)
    index = {}
    for name, samples in splits.items():
        lengths, shards = [], []
        for k, chunk in enumerate(itertools.batched(samples, SHARD_SAMPLES)):
            file = f"{name}-{k:05d}.safetensors"
            sha = write_shard(os.path.join(out_dir, file), chunk, len(lengths))
            shards.append({"file": file, "first": len(lengths), "count": len(chunk), "sha256": sha})
            lengths += [embeds.shape[0] for embeds, _ in chunk]
        index[name] = {"lengths": lengths, "shards": shards}
    manifest = {**meta, "format": FORMAT, "splits": index}
    write_json(os.path.join(out_dir, "manifest.json"), manifest)
    return manifest


def load_split(cache_dir, split, device="cpu", limit=None):
    """[(embeds FP16 [S, H], pos int64 [3, S])] of one split (first `limit` samples), shard checksums verified."""
    info = read_json(os.path.join(cache_dir, "manifest.json"))["splits"][split]
    n = len(info["lengths"]) if limit is None else min(limit, len(info["lengths"]))
    out = []
    for shard in info["shards"]:
        if shard["first"] >= n:
            break
        path = os.path.join(cache_dir, shard["file"])
        if sha256_file(path) != shard["sha256"]:
            raise ValueError(f"checksum mismatch: {path}")
        tensors = load_file(path, device=str(device))
        for i in range(shard["first"], min(shard["first"] + shard["count"], n)):
            embeds, pos = tensors[f"{i}.embeds"], tensors[f"{i}.pos"]
            if embeds.shape[0] != info["lengths"][i] or pos.shape != (3, embeds.shape[0]):
                raise ValueError(f"sample {i} of {split} does not match the manifest")
            out.append((embeds, pos))
    return out


def load_processor(model_dir):
    return AutoProcessor.from_pretrained(model_dir)


def processor_settings(processor):
    ip = processor.image_processor
    return {
        "processor": type(processor).__name__,
        "image_processor": type(ip).__name__,
        "tokenizer": type(processor.tokenizer).__name__,
        "image_processor_config": ip.to_dict(),
        "chat_template_sha256": hashlib.sha256(processor.chat_template.encode()).hexdigest(),
    }


def build(args):
    t0 = time.time()
    device = torch.device(args.device)
    with open(args.captions, encoding="utf-8") as f:
        selection = select(json.load(f), args.train, args.val, args.seed)
    processor = load_processor(args.model)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(args.model, dtype=torch.float16).eval()
    model.model.visual.to(device)
    model.model.language_model.embed_tokens.to(device)
    example = {}  # first rendered prompt, filled while the shards are written

    def samples(rows):
        for k, (_, file_name, _, caption) in enumerate(rows):
            with Image.open(os.path.join(args.images, file_name)) as im:
                image = im.convert("RGB")
            text, inputs = tokenize(processor, messages(image, args.instruction, caption))
            example.setdefault("text", text)
            if k % 256 == 0:
                log(f"encoded {k}/{len(rows)}")
            yield encode(model, inputs, device)

    meta = {
        "example": example,
        "seed": args.seed,
        "instruction": args.instruction,
        "model_rev": MODEL_REV,
        "model_config_sha256": sha256_file(os.path.join(args.model, "config.json")),
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "processor": processor_settings(processor),
        "selection": {name: {"image_ids": [r[0] for r in rows], "caption_ids": [r[2] for r in rows],
                             "captions": [r[3] for r in rows]} for name, rows in selection.items()},
    }
    manifest = write_cache(args.out, {name: samples(rows) for name, rows in selection.items()}, meta)
    lengths = [n for split in manifest["splits"].values() for n in split["lengths"]]
    return {"ok": True, "out": args.out, "train": args.train, "val": args.val, "tokens": sum(lengths),
            "max_len": max(lengths), "seconds": round(time.time() - t0, 1)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", required=True)
    p.add_argument("--images", required=True, help="COCO val2017 image directory")
    p.add_argument("--captions", required=True, help="captions_val2017.json")
    p.add_argument("--out", required=True)
    p.add_argument("--train", type=int, default=4096)
    p.add_argument("--val", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--instruction", default="Describe this image in one sentence.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args(argv)
    try:
        out = build(args)
    except Exception as e:
        traceback.print_exc()
        receipt({"ok": False, "error": f"{type(e).__name__}: {e}"})
        return 1
    receipt(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
