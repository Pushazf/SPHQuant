# [NeurIPS 2026] SPHQuant: Efficient extreme low bit weight quantization for Vision-Language Models

[Kewei Zhang](https://github.com/Pushazf), [Zheng Chen](https://zheng-chen.cn/), [Haotong Qin](https://htqin.github.io/), and [Yulun Zhang](https://yulunzhang.com/), "SPHQuant: Efficient extreme low bit weight quantization for Vision-Language Models", NeurIPS 2026

<div>
<a href="https://visitor-badge.laobi.icu/badge?page_id=Pushazf/SPHQuant" style="text-decoration: none;">
<img src="https://visitor-badge.laobi.icu/badge?page_id=Pushazf/SPHQuant" alt="Visitors">
</a>
<a href="https://github.com/Pushazf/SPHQuant/stargazers" style="text-decoration: none;">
<img src="https://img.shields.io/github/stars/Pushazf/SPHQuant?style=social&cacheSeconds=3600" alt="GitHub stars">
</a>
<a href="https://arxiv.org/abs/2609.24875" style="text-decoration: none;">
<img src="https://img.shields.io/badge/arXiv-2609.24875-b31b1b?style=flat" alt="arXiv: 2609.24875">
</a>
</div>

---

> **Abstract:** Recent foundation models are moving toward native multimodal Vision-Language Models (VLMs), making VLMs a central form of next-generation foundation models. However, their large language backbones make edge deployment difficult due to high memory footprint and memory-bound autoregressive decoding. Weight-only post-training quantization is a practical solution, but pushing VLMs to extreme low bit-widths remains challenging: existing rotation-free methods suffer from outliers at 2–3 bits, while rotation-based methods improve accuracy at the cost of additional runtime overhead. We propose SPHQuant, a rotation-free spherical weight-only quantization framework for VLMs. Instead of quantizing weights directly in Cartesian coordinates, SPHQuant decomposes each 8D weight vector into coordinate signs, radius, and a positive unit direction. This representation isolates outlier magnitude into the radius while keeping directions bounded and statistically regular. Based on this insight, SPHQuant allocates extra precision to the radius to mitigate accuracy degradation induced by outliers. It further uses a compact positive-direction codebook and fine-tunes codebook entries through angular parameterization to preserve the unit-sphere constraint. We also design a hardware-friendly GEMV kernel that keeps the direction codebook small enough for shared-memory lookup and packs radial bits efficiently. Experiments show that SPHQuant matches the performance of state-of-the-art extreme low-bit quantization methods while improving decode throughput over QTIP by **30.3%** on RTX A6000.

---

### Overview

![Overview of SPHQuant: spherical weight quantization and packed extra radius bits](assets/overview.png)

### Results

SPHQuant achieves competitive accuracy across three recent VLMs at extreme low bit-widths.

<details open>
<summary><strong>Accuracy on VLM Benchmarks</strong></summary>

<p align="center">
  <a href="assets/main_results.png">
    <img width="100%" src="assets/main_results_table.png" alt="Standard VLM benchmark results for Qwen3-VL-8B-Instruct, Qwen3.5-9B, and Gemma4-26B-A4B. SPHQuant rows are shaded gray.">
  </a>
</p>

</details>

<details open>
<summary><strong>Reproduction with this repository</strong></summary>

Qwen3.5-9B quantized with `scripts/calibrate_w2.sh` (default settings) and evaluated with `scripts/benchmark.sh --limit 100` on one NVIDIA B300: the same 100 seeded questions per dataset for both weights, thinking off, greedy decoding.

| Qwen3.5-9B | RealWorldQA | ScienceQA | TextVQA |
|---|---|---|---|
| BF16 | 70.0 | 98.0 | 80.2 |
| SPHQuant W2 | 62.0 | 81.0 | 79.0 |

Prompts and answer parsing differ from the harness behind the paper's table, and 100 questions per dataset leave a wide margin of error, so these numbers are not directly comparable with it.

</details>

## ⚙️ Requirements

- Linux with one NVIDIA GPU and a CUDA 13 driver. Our runs used one NVIDIA B300; peak GPU memory during calibration was 44 GB.
- Python 3.12, `curl`, and about 80 GB of free disk for the model, COCO, the calibration cache and the exported checkpoints.
- The CUDA 13 toolkit (`nvcc`) for the decode-throughput test, which builds its kernels.

The scripts create a virtual environment in `.venv` (change it with `--venv`). Calibration and the throughput test install `requirements.txt` (PyTorch 2.14.1, Transformers 5.19.0); the benchmark adds SGLang 0.5.23, which pins the same versions, so all three share one environment.

## 🔧 W2 Calibration

One command quantizes the text decoder of [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) to W2 (8D vectors: 8 sign bits, a 6-bit index into a 64-entry direction codebook and a 4-bit radius, 2.25 bits per weight):

```bash
bash scripts/calibrate_w2.sh --work-dir /path/to/work
```

| Step | Output in `--work-dir` | Time on one B300 |
|---|---|---|
| 1. Environment from `requirements.txt` | `.venv` in the repository | a few minutes |
| 2. Qwen3.5-9B (pinned revision) and COCO val2017 images and captions | `models/Qwen3.5-9B`, `data/coco` | depends on the network (~21 GB) |
| 3. Calibration cache: 4,096 training and 128 validation COCO samples | `calib-cache` (~10 GB) | ~5 min |
| 4. Layer-wise calibration of all 32 decoder layers, 20 epochs each | `calib-w2` | ~10.5 h |
| 5. Export | `w2-dense` (decoded BF16 checkpoint for the benchmark), `w2-packed` (packed W2 weights, 2.5 bits per weight) | ~1 min |

Running the script again resumes: finished steps are skipped and an interrupted calibration continues from its last finished layer. Options: `--layers A-B` (default `0-31`), `--epochs N` (default `20`), `--venv DIR`, `--python PATH`; see `bash scripts/calibrate_w2.sh --help`.

## 📊 Benchmark

The benchmark serves a model with SGLang's own server (`python -m sglang.launch_server`) and scores the answers with the client in `benchmark/`. `scripts/benchmark.sh` installs SGLang, downloads every dataset at a pinned revision (checking its sha256), starts and stops the server, runs the evaluation, and writes a summary per dataset.

```bash
# BF16 baseline (the model downloaded by calibrate_w2.sh)
bash scripts/benchmark.sh --model /path/to/work/models/Qwen3.5-9B --weights-id bf16
# SPHQuant W2, decoded to BF16
bash scripts/benchmark.sh --model /path/to/work/w2-dense --weights-id w2
```

Both runs write to `outputs/eval/<dataset>/`:

- `<weights-id>_think-<off|on>.jsonl`: one record per question (the model's output, the extracted answer, its score, truncation and repetition-loop flags). A rerun resumes from these files.
- `summary.md`: accuracy, truncation, loops and unparsable answers per weights id, plus the paired difference to `bf16` with a bootstrap 95% confidence interval.

| Dataset (`--datasets` name) | Source | Questions | Metric |
|---|---|---|---|
| RealWorldQA (`realworldqa`) | `xai-org/RealworldQA`, test | 765 | accuracy (multiple choice and short answers) |
| ScienceQA (`scienceqa`) | `lmms-lab-encoder/ScienceQA`, ScienceQA-IMG test | 2,017 | accuracy |
| TextVQA (`textvqa`) | `lmms-lab-encoder/textvqa`, validation | 5,000 | VQA accuracy |

As in the paper, thinking is off and generation is greedy (temperature 0), with at most 2,048 output tokens. Answers are parsed by rules; there is no LLM judge and no random fallback, so an unparsable answer counts as wrong.

Useful options (`bash scripts/benchmark.sh --help` lists all):

- `--datasets realworldqa,scienceqa` evaluates only some datasets.
- `--limit 100` evaluates a seeded subset per dataset, stratified by subject or question type; both weights get the same questions.
- `--think on` (or `off,on`) also evaluates with thinking, with at most 16,384 output tokens.
- `--out`, `--data`, `--port` and `--concurrency` set where results and datasets go, the server port and the number of parallel requests. Extra server flags go in `SGLANG_ARGS` (default `--attention-backend triton`).

If the SGLang scheduler fails at startup with `GLIBCXX_3.4.29 not found`, the system `libstdc++` is too old for SGLang's compiled radix-cache backend. Select its Python implementation instead: `SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND=python bash scripts/benchmark.sh ...`.

## 🚀 Decode Throughput

`throughput/` measures batch-1 decode speed of a dense FP16 model and of the same model with SPHQuant W2 weights, whose decoder projections run the fused GEMV kernel in `cuda_kernel/`. Only the shapes matter for speed, so by default the model is [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) built from its config (`throughput/configs/qwen3-8b.json`) with random weights and random W2 codes; nothing is downloaded:

```bash
bash scripts/throughput.sh
```

The script builds the kernels for GPU 0's architecture and then:

1. times every decoder projection on its own, dense against W2, with weights read from memory as in a real decode step (`outputs/throughput/gemv.json`);
2. builds each model, checks every W2 module against its weight decoded in PyTorch, and times it with a 256-token random prompt and 512 generated tokens, using a static KV cache. The single-token forward is compiled with `torch.compile(fullgraph=True, mode="reduce-overhead")`, so each decode step replays a CUDA graph. In both models q/k/v and gate/up are fused into one projection each, and the attention of every decode step runs the split-KV kernel in `cuda_kernel/decode_attention.cu`; the prompt uses PyTorch's SDPA. With `--attention sdpa` the decode steps use SDPA as well, as transformers does by default; at batch 1 that is PyTorch's memory-efficient kernel, which uses one block per query head and leaves most SMs of a large GPU idle.

Each model runs in its own process with untimed warmup, and the median of 5 windows is reported as decode tokens/s (single-token forwards after the prompt), end-to-end tokens/s (prompt plus greedy generation) and time to first token. A profile of 64 decode steps shows where the GPU time of a step goes (W2 GEMV, dense matmul, attention, fused elementwise kernels). Results go to `outputs/throughput/` (one JSON per model and `summary.md` with the W2 speedup). Options: `--attention kernel,sdpa`, `--modes compiled,eager`, `--prefill N`, `--decode N`, `--samples N`, `--dtype bfloat16`, `--config FILE` for another architecture; see `bash scripts/throughput.sh --help`. For comparison, `SPH_PDL=0` launches the W2 GEMV without programmatic dependent launch, and `SPH_GLU=0` computes silu(gate) * up after the fused gate/up GEMV instead of inside it.

The packed W2 weights take 2.5 bits per weight (the 2.25 bits of codes plus an FP16 step and minimum per 128 weights). At load time they are rearranged into the layout the kernel reads, with the same size; prompts of more than 8 tokens multiply by the dequantized FP16 weight instead.

## 📁 Repository Structure

```
sphquant/      calibration: COCO cache, spherical quantizer, layer-wise calibration, export, packing
cuda_kernel/   fused W2 GEMV kernel (sph_gemv.cu), decode attention kernel, build script and PyTorch operators
benchmark/     RealWorldQA, ScienceQA and TextVQA loaders, prompts and scoring, SGLang client, summaries
throughput/    batch-1 decode throughput: random-weight models, per-projection and whole-model timings, summary
scripts/       calibrate_w2.sh, benchmark.sh, throughput.sh
```

## 🔖 To Do

- [x] Add the paper and arXiv links.
- [x] Release calibration, quantization, and CUDA inference code.
- [ ] Release 2-bit quantized model weights.
- [x] Release evaluation scripts and reproduction instructions.

## 📄 License

This project is released under the [Apache 2.0 license](LICENSE).

## <a name="citation"></a>📎 Citation

If you find this work useful, please cite [our paper](https://arxiv.org/abs/2609.24875).

```bibtex
@inproceedings{zhang2026sphquant,
  title={{SPHQuant}: Efficient extreme low bit weight quantization for {Vision-Language Models}},
  author={Zhang, Kewei and Chen, Zheng and Qin, Haotong and Zhang, Yulun},
  booktitle={NeurIPS},
  year={2026},
  url={https://arxiv.org/abs/2609.24875}
}
```
