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
<summary><strong>Qwen3.5-9B with this repository</strong></summary>

Qwen3.5-9B quantized with `scripts/calibrate_w2.sh` (default settings) and evaluated with `scripts/benchmark.sh --limit 100`: the same 100 seeded questions per dataset for both weights, thinking off, greedy decoding.

| Qwen3.5-9B | RealWorldQA | ScienceQA | TextVQA |
|---|---|---|---|
| BF16 | 70.0 | 98.0 | 80.2 |
| SPHQuant W2 | 62.0 | 81.0 | 79.0 |

Prompts and answer parsing differ from the harness behind the paper's table, and 100 questions per dataset leave a wide margin of error, so these numbers are not directly comparable with it.

</details>

## ⚙️ Requirements

- Linux with one NVIDIA GPU, a CUDA 13 driver and Python 3.12.
- The CUDA 13 toolkit (`nvcc`) for the decode-throughput test. Each script installs its Python dependencies into `.venv`.

## 🔧 Reproduction

```bash
# 1. W2 calibration of Qwen3.5-9B (downloads the model and COCO; rerun to resume)
bash scripts/calibrate_w2.sh --work-dir /path/to/work

# 2. RealWorldQA, ScienceQA and TextVQA: BF16 baseline and SPHQuant W2
bash scripts/benchmark.sh --model /path/to/work/models/Qwen3.5-9B --weights-id bf16
bash scripts/benchmark.sh --model /path/to/work/w2-dense --weights-id w2

# 3. Decode throughput: dense FP16 vs the W2 kernel (Qwen3-8B with random weights)
bash scripts/throughput.sh
```

Results go to `outputs/eval/<dataset>/summary.md` and `outputs/throughput/summary.md`. Each script lists its options with `--help`.

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
