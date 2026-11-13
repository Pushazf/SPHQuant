#!/usr/bin/env bash
# Batch-1 decode throughput: dense FP16 weights vs SPHQuant W2 through the fused GEMV kernel (Linux, one NVIDIA GPU,
# CUDA 13 toolkit with nvcc on PATH). By default the model is Qwen3-8B with random weights (no download): only its
# shapes matter for speed, and the W2 codes are random as well.
#
#   bash scripts/throughput.sh [options]
#
# 1. builds the kernels in cuda_kernel/ (W2 GEMV and dequantization, split-KV decode attention) into
#    BUILD/sph_gemv.so;
# 2. times every projection shape on its own (python -m throughput.gemv_bench), unless --no-gemv;
# 3. measures every method (dense, w2) in every mode and with every attention, one configuration per process
#    (python -m throughput.decode_bench). Before timing W2, every packed module is checked against its weight decoded
#    in PyTorch; a failed check stops the run;
# 4. writes OUT/<method>_<mode>.json (OUT/<method>_<mode>_sdpa.json with --attention sdpa) and the table
#    OUT/summary.md.
#
# Options:
#   --config FILE      architecture with random weights (default throughput/configs/qwen3-8b.json)
#   --modes LIST       compiled, eager or compiled,eager (default compiled)
#   --attention LIST   attention of the decode steps, for dense and W2 alike: kernel (the split-KV kernel, default),
#                      sdpa (PyTorch's SDPA as transformers calls it) or kernel,sdpa
#   --prefill N        prompt tokens (default 256)
#   --decode N         generated tokens (default 512)
#   --warmup N         untimed windows after compilation (default 3)
#   --samples N        timed windows; the median is reported (default 5)
#   --dtype TYPE       float16 or bfloat16 activations (default float16)
#   --profile N        also record the GPU kernels of N decode steps (default 64; 0: off)
#   --no-gemv          skip the per-projection timings
#   --out DIR          results (default: outputs/throughput in the repository)
#   --build DIR        where sph_gemv.so goes (default: build in the repository)
#   --arch sm_XY       GPU architecture for the kernel (default: that of GPU 0)
#   --venv DIR         virtual environment, created when missing (default: .venv in the repository)
#   --python PATH      interpreter that creates the environment (default: python3; 3.12 recommended)
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CONFIG=$ROOT/throughput/configs/qwen3-8b.json MODES=compiled ATTENTION=kernel PREFILL=256 DECODE=512 WARMUP=3
SAMPLES=5 DTYPE=float16 PROFILE=64 GEMV=1 OUT=$ROOT/outputs/throughput BUILD=$ROOT/build ARCH= VENV=$ROOT/.venv
PYTHON=python3

usage() { sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed '$d; s/^# \{0,1\}//'; exit "${1:-0}"; }
while [ $# -gt 0 ]; do
  case $1 in
    --config) CONFIG=$2; shift 2 ;;
    --modes) MODES=$2; shift 2 ;;
    --attention) ATTENTION=$2; shift 2 ;;
    --prefill) PREFILL=$2; shift 2 ;;
    --decode) DECODE=$2; shift 2 ;;
    --warmup) WARMUP=$2; shift 2 ;;
    --samples) SAMPLES=$2; shift 2 ;;
    --dtype) DTYPE=$2; shift 2 ;;
    --profile) PROFILE=$2; shift 2 ;;
    --no-gemv) GEMV=; shift ;;
    --out) OUT=$2; shift 2 ;;
    --build) BUILD=$2; shift 2 ;;
    --arch) ARCH=$2; shift 2 ;;
    --venv) VENV=$2; shift 2 ;;
    --python) PYTHON=$2; shift 2 ;;
    -h | --help) usage ;;
    *) echo "throughput.sh: unknown option $1" >&2; usage 2 ;;
  esac
done
[ -f "$CONFIG" ] || { echo "throughput.sh: no config $CONFIG" >&2; exit 2; }
IFS=, read -ra MODE_LIST <<< "$MODES"
for mode in "${MODE_LIST[@]}"; do
  case $mode in compiled | eager) ;; *) echo "throughput.sh: unknown mode $mode" >&2; exit 2 ;; esac
done
IFS=, read -ra ATTENTION_LIST <<< "$ATTENTION"
for attention in "${ATTENTION_LIST[@]}"; do
  case $attention in kernel | sdpa) ;; *) echo "throughput.sh: unknown attention $attention" >&2; exit 2 ;; esac
done
export PYTHONPATH=$ROOT${PYTHONPATH:+:$PYTHONPATH}
step() { echo "== [$(date -u +%H:%M:%S)] $*"; }

step "1/4 environment and kernel -> $BUILD/sph_gemv.so"
[ -x "$VENV/bin/python" ] || "$PYTHON" -m venv "$VENV"
PY=$VENV/bin/python
if ! "$PY" -c 'import torch, transformers, safetensors' 2> /dev/null; then
  "$PY" -m pip install --upgrade pip
  "$PY" -m pip install -r "$ROOT/requirements.txt"
fi
"$PY" -m cuda_kernel.build --out "$BUILD" ${ARCH:+--arch "$ARCH"}
mkdir -p "$OUT"

if [ -n "$GEMV" ]; then
  step "2/4 per-projection timings -> $OUT/gemv.json"
  "$PY" -m throughput.gemv_bench --config "$CONFIG" --kernel "$BUILD/sph_gemv.so" --out "$OUT/gemv.json"
else
  step "2/4 per-projection timings skipped"
fi

step "3/4 decode measurements -> $OUT"
runs=()
for attention in "${ATTENTION_LIST[@]}"; do
  for mode in "${MODE_LIST[@]}"; do
    for method in dense w2; do
      echo "-- $method, $mode, $attention attention"
      prof=()
      [ "$mode" = compiled ] && [ "$PROFILE" -gt 0 ] && prof=(--profile "$PROFILE")
      suffix=
      [ "$attention" = sdpa ] && suffix=_sdpa
      out=$OUT/${method}_$mode$suffix.json
      "$PY" -m throughput.decode_bench --config "$CONFIG" --method "$method" --kernel "$BUILD/sph_gemv.so" \
        --mode "$mode" --attention "$attention" --dtype "$DTYPE" --prefill "$PREFILL" --decode "$DECODE" \
        --warmup "$WARMUP" --samples "$SAMPLES" "${prof[@]}" --out "$out"
      runs+=("$out")
    done
  done
done

step "4/4 summary -> $OUT/summary.md"
gemv=()
[ -n "$GEMV" ] && gemv=(--gemv "$OUT/gemv.json")
"$PY" -m throughput.summarize --runs "${runs[@]}" "${gemv[@]}" --out "$OUT/summary.md"
