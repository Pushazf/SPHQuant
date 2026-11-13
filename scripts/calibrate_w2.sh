#!/usr/bin/env bash
# SPHQuant W2 calibration of Qwen3.5-9B on one GPU (Linux).
#
#   bash scripts/calibrate_w2.sh --work-dir DIR [options]
#
# Steps; each one is skipped when its output is already complete, so running the script again resumes:
#   1. virtual environment with requirements.txt
#   2. Qwen3.5-9B at a pinned revision -> DIR/models/Qwen3.5-9B; COCO val2017 images and captions -> DIR/data/coco
#   3. calibration cache: 4096 training + 128 validation COCO samples -> DIR/calib-cache
#   4. layer-wise calibration -> DIR/calib-w2 (an interrupted run continues from its last finished layer)
#   5. export (always redone, about a minute): the decoded dense BF16 checkpoint DIR/w2-dense (serve it with
#      scripts/benchmark.sh) and the packed W2 weights DIR/w2-packed (2.5 bits per weight)
#
# Options:
#   --work-dir DIR   models, data, cache and results go here (required)
#   --venv DIR       virtual environment, created when missing (default: .venv in the repository)
#   --python PATH    interpreter that creates the environment (default: python3; 3.12 recommended)
#   --layers A-B     decoder layers to quantize (default 0-31, all of Qwen3.5-9B)
#   --epochs N       epochs per layer (default 20)
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MODEL_REPO=Qwen/Qwen3.5-9B
MODEL_REV=c202236235762e1c871ad0ccb60c8ee5ba337b9a
COCO_URL=http://images.cocodataset.org
WORK= VENV=$ROOT/.venv PYTHON=python3 LAYERS=0-31 EPOCHS=20

usage() { sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed '$d; s/^# \{0,1\}//'; exit "${1:-0}"; }
while [ $# -gt 0 ]; do
  case $1 in
    --work-dir) WORK=$2; shift 2 ;;
    --venv) VENV=$2; shift 2 ;;
    --python) PYTHON=$2; shift 2 ;;
    --layers) LAYERS=$2; shift 2 ;;
    --epochs) EPOCHS=$2; shift 2 ;;
    -h | --help) usage ;;
    *) echo "calibrate_w2.sh: unknown option $1" >&2; usage 2 ;;
  esac
done
[ -n "$WORK" ] || { echo "calibrate_w2.sh: --work-dir is required" >&2; usage 2; }
mkdir -p "$WORK"
WORK=$(cd "$WORK" && pwd)
MODEL=$WORK/models/Qwen3.5-9B
COCO=$WORK/data/coco
CACHE=$WORK/calib-cache
CALIB=$WORK/calib-w2
export PYTHONPATH=$ROOT${PYTHONPATH:+:$PYTHONPATH}
step() { echo "== [$(date -u +%H:%M:%S)] $*"; }

step "1/5 environment: $VENV"
[ -x "$VENV/bin/python" ] || "$PYTHON" -m venv "$VENV"
PY=$VENV/bin/python
if ! "$PY" -c 'import torch, transformers, safetensors, pyarrow, PIL' 2> /dev/null; then
  "$PY" -m pip install --upgrade pip
  "$PY" -m pip install -r "$ROOT/requirements.txt"
fi
"$PY" -c 'import torch; assert torch.cuda.is_available(), "calibration needs a CUDA GPU"'

step "2/5 model $MODEL_REPO@${MODEL_REV:0:12} and COCO val2017"
"$PY" -c '
import sys
from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1], revision=sys.argv[2], local_dir=sys.argv[3])' "$MODEL_REPO" "$MODEL_REV" "$MODEL"
mkdir -p "$COCO"
fetch() {  # fetch URL SIZE FILE: download unless FILE already has exactly SIZE bytes
  [ "$(stat -c %s "$3" 2> /dev/null || echo 0)" = "$2" ] && return 0
  curl -fL --retry 5 -o "$3.part" "$1"
  [ "$(stat -c %s "$3.part")" = "$2" ] || { echo "$1 is not $2 bytes" >&2; return 1; }
  mv -f "$3.part" "$3"
}
fetch "$COCO_URL/zips/val2017.zip" 815585330 "$COCO/val2017.zip"
fetch "$COCO_URL/annotations/annotations_trainval2017.zip" 252907541 "$COCO/annotations_trainval2017.zip"
"$PY" -c '
import glob, os, sys, zipfile
coco = sys.argv[1]
if len(glob.glob(os.path.join(coco, "val2017", "*.jpg"))) != 5000:
    zipfile.ZipFile(os.path.join(coco, "val2017.zip")).extractall(coco)
if not os.path.isfile(os.path.join(coco, "annotations", "captions_val2017.json")):
    zipfile.ZipFile(os.path.join(coco, "annotations_trainval2017.zip")).extract(
        "annotations/captions_val2017.json", coco)' "$COCO"

step "3/5 calibration cache -> $CACHE"
if [ -f "$CACHE/manifest.json" ]; then
  echo "complete (manifest.json present); delete $CACHE to rebuild it"
else
  "$PY" -m sphquant.coco_data --model "$MODEL" --images "$COCO/val2017" \
    --captions "$COCO/annotations/captions_val2017.json" --out "$CACHE" --train 4096 --val 128 --seed 0 \
    --instruction "Describe this image in one sentence."
fi

step "4/5 calibration, layers $LAYERS, $EPOCHS epochs -> $CALIB"
"$PY" -m sphquant.calibrate --model "$MODEL" --cache "$CACHE" --out "$CALIB" --layers "$LAYERS" \
  --epochs "$EPOCHS" --batch 16 --lr 1e-3 --seed 0 --grid recompute --lr-schedule constant

step "5/5 export -> $WORK/w2-dense, $WORK/w2-packed"
"$PY" -m sphquant.export --model "$MODEL" --calib "$CALIB" --dense-out "$WORK/w2-dense" \
  --packed-out "$WORK/w2-packed" --layers "$LAYERS"
step "done: dense W2 model $WORK/w2-dense, packed weights $WORK/w2-packed"
