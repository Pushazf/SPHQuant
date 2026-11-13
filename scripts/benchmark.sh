#!/usr/bin/env bash
# Evaluate one model on the SPHQuant benchmarks, served by SGLang (Linux, one GPU).
#
#   bash scripts/benchmark.sh --model DIR --weights-id NAME [options]
#
# 1. installs SGLang (pinned) into a virtual environment;
# 2. downloads each dataset at the revision pinned in benchmark/<dataset>.py and checks its hashes;
# 3. starts SGLang's own server, `python -m sglang.launch_server`, and waits until it is healthy;
# 4. runs the in-repo client (benchmark/client.py) for every dataset and thinking mode;
# 5. stops the server and writes DIR_OUT/<dataset>/summary.md over every weights id evaluated into the same --out
#    (accuracy, truncation, loops and, when bf16 is there, the paired difference to bf16 with a bootstrap 95% CI).
# Answers are appended to DIR_OUT/<dataset>/<weights-id>_think-<off|on>.jsonl; running again resumes.
#
# Options:
#   --model DIR          model directory: the BF16 checkpoint or calibrate_w2.sh's w2-dense (required)
#   --weights-id NAME    label of these weights in the results, e.g. bf16 or w2 (required)
#   --datasets LIST      comma list of realworldqa,scienceqa,textvqa (default: all three)
#   --think MODE         off, on or off,on (default off, as in the paper)
#   --limit N            seeded subset of N questions per dataset, stratified by subject (default 0: all)
#   --out DIR_OUT        results (default: outputs/eval in the repository)
#   --data DIR           dataset snapshots (default: data in the repository)
#   --venv DIR           virtual environment, created when missing (default: .venv in the repository)
#   --python PATH        interpreter that creates the environment (default: python3; 3.12 recommended)
#   --port PORT          server port (default 30000)
#   --concurrency N      parallel requests (default 64)
#   --max-tokens-off N   output cap with thinking off (default 2048)
#   --max-tokens-on N    output cap with thinking on (default 16384)
# Environment: SGLANG_ARGS adds server flags (default "--attention-backend triton"; set it empty for none).
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SGLANG_VERSION=0.5.23
ALL=realworldqa,scienceqa,textvqa
MODEL= WEIGHTS_ID= DATASETS=$ALL THINK=off LIMIT=0 OUT=$ROOT/outputs/eval DATA=$ROOT/data VENV=$ROOT/.venv
PYTHON=python3 PORT=30000 CONCURRENCY=64 MAX_OFF=2048 MAX_ON=16384 HEALTH_TIMEOUT=1800
SGLANG_ARGS=${SGLANG_ARGS---attention-backend triton}

usage() { sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed '$d; s/^# \{0,1\}//'; exit "${1:-0}"; }
while [ $# -gt 0 ]; do
  case $1 in
    --model) MODEL=$2; shift 2 ;;
    --weights-id) WEIGHTS_ID=$2; shift 2 ;;
    --datasets) DATASETS=$2; shift 2 ;;
    --think) THINK=$2; shift 2 ;;
    --limit) LIMIT=$2; shift 2 ;;
    --out) OUT=$2; shift 2 ;;
    --data) DATA=$2; shift 2 ;;
    --venv) VENV=$2; shift 2 ;;
    --python) PYTHON=$2; shift 2 ;;
    --port) PORT=$2; shift 2 ;;
    --concurrency) CONCURRENCY=$2; shift 2 ;;
    --max-tokens-off) MAX_OFF=$2; shift 2 ;;
    --max-tokens-on) MAX_ON=$2; shift 2 ;;
    -h | --help) usage ;;
    *) echo "benchmark.sh: unknown option $1" >&2; usage 2 ;;
  esac
done
[ -n "$MODEL" ] && [ -n "$WEIGHTS_ID" ] || { echo "benchmark.sh: --model and --weights-id are required" >&2; usage 2; }
[ -f "$MODEL/config.json" ] || { echo "benchmark.sh: $MODEL is not a model directory (no config.json)" >&2; exit 2; }
case $THINK in off | on | off,on | on,off) ;; *) echo "benchmark.sh: bad --think $THINK" >&2; exit 2 ;; esac
IFS=, read -ra DS <<< "$DATASETS"
for ds in "${DS[@]}"; do
  case ",$ALL," in *",$ds,"*) ;; *) echo "benchmark.sh: unknown dataset $ds" >&2; exit 2 ;; esac
done
MODEL=$(cd "$MODEL" && pwd)
mkdir -p "$OUT" "$DATA"
export PYTHONPATH=$ROOT${PYTHONPATH:+:$PYTHONPATH}
step() { echo "== [$(date -u +%H:%M:%S)] $*"; }

step "1/5 SGLang $SGLANG_VERSION in $VENV"
[ -x "$VENV/bin/python" ] || "$PYTHON" -m venv "$VENV"
PY=$VENV/bin/python
if ! "$PY" -c 'import sys, sglang, httpx, pyarrow, PIL; sys.exit(sglang.__version__ != sys.argv[1])' \
    "$SGLANG_VERSION" 2> /dev/null; then
  "$PY" -m pip install --upgrade pip
  "$PY" -m pip install "sglang==$SGLANG_VERSION" httpx pyarrow pillow numpy
fi

step "2/5 datasets -> $DATA"
for ds in "${DS[@]}"; do
  "$PY" -m benchmark.fetch --dataset "$ds" --out "$DATA/$ds"
done

BASE=http://127.0.0.1:$PORT
LOG=$OUT/server_$WEIGHTS_ID.log
step "3/5 server: $MODEL on port $PORT (log $LOG)"
if curl -s -o /dev/null "$BASE/health"; then
  echo "benchmark.sh: something already listens on port $PORT; pick another with --port" >&2
  exit 1
fi
# shellcheck disable=SC2086  # SGLANG_ARGS holds several flags
setsid "$PY" -m sglang.launch_server --model-path "$MODEL" --dtype bfloat16 --host 127.0.0.1 --port "$PORT" \
  --random-seed 0 $SGLANG_ARGS < /dev/null >> "$LOG" 2>&1 &
SERVER=$!
stop_server() {  # TERM the server's process group, KILL what is left after 30 s
  kill -TERM -- "-$SERVER" 2> /dev/null || return 0
  for _ in $(seq 30); do
    kill -0 -- "-$SERVER" 2> /dev/null || return 0
    sleep 1
  done
  kill -KILL -- "-$SERVER" 2> /dev/null || true
}
trap stop_server EXIT
trap 'exit 130' INT TERM HUP
deadline=$((SECONDS + HEALTH_TIMEOUT))
until curl -sf -o /dev/null "$BASE/health"; do
  kill -0 "$SERVER" 2> /dev/null || { echo "benchmark.sh: the server exited; see $LOG" >&2; exit 1; }
  [ "$SECONDS" -lt "$deadline" ] || { echo "benchmark.sh: no healthy server after $HEALTH_TIMEOUT s" >&2; exit 1; }
  sleep 5
done

step "4/5 evaluation of $WEIGHTS_ID"
for ds in "${DS[@]}"; do
  mkdir -p "$OUT/$ds"
  for mode in ${THINK//,/ }; do
    max=$MAX_OFF
    [ "$mode" = on ] && max=$MAX_ON
    run=$OUT/$ds/${WEIGHTS_ID}_think-$mode
    echo "-- $ds, thinking $mode, at most $max tokens -> $run.jsonl"
    "$PY" -m benchmark.client --base-url "$BASE" --expect-model "$MODEL" --tokenizer "$MODEL" \
      --dataset "$ds" --data "$DATA/$ds" --weights-id "$WEIGHTS_ID" --think "$mode" --max-tokens "$max" \
      --concurrency "$CONCURRENCY" --limit "$LIMIT" --sample-seed 0 --out "$run.jsonl" | tee "$run.summary.json"
  done
done
stop_server
trap - EXIT

step "5/5 summaries"
for ds in "${DS[@]}"; do
  runs=("$OUT/$ds"/*_think-*.jsonl)
  "$PY" -m benchmark.aggregate --runs "${runs[@]}" --out "$OUT/$ds/summary.md" > /dev/null
  echo "$OUT/$ds/summary.md"
done
