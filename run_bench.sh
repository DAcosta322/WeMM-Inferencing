#!/usr/bin/env bash
# WeMM-Video-35B-A3B decode-TPS benchmark — single entrypoint.
#
#   ./run_bench.sh env         probe GPU / vLLM / FlashInfer / WeMM patch
#   ./run_bench.sh dry-run     print the exact server command for every arm
#   ./run_bench.sh prep-mtp    build the MTP sidecar (downloads Qwen MTP head)
#   ./run_bench.sh run         measure baseline + optimised arms, write report
#   ./run_bench.sh report      rebuild the report from existing results
#   ./run_bench.sh self-test   validate the harness with the mock server (no GPU)
#
# Use BENCH_PYTHON=/path/to/venv/bin/python to select the patched-vLLM venv.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

PY="${BENCH_PYTHON:-python}"
if ! command -v "$PY" >/dev/null 2>&1; then
  PY=python3
fi
if ! "$PY" -c 'import requests' >/dev/null 2>&1; then
  echo "error: '$PY' cannot import requests; set BENCH_PYTHON to the patched-vLLM venv python" >&2
  exit 2
fi

usage() {
  cat <<'EOF'
WeMM-Video-35B-A3B decode-TPS benchmark.

  ./run_bench.sh env         probe GPU / vLLM / FlashInfer / WeMM patch
  ./run_bench.sh dry-run     print the exact server command for every arm
  ./run_bench.sh prep-mtp    build the MTP sidecar (downloads the Qwen MTP head)
  ./run_bench.sh run         measure baseline + optimised arms, write the report
  ./run_bench.sh report      rebuild the report from existing results
  ./run_bench.sh self-test   validate the harness with the mock server (no GPU)

Use BENCH_PYTHON=/path/to/venv/bin/python to select the patched-vLLM venv.

Common options (passed through to bench.run_all):
  --full                 include MTP K=5/K=7, n-gram and eager diagnostic arms
  --quick                fewer repetitions
  --arms a,b,c           run only the named arms
  --list-arms            show arm names
  --mtp-head PATH        use an existing sidecar instead of results/mtp_head
  --gpu-mem 0.90         vLLM --gpu-memory-utilization (raise if BF16+MTP OOMs)
EOF
}

case "${1:-help}" in
  help|-h|--help) usage ;;
  env)        exec "$PY" -m bench.env_check ;;
  prep-mtp)   shift; exec "$PY" -m bench.make_mtp_head "$@" ;;
  run)        shift; exec "$PY" -m bench.run_all "$@" ;;
  dry-run)    shift; exec "$PY" -m bench.run_all --dry-run "$@" ;;
  report)     shift; exec "$PY" -m bench.run_all --report-only "$@" ;;
  self-test)  shift; exec "$PY" -m bench.run_all --self-test --yes "$@" ;;
  list-arms)  shift; exec "$PY" -m bench.run_all --list-arms "$@" ;;
  *)          exec "$PY" -m bench.run_all "$@" ;;
esac
