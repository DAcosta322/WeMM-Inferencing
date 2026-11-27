#!/usr/bin/env bash
# WeMM-Video-35B-A3B decode-TPS benchmark — single entrypoint.
#
#   ./run_bench.sh env         probe GPU / vLLM / FlashInfer / WeMM patch
#   ./run_bench.sh dry-run     print the exact server command for every arm
#   ./run_bench.sh prep-mtp    build the MTP sidecar (downloads Qwen MTP head)
#   ./run_bench.sh run         measure baseline + optimised arms, write report
#   ./run_bench.sh report      rebuild the report from existing results
#   ./run_bench.sh self-test   validate the harness with the mock server (no GPU)
#   ./run_bench.sh fix-mtp-config  backport the vLLM 0.18.0 Qwen3.5-MoE config fix
#
# An active venv is used automatically. Only set BENCH_PYTHON if you need a
# different interpreter: BENCH_PYTHON="$(command -v python)" ./run_bench.sh env
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

# Pick the interpreter: BENCH_PYTHON wins, then an activated venv, then `python`.
PY_SOURCE="BENCH_PYTHON"
PY="${BENCH_PYTHON:-}"
if [[ -z "$PY" ]]; then
  if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    PY="$VIRTUAL_ENV/bin/python"
    PY_SOURCE="VIRTUAL_ENV"
  else
    PY=python
    PY_SOURCE="python"
  fi
fi
# Expand a leading ~ that survived quoting ("~/venv/bin/python").
case "$PY" in
  "~/"*) PY="$HOME/${PY#\~/}" ;;
esac

_bench_usable() {
  command -v "$1" >/dev/null 2>&1 && "$1" -c 'import requests' >/dev/null 2>&1
}

# A stale/placeholder BENCH_PYTHON should not block an activated venv.
if [[ "$PY_SOURCE" == "BENCH_PYTHON" ]] && ! command -v "$PY" >/dev/null 2>&1; then
  _fallback=""
  if [[ -n "${VIRTUAL_ENV:-}" && -x "$VIRTUAL_ENV/bin/python" ]]; then
    _fallback="$VIRTUAL_ENV/bin/python"
  elif command -v python >/dev/null 2>&1; then
    _fallback="$(command -v python)"
  fi
  if [[ -n "$_fallback" ]] && _bench_usable "$_fallback"; then
    echo "warning: BENCH_PYTHON='$PY' is not executable; using $_fallback instead." >&2
    echo "         unset BENCH_PYTHON to silence this; an active venv is detected automatically." >&2
    PY="$_fallback"
    PY_SOURCE="fallback"
  fi
  unset _fallback
fi

if ! command -v "$PY" >/dev/null 2>&1; then
  echo "error: interpreter '$PY' not found (from $PY_SOURCE)." >&2
  echo "       either activate the patched-vLLM venv, or set an absolute path:" >&2
  echo "       BENCH_PYTHON=\"\$HOME/WeMM-Inferencing/WeMM-Video/.venv-vllm/bin/python\" ./run_bench.sh env" >&2
  exit 2
fi
# CUDA NPP is not bundled with PyTorch; torchcodec (imported by WeMM at model
# load) needs it on the dynamic-loader path. Export it once for every child.
NPP_LIB="$("$PY" - <<'PYNPP' 2>/dev/null
try:
    from importlib.metadata import distribution
    print(distribution("nvidia-npp-cu12").locate_file("nvidia/npp/lib"))
except Exception:
    pass
PYNPP
)"
if [[ -n "${NPP_LIB:-}" && -d "$NPP_LIB" ]]; then
  case ":${LD_LIBRARY_PATH:-}:" in
    *":$NPP_LIB:"*) ;;
    *) export LD_LIBRARY_PATH="$NPP_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
  esac
fi

if ! "$PY" -c 'import requests' >/dev/null 2>&1; then
  echo "error: '$PY' cannot import requests." >&2
  "$PY" - <<'PYCHECK' >&2 || true
import sys
print(f"        interpreter: {sys.executable} (python {sys.version.split()[0]})")
PYCHECK
  echo "       set BENCH_PYTHON to the patched-vLLM venv python." >&2
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

An active venv is picked automatically. BENCH_PYTHON="$(command -v python)"
overrides it; do not paste placeholder paths like /path/to/... into it.

Common options (passed through to bench.run_all):
  --full                 include MTP K=5/K=7, n-gram and eager diagnostic arms
  --quick                fewer repetitions
  --arms a,b,c           run only the named arms
  --list-arms            show arm names
  --mtp-head PATH        use an existing sidecar instead of results/mtp_head
  --gpu-mem 0.90         vLLM --gpu-memory-utilization (raise if BF16+MTP OOMs)
  --port 8000            preferred port; auto-advances if busy (--strict-port to fail)
  env --port 8010        check a specific port and see which process holds it
EOF
}

case "${1:-help}" in
  help|-h|--help) usage ;;
  env)        shift; exec "$PY" -m bench.env_check "$@" ;;
  prep-mtp)   shift; exec "$PY" -m bench.make_mtp_head "$@" ;;
  run)        shift; exec "$PY" -m bench.run_all "$@" ;;
  dry-run)    shift; exec "$PY" -m bench.run_all --dry-run "$@" ;;
  report)     shift; exec "$PY" -m bench.run_all --report-only "$@" ;;
  self-test)  shift; exec "$PY" -m bench.run_all --self-test --yes "$@" ;;
  fix-mtp-config)
    shift
    if [ -d results/mtp_head ]; then
      exec "$PY" -m bench.fix_vllm_config --apply --validate results/mtp_head "$@"
    else
      exec "$PY" -m bench.fix_vllm_config --apply "$@"
    fi
    ;;
  list-arms)  shift; exec "$PY" -m bench.run_all --list-arms "$@" ;;
  *)          exec "$PY" -m bench.run_all "$@" ;;
esac
