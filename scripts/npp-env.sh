#!/usr/bin/env bash
# Put CUDA NPP on the dynamic-loader path for the current shell.
#
#   source scripts/npp-env.sh
#
# Needed when you run `python`, `vllm` or `torchcodec` by hand. ./run_bench.sh
# already does this for every process it starts, so it is optional there.
#
# NPP is not bundled with PyTorch and dlopen does not search site-packages;
# the nvidia-npp-cu12 wheel ships libnppicc.so.12 under nvidia/npp/lib, which
# has to be on LD_LIBRARY_PATH.
set -u

_npp_lib="$(
  python - <<'PY' 2>/dev/null
try:
    from importlib.metadata import distribution
    print(distribution("nvidia-npp-cu12").locate_file("nvidia/npp/lib"))
except Exception:
    pass
PY
)"

if [ -n "${_npp_lib:-}" ] && [ -d "$_npp_lib" ] \
   && { [ -f "$_npp_lib/libnppicc.so.12" ] || [ -f "$_npp_lib/libnppicc.so.13" ]; }; then
  case ":${LD_LIBRARY_PATH:-}:" in
    *":$_npp_lib:"*) ;;
    *) export LD_LIBRARY_PATH="$_npp_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
  esac
  echo "npp: $_npp_lib"
elif [ -n "${_npp_lib:-}" ] && [ -d "$_npp_lib" ]; then
  echo "npp: found $_npp_lib but no libnppicc.so.12/.13 inside it" >&2
  echo "     reinstall: python -m pip install --force-reinstall nvidia-npp-cu12==12.4.1.87" >&2
else
  echo "npp: nvidia-npp-cu12 not found in this Python; install it first" >&2
  echo "     python -m pip install --index-url https://pypi.org/simple nvidia-npp-cu12==12.4.1.87" >&2
fi
unset _npp_lib
