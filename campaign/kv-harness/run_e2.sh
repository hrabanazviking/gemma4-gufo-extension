#!/usr/bin/env bash
# KV REDO R6 E2 -- build + run the int8-WMMA QK spike microbenchmark.
# Standalone hipcc compile; no shipping kernel is touched.
set -eu
HERE="$(cd "$(dirname "$0")" && pwd)"
HIPCC=/nix/store/al4qwi7jfyrdkyb25xsri8b00b1agj1r-clr-10.1/bin/hipcc
DEVLIB=/nix/store/gyzwfffwdwj8dpkrasiqd5hkr2adx844-rocm-device-libs-24.0.0-rocm/amdgcn/bitcode
OUT="${TMPDIR:-/tmp}/e2_int8_qk_spike"
"$HIPCC" --rocm-device-lib-path="$DEVLIB" --offload-arch=gfx1151 -O3 \
  -o "$OUT" "$HERE/e2_int8_qk_spike.hip.cpp"
export LD_LIBRARY_PATH=/nix/store/ivl6g6m6cas1027wynvzrkmm7hnrb8jg-rocm-comgr-24.0.0-rocm/lib:/nix/store/al4qwi7jfyrdkyb25xsri8b00b1agj1r-clr-10.1/lib:/nix/store/88jjsz00km321bcl9851aiwwdig79b4l-rocm-runtime-10.1/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
grep MemAvailable /proc/meminfo
"$OUT"
