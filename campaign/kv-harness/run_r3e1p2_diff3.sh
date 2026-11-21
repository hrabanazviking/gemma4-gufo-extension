#!/usr/bin/env bash
# R3-E1 part 2 differential test -- build + run.
# Standalone hipcc compile of the shipping TUs the test links
# (hip/kernels.hip.cpp, hip/attention_wmma.hip.cpp,
# hip/ggml/prefill_quant_gemm.hip.cpp) with the extension's own
# compile options from extension.cmake (-fno-fast-math
# -ffp-contract=off, ENGINE_ENABLE_HIP defined), the wave64 TU
# separate with its own options (-mwavefrontsize64
# -fno-slp-vectorize), and the pinned gufo SDK's ggml_dequant.cpp
# for gufo::quant::QuantizedRowBytes. No shipping file is
# modified by this harness; the binary runs from /tmp.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
HIPCC=/nix/store/al4qwi7jfyrdkyb25xsri8b00b1agj1r-clr-10.1/bin/hipcc
DEVLIB=/nix/store/gyzwfffwdwj8dpkrasiqd5hkr2adx844-rocm-device-libs-24.0.0-rocm/amdgcn/bitcode
SDK=/nix/store/3ay54spwrbl6mgjnqnahrg4zhpcc7i7l-source
INC="-I$ROOT -I$SDK/include -I$SDK -I/nix/store/cwa50bpg7sgalmipfv4y2zk8ywq7fcgz-hipblas-10.1/include -I/nix/store/mhj9saayv4y5nvbdn8vq565qbhrszj1i-hipblas-common-10.1/include -I/nix/store/8n05wfii2rjin9nfkamv589sa15bgkiy-hipblaslt-gfx1151-10.1/include -I/nix/store/7ip5lk8jivjig339mcc8b981fa736n5m-rocblas-gfx1151-10.1/include"
BASE="--rocm-device-lib-path=$DEVLIB --offload-arch=gfx1151 -O2 -std=c++20 -DENGINE_ENABLE_HIP -fno-fast-math -ffp-contract=off $INC"
OBJ="${TMPDIR:-/tmp}/r3obj"
mkdir -p "$OBJ"
cd "$ROOT"
"$HIPCC" $BASE -mwavefrontsize64 -fno-slp-vectorize \
  -c hip/ggml/prefill_quant_wave64.hip.cpp -o "$OBJ/wave64.o" || exit 1
"$HIPCC" $BASE -c campaign/kv-harness/r3e1p2_diff3.hip.cpp \
  -o "$OBJ/harness.o" || exit 1
"$HIPCC" $BASE -c hip/kernels.hip.cpp -o "$OBJ/kernels.o" || exit 1
"$HIPCC" $BASE -c hip/attention_wmma.hip.cpp -o "$OBJ/wmma.o" || exit 1
"$HIPCC" $BASE -fno-slp-vectorize -c hip/ggml/prefill_quant_gemm.hip.cpp \
  -o "$OBJ/pqg.o" || exit 1
g++ -std=c++20 -O2 -I"$SDK/include" -I"$SDK" \
  -c "$SDK/src/core/quant/ggml_dequant.cpp" -o "$OBJ/dequant.o" || exit 1
"$HIPCC" --rocm-device-lib-path="$DEVLIB" --offload-arch=gfx1151 \
  -o "${TMPDIR:-/tmp}/r3e1p2_diff3" "$OBJ/harness.o" "$OBJ/kernels.o" \
  "$OBJ/wmma.o" "$OBJ/pqg.o" "$OBJ/wave64.o" "$OBJ/dequant.o" || exit 1
export LD_LIBRARY_PATH=/nix/store/ivl6g6m6cas1027wynvzrkmm7hnrb8jg-rocm-comgr-24.0.0-rocm/lib:/nix/store/al4qwi7jfyrdkyb25xsri8b00b1agj1r-clr-10.1/lib:/nix/store/88jjsz00km321bcl9851aiwwdig79b4l-rocm-runtime-10.1/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
grep MemAvailable /proc/meminfo
"${TMPDIR:-/tmp}/r3e1p2_diff3"
