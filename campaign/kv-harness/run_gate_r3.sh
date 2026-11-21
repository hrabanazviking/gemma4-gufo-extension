#!/usr/bin/env bash
set -u
cd "$HOME/src/gemma4-gufo-extension"
BIN="$PWD/result-r3e2/bin/gufo"
OUT="campaign/kv-harness/results/gate-r3"
mkdir -p "$OUT"
{
  echo "gate-r3 start $(date -Is)"
  grep MemAvailable /proc/meminfo
  if g++ -std=c++20 -Wall -Wextra -I. tests/kv_cache_format_test.cpp -o /tmp/kvfmt_gate && /tmp/kvfmt_gate; then echo HOST_TEST_PASS; else echo HOST_TEST_FAIL; fi
} > "$OUT/hosttest.log" 2>&1
GUFO_KV_CACHE_TYPE=q8_0 bash campaign/kv-harness/run_probes.sh r3gate-q8 "$BIN" > "$OUT/probes-q8.log" 2>&1
GUFO_KV_CACHE_TYPE=f16 bash campaign/kv-harness/run_probes.sh r3gate-f16 "$BIN" > "$OUT/probes-f16.log" 2>&1
GUFO_KV_CACHE_TYPE=q8_0 bash campaign/kv-harness/run_matrix.sh r3gate-q8 "$BIN" > "$OUT/matrix-q8.log" 2>&1
GUFO_KV_CACHE_TYPE=f16 bash campaign/kv-harness/run_matrix.sh r3gate-f16 "$BIN" > "$OUT/matrix-f16.log" 2>&1
grep MemAvailable /proc/meminfo >> "$OUT/hosttest.log"
echo "gate-r3 done $(date -Is)" >> "$OUT/hosttest.log"
touch campaign/kv-harness/results/gate-r3.done
