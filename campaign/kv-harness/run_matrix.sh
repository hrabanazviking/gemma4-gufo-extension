#!/usr/bin/env bash
# KV campaign benchmark matrix driver (KV-01).
# Usage: run_matrix.sh <label> [gufo-bin]
# Protocol (roadmap): per config, 1 warmup invocation discarded +
# 5 measured invocations; the median of the 5 is the result.
# Raw logs: results/<label>/<config>.run<N>.log ; summary.txt alongside.
set -u
LABEL="${1:?usage: run_matrix.sh <label> [gufo-bin]}"
GUFO="${2:-$HOME/.nix-profile/bin/gufo}"
MODEL="${KV_MODEL:-$HOME/models/gufo/gemma-4-26B-A4B-it-heretic.Q8_0.gguf}"
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="$HERE/results/$LABEL"
mkdir -p "$OUT"

# config: name|bench args|exact test-row label printed by gufo bench
CONFIGS=(
  "pp512|-p 512 -n 128 -d 0|pp512"
  "pp4096|-p 4096 -n 128 -d 0|pp4096"
  "pp16384|-p 16384 -n 128 -d 0|pp16384"
  "tg512|-p 512 -n 128 -d 0|tg128"
  "tg4096|-p 512 -n 128 -d 3584|tg128 @ d3584"
  "tg16384|-p 512 -n 128 -d 15872|tg128 @ d15872"
)

extract() { # file, rowlabel -> t/s value or empty
  awk -F'|' -v want="$2" '{
    t=$5; gsub(/^[ \t]+|[ \t]+$/, "", t);
    if (t == want) { v=$6; gsub(/^[ \t]+|[ \t]+$/, "", v); split(v, a, " "); print a[1]; exit }
  }' "$1"
}

{
  echo "label: $LABEL"
  echo "date: $(date -Is)"
  echo "gufo: $($GUFO --version 2>/dev/null)"
  echo "model: $MODEL"
  grep MemAvailable /proc/meminfo
} > "$OUT/summary.txt"

for cfg in "${CONFIGS[@]}"; do
  IFS='|' read -r name args row <<< "$cfg"
  values=()
  for run in 0 1 2 3 4 5; do
    log="$OUT/$name.run$run.log"
    # shellcheck disable=SC2086
    "$GUFO" bench -m "$MODEL" $args > "$log" 2>&1
    v="$(extract "$log" "$row")"
    if [ "$run" -eq 0 ]; then
      echo "$name warmup: ${v:-FAILED}" >> "$OUT/summary.txt"
    elif [ -n "$v" ]; then
      values+=("$v")
    else
      echo "$name run$run: FAILED (see $log)" >> "$OUT/summary.txt"
    fi
  done
  if [ "${#values[@]}" -eq 5 ]; then
    median="$(printf '%s\n' "${values[@]}" | sort -n | sed -n '3p')"
    echo "$name runs: ${values[*]}" >> "$OUT/summary.txt"
    echo "$name MEDIAN: $median" >> "$OUT/summary.txt"
  else
    echo "$name MEDIAN: UNAVAILABLE (${#values[@]}/5 runs ok)" >> "$OUT/summary.txt"
  fi
done
echo "done: $OUT/summary.txt"
