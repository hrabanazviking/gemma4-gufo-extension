#!/usr/bin/env bash
# KV campaign probe runner (KV-01). Frozen probe set: probes.txt
# (short probes, "ID :: PROMPT" per line) + one long-context needle
# probe built here. Greedy decoding: temperature 0, seed 42,
# thinking off. Full answers land in results/<label>/probes/.
# Usage: run_probes.sh <label> [gufo-bin]
set -u
LABEL="${1:?usage: run_probes.sh <label> [gufo-bin]}"
GUFO="${2:-$HOME/.nix-profile/bin/gufo}"
MODEL="${KV_MODEL:-$HOME/models/gufo/gemma-4-26B-A4B-it-heretic.Q8_0.gguf}"
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="$HERE/results/$LABEL/probes"
mkdir -p "$OUT"

while IFS= read -r line; do
  case "$line" in ''|\#*) continue;; esac
  id="${line%% :: *}"; prompt="${line##* :: }"
  "$GUFO" prompt -m "$MODEL" -p "$prompt" -t 0 -s 42 --think off \
    -n 96 --no-display-prompt > "$OUT/$id.txt" 2>"$OUT/$id.err"
done < "$HERE/probes.txt"

# Long-context needle probe (~16k tokens of filler, fact in the middle).
LONGTXT="$(mktemp)"
python3 - "$LONGTXT" <<'PYEOF'
import sys
filler = ("The road wound through pine hills and the river ran cold "
          "beside it, and the keeper of the pass walked his rounds "
          "as he always had. ")
needle = ("In his ledger the keeper wrote one line that mattered: "
          "the vault combination is 47-12-89. ")
parts = [filler * 650, needle, filler * 650,
         "\nQuestion: What is the vault combination given in the "
         "text above? Answer with the numbers only."]
open(sys.argv[1], "w").write("".join(parts))
PYEOF
"$GUFO" prompt -m "$MODEL" -f "$LONGTXT" -t 0 -s 42 --think off \
  -n 32 --no-display-prompt > "$OUT/LONG-1.txt" 2>"$OUT/LONG-1.err"
rm -f "$LONGTXT"
echo "probes done: $OUT"
