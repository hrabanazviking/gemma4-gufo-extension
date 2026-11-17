#!/usr/bin/env bash
# KV campaign probe runner (KV-01). Frozen probe set: probes.txt
# (short probes, "ID :: PROMPT" per line) + one long-context needle
# probe built here. Greedy decoding: temperature 0, seed 42,
# thinking off. Full answers land in results/<label>/probes/.
#
# Roads: short probes use `gufo prompt` (its session context is
# fixed at 4096 and it has no --context option). The long probe
# (~16k tokens) therefore runs through a throwaway `gufo serve llm`
# on a spare port with --context 32768, OpenAI chat completions,
# killed afterward. Same road every phase, so answers compare
# like-for-like.
# Usage: run_probes.sh <label> [gufo-bin]
set -u
LABEL="${1:?usage: run_probes.sh <label> [gufo-bin]}"
GUFO="${2:-$HOME/.nix-profile/bin/gufo}"
MODEL="${KV_MODEL:-$HOME/models/gufo/gemma-4-26B-A4B-it-heretic.Q8_0.gguf}"
PORT="${KV_PROBE_PORT:-18200}"
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="$HERE/results/$LABEL/probes"
mkdir -p "$OUT"

while IFS= read -r line; do
  case "$line" in ''|\#*) continue;; esac
  id="${line%% :: *}"; prompt="${line##* :: }"
  "$GUFO" prompt -m "$MODEL" -p "$prompt" -t 0 -s 42 --think off \
    -n 96 --no-display-prompt > "$OUT/$id.txt" 2>"$OUT/$id.err"
done < "$HERE/probes.txt"

# Long-context needle probe (fact in the middle). Filler repeat is
# ~28 tokens; 280 repeats per side measures ~15.7k prompt tokens
# (verified from the response usage in LONG-1.raw.json). An earlier
# 650-per-side version measured 36,459 tokens and overflowed the
# 32768 context with a clean 400 -- the count here is deliberate.
LONGTXT="$(mktemp)"
python3 - "$LONGTXT" <<'PYEOF'
import sys
filler = ("The road wound through pine hills and the river ran cold "
          "beside it, and the keeper of the pass walked his rounds "
          "as he always had. ")
needle = ("In his ledger the keeper wrote one line that mattered: "
          "the vault combination is 47-12-89. ")
parts = [filler * 280, needle, filler * 280,
         "\nQuestion: What is the vault combination given in the "
         "text above? Answer with the numbers only."]
open(sys.argv[1], "w").write("".join(parts))
PYEOF

"$GUFO" serve llm --host 127.0.0.1 --port "$PORT" --model "$MODEL" \
  --served-model-name kv-probe --context 32768 \
  > "$OUT/LONG-1.server.log" 2>&1 &
SERVER_PID=$!
cleanup() { kill "$SERVER_PID" 2>/dev/null; wait "$SERVER_PID" 2>/dev/null; }
trap cleanup EXIT
ready=0
for _ in $(seq 1 60); do
  if curl -s "http://127.0.0.1:$PORT/v1/models" > /dev/null 2>&1; then
    ready=1; break
  fi
  sleep 2
done
rm -f "$OUT/LONG-1.err"
if [ "$ready" -ne 1 ]; then
  echo "serve did not come up on :$PORT (see LONG-1.server.log)" \
    > "$OUT/LONG-1.err"
  echo "LONG-1 FAILED: serve never came up" >&2
  exit 1
fi
if ! python3 - "$LONGTXT" "$PORT" "$OUT/LONG-1.err" \
    > "$OUT/LONG-1.raw.json" <<'PYEOF'
import json, sys, urllib.error, urllib.request
text = open(sys.argv[1]).read()
body = json.dumps({
    "model": "kv-probe",
    "messages": [{"role": "user", "content": text}],
    "temperature": 0, "seed": 42, "max_tokens": 32,
}).encode()
req = urllib.request.Request(
    "http://127.0.0.1:%s/v1/chat/completions" % sys.argv[2],
    data=body, headers={"Content-Type": "application/json"})
try:
    sys.stdout.write(urllib.request.urlopen(req, timeout=600).read().decode())
except urllib.error.HTTPError as e:
    open(sys.argv[3], "w").write("HTTP %s: %s" % (e.code, e.read().decode()))
    sys.exit(1)
PYEOF
then
  echo "LONG-1 FAILED: $(cat "$OUT/LONG-1.err")" >&2
  exit 1
fi
python3 - "$OUT/LONG-1.raw.json" > "$OUT/LONG-1.txt" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1]))
print(d["choices"][0]["message"]["content"])
PYEOF
rm -f "$LONGTXT"
echo "probes done: $OUT"
