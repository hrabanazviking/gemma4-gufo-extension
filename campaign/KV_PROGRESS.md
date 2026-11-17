# KV Q8_0 campaign — progress ledger

Line: hrabanazviking/gemma4-gufo-extension, branch `yrsa/our-gemma4`.
Roadmap: `campaign/KV_Q8_ROADMAP.md`. Charter:
`RUNEFORGEAI_MAD_DATA_SCIENTIST_GEMMA4_EXTENSION.md`. House-only file.

## Build / test road (discovered 2026-11-17, before any code change)

- **Shipping build:** nix, from this tree. `flake.nix` overrides the
  pinned Gufo package (gufo-org/gufo @ dev rev 4938016) with
  `extensions = [ this repo ]`. Command: `nix build` in
  `~/src/gemma4-gufo-extension`; output symlink `result` ->
  `/nix/store/...-gufo-0-unstable-4938016` (binaries `gufo`,
  `gufo-server`). Installed profile: `~/.nix-profile/bin/gufo`
  (build identity `gufo version development (4f73737)`, Extension
  API 0.2.14, gemma4 provider built in). First full build of this
  stack took ~50 min (ROCm 10.1 gfx1151 + Gufo, ~15 GB fetched);
  the store is now warm, so extension-only rebuilds are
  incremental (extension recompile + relink).
- **Bench/probe tools:** the built `gufo` CLI — `gufo bench`
  (-p prompt tokens, -n gen tokens, -d prefix depth, markdown
  table out), `gufo prompt` (greedy: -t 0, fixed -s seed,
  --think off), `gufo bench --validate-prefill N` (built-in
  prefill-logit validation against the extension reference).
  The build ships NO logit-dump tool; reference logits for gate
  comparisons are therefore the frozen greedy argmax paths of the
  probe set plus `--validate-prefill` — recorded honestly as such.
- **Extension tests:** `tests/*.cpp` wire into the Gufo core CMake
  build via `extension.cmake` when BUILD_TESTING is on (target
  `gemma4_tests`, ctest names `gemma4.<name>`). Core clone:
  `~/src/gufo-dev` (CMakePresets include `gpu-test`, `cpu-test`).
  The nix shipping build does not run them; the test road is a
  core CMake preset build — still unpinned (Phase 1 update: the
  first test, KV-03's kv_cache_format test, is registered there
  and was verified by standalone compile; pinning the preset road
  remains open work for the next builder).
- **Model under test:** `/home/volmarr/models/gufo/gemma-4-26B-A4B-it-heretic.Q8_0.gguf`
  (SSD copy; sha256 1f2e89b8382fd2182ef3a7e15a5cf272722f767a8a08a58dd69ae36c52edbf1c).
  Env: GUFO_MODELS_DIR=/home/volmarr/models/gufo,
  GUFO_DEFAULT_MODEL=gemma-4-26B-A4B-it-heretic.Q8_0.gguf.
- **Machine rules:** MemAvailable checked before every build/bench
  (stand down under ~40 GiB), one heavy run at a time, detached
  units for long runs. gufo-resident.service and the machine's
  Ollama (:11434) are NOT touched in Phases 0-6.

## Benchmark protocol (KV-01 harness, `campaign/kv-harness/`)

- Driver: `campaign/kv-harness/run_matrix.sh <label>` runs
  `gufo bench` for the matrix below: 1 warmup invocation discarded
  + 5 measured invocations per config; prints per-run t/s and the
  median. Raw logs land in `campaign/kv-harness/results/<label>/`.
- Matrix (one invocation can carry several depths; driver uses
  separate invocations per config for clean medians):
  - prefill: `-p 512 -n 128 -d 0` (pp512), `-p 4096 -n 128 -d 0`
    (pp4096), `-p 16384 -n 128 -d 0` (pp16384)
  - decode: tg128 from the same runs at d0 (~512 ctx), plus
    `-p 512 -n 128 -d 3584` (tg128 @ ~4k ctx) and
    `-p 512 -n 128 -d 15872` (tg128 @ ~16k ctx)
- Probes: `campaign/kv-harness/probes.txt` (frozen set: factual
  recall x3, coherence x1, long-context recall x1), run via
  `campaign/kv-harness/run_probes.sh <label>` — `gufo prompt`
  greedy (temperature 0, seed 42, think off), full answers saved
  under `results/<label>/probes/`.
- KV bytes/token (analytic, from the model shape in
  docs/models/gemma-4-26b-a4b): sliding layer f16 = 8,192 B/token
  (K 4,096 + V 4,096); global layer f16 = 2,048 B/token V-store +
  up to 1,024 B/token rotated-K store. q8_0 targets (roadmap):
  sliding 4,352 B/token.

## Slice ledger

| Slice | Status | Evidence |
|---|---|---|
| KV-01 harness | DONE | campaign/kv-harness/ (run_matrix.sh, run_probes.sh, probes.txt) + this file |
| KV-02 f16 baseline | DONE | table below; raw logs results/baseline-f16/ (device-side) |
| KV-03 format field + allocation | DONE | ff8cf4f; section below |
| KV-04 GUFO_KV_CACHE_TYPE plumbing | DONE | f427c09 + format fix 5f59d2e; section below |
| KV-05 quantize-on-write K (sliding) | PENDING | |
| KV-06 quantize-on-write V + ring | PENDING | |
| KV-07 dequant-in-decode (sliding) | PENDING | |
| KV-08 decode correctness + speed gate | PENDING | |
| KV-09 global layers derived-K | PENDING | |
| KV-10 global validation gate | PENDING | |
| KV-11 non-WMMA prefill q8_0 | PENDING | |
| KV-12 WMMA prefill / dispatch rule | PENDING | |
| KV-13 machinery re-proof | PENDING | |
| KV-14 full-stack integration | PENDING | |
| KV-15 resident service re-proof | PENDING | |
| KV-16 soak + default decision + close | PENDING | |

## Baseline (KV-02) — f16, current build (4f73737)

Measured 2026-11-17; median of 5 (1 warmup discarded), machine
otherwise idle, MemAvailable 124,734,920 kB at start. Raw logs:
`campaign/kv-harness/results/baseline-f16/` (device-side).

| Metric | Median tok/s |
|---|---|
| Prefill pp512 | 2237.76 |
| Prefill pp4096 | 2939.85 |
| Prefill pp16384 | 2545.34 |
| Decode tg128 @ ~512 ctx | 42.69 |
| Decode tg128 @ ~4k ctx | 41.18 |
| Decode tg128 @ ~16k ctx | 39.90 |

KV bytes/token (f16, all 30 layers): **220,160 B (215.0 KiB)** —
25 sliding × 8,192 B + 5 global × 3,072 B (V-store 2,048 +
rotated-K store 1,024). Sliding growth is ring-capped in long
sessions; the figure is the nominal per-token rate. At 16k
context the KV regions total ≈ 3.52 GiB plus fixed per-session
extras (hidden/ahead/logits/draft) that do not scale with tokens.

Probe answers (greedy, seed 42; full text in
results/baseline-f16/probes/):
- FACT-1: `Paris` ✓ · FACT-2: `Au` ✓ · FACT-3: `366` ✓
- COH-1: two-sentence fox story, coherent and on-prompt ✓
- LONG-1: `47-12-89` ✓ (needle recalled; prompt_tokens 15,739)

Reference logits: no logit-dump tool ships; the frozen greedy
probe answers above are the reference. Short probes reproduced
byte-identically across three runs — the harness is deterministic.

Harness lessons (paid for once, recorded so nobody pays twice):
- `gufo prompt` has NO `--context` option; its session context is
  fixed at 4096 and the engine's "increase --context" error text
  does not apply to it. Long probes run through a throwaway
  `gufo serve llm --context 32768` (run_probes.sh).
- Serve's HTTP layer binds before the model loads; chat
  completions answer 503 "Loading model" until ready and
  /v1/models is not a readiness signal — the probe POST retries
  on 503.
- Size long-probe filler by measured tokens (usage.prompt_tokens),
  never by character estimates: a first "16k" draft measured
  36,459 tokens and overflowed with a clean 400.

## Phase gates

- Phase 0 gate: PASSED 2026-11-17 — harness committed (KV-01),
  baseline table + probe answers banked (KV-02).
- Phase 1 gate: PASSED 2026-11-17 — evidence below.

## Phase 1 (KV-03 + KV-04) — format field, allocation, plumbing

What landed:
- `kv_cache_format.hpp` (new, header-only): `KvCacheFormat`
  {kF16, kQ8_0, kQ4_0-reserved}, `KvCacheFormatName`,
  `ParseKvCacheType` (invalid values throw naming the value;
  q4_0 throws as reserved), `KvCacheRowBytes` (q8_0 = ggml blocks,
  34 B per 32 values, tail padded), `KvCacheStorageBytes`
  (per-region 256 B alignment, matching executor kAlign).
- KV-03: `KvCache.format` records the format; `Executor` takes it
  from `ModelOptions.kv_cache_format` (engine.cpp) and
  `CacheBytes` sizes K/V regions through `KvCacheStorageBytes`;
  `CreateCache` carves via `KvCacheRowBytes`. For f16 every figure
  reduces to the pre-campaign formula exactly. CreateCache refuses
  non-f16 with a named error until the write/read paths land
  (Phases 2–3): sized and plumbed, never silently corrupt.
- KV-04: `extension.json` declares `--kv-cache-type` (serve /
  bench / prompt / chat); `ApplyProviderOptions` resolves flag >
  GUFO_KV_CACHE_TYPE > silent f16 default into ModelOptions. An
  explicit selection prints `[gemma4] kv cache type: <name> (from
  <source>)` on stderr once per model load; an invalid value
  throws naming value + source. Diagnose-side format reporting
  rides the core slice in Phase 7 (KV-14), recorded there.
- Test: `tests/kv_cache_format_test.cpp`, registered as
  `gemma4.kv_cache_format` (extension.cmake + extension.json).
  Pins parse/name/row sizes and the storage math, including the
  KV-02 baseline figure (220,160 f16 B/token for the 26B-A4B
  geometry) and the q8_0 ring-scale figure (71,303,168 B for one
  sliding layer at 16,384 slots vs 134,217,728 B f16).

Verification (2026-11-17, device):
- nix build of the Phase 1 tree: clean, 2m07s wall (store path
  9gz90mf3…-gufo-0-unstable-4938016); clang-format clean on all
  touched files. NOTE for detached units: pass
  `--extra-experimental-features "nix-command flakes"` — a unit
  without the login shell's nix config fails instantly with
  "experimental Nix feature 'nix-command' is disabled" and the
  build silently does nothing (first attempt, 0.014 s, no-op).
- Host test: standalone compile of the registered test source
  (g++ -std=c++20 -Wall -Wextra, headers only) → PASS. The full
  CMake preset test build was not run in this slice; the test is
  wired for `ctest -R gemma4.kv_cache_format` on that road.
- f16 identity, Phase 1 binary (`result/bin/gufo`), full harness
  re-run (label phase1-f16): all five probe answers byte-identical
  to baseline (timing lines excluded); bench medians vs baseline —
  pp512 2242.28 (+0.20%), pp4096 2934.02 (−0.20%), pp16384
  2537.75 (−0.30%), tg512 42.71 (+0.05%), tg4096 41.19 (+0.02%),
  tg16384 39.91 (+0.03%) — noise, no regression.
- Plumbing battery (Phase 1 binary): `--kv-cache-type q8_0` →
  report line + session refusal naming q8_0; env
  GUFO_KV_CACHE_TYPE=q8_0 → same; env =bogus → fast error
  "GUFO_KV_CACHE_TYPE: unknown KV cache type 'bogus'; use f16 or
  q8_0"; `--kv-cache-type q4_0` → reserved error naming the flag;
  `--kv-cache-type f16` → report line + normal answer; bogus env
  + f16 flag → flag wins, normal answer.
- The installed profile (~/.nix-profile) still runs the
  pre-campaign build and gufo-resident.service is untouched, per
  the Phase 7 integration plan; Ollama (:11434) untouched.
