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
| KV-05 quantize-on-write K (sliding) | DONE | section below; roundtrip max_abs_err 0.010000, SNR 48.25 dB |
| KV-06 quantize-on-write V + ring | DONE | section below; V roundtrip max_abs_err 0.010000, SNR 48.15 dB; ring max_abs_err 0.006234 |
| KV-07 dequant-in-decode (sliding) | DONE | section below; q8_0 probes FACT-1/2/3 + LONG-1 identical to the f16 baseline |
| KV-08 decode correctness + speed gate | DONE | section below; probes byte-identical to f16 + Phase 0 baseline; decode +0.09/+0.27/+0.10 pct vs same-binary f16 (noise) |
| KV-09 global layers derived-K | DONE | section below; reconstruction max_abs_err 0.015822, SNR 47.14 dB; probes byte-identical to f16 |
| KV-10 global validation gate | DONE | section below; Phase 4 gate PASSED; probes identical excl load timing; matrix kv10-q8 vs kv10-f16 recorded |
| KV-11 non-WMMA prefill q8_0 | DONE | section below; probes byte-identical q8 vs f16; pp4096 +40.7 pct, pp16384 +103.8 pct vs KV-10 q8 |
| KV-12 WMMA prefill / dispatch rule | SUPERSEDED BY REDO 2026-11-17, GATE OPEN | section below; WMMA dequant-tile stage landed, probes byte-identical; pp16384 -2.98 pct vs f16 is real, beyond noise -- Phase 5 gate not yet passed |
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

## Phase 2 first slice (KV-05) -- q8_0 K quantize-on-write, sliding

What landed:
- `kv_cache_format.hpp`: host reference for the device store --
  `QuantizeQ8_0Block` / `DequantizeQ8_0Block` and the row forms
  `QuantizeQ8_0Row` / `DequantizeQ8_0Row` (ggml q8_0 layout: 32
  int8 codes + fp16 scale per 32 values, scale = amax / 127), plus
  `FloatToFp16Bits` / `Fp16BitsToFloat` for the block scales.
- `hip/kernels.hpp` + `hip/kernels.hip.cpp`: `QkvPostArgs` gains
  `k_cache_q8`. When set (and rotated_pairs == 0), QkvPost stores
  the K row for a slot as q8_0 blocks at slot * KvCacheRowBytes(
  kv_heads * head_dim, kQ8_0), one 34-byte block per 32 values in
  row order. In the kernel one block lies in one wave (lane holds
  dim 32 * b + lane), so WaveMax gives the block amax, each lane
  writes its own code, and lane 0 writes the fp16 scale -- the same
  values the host reference computes. The slot comes from the
  existing key/ring computation, so the store addresses linear and
  ring slots identically; V still stores binary16 in this slice.
- `hip/executor.cpp`: Forward passes `k_cache_q8` for sliding
  layers when `cache.format == kQ8_0`. Honest scoping: this is
  unreachable today, because CreateCache still refuses q8_0 --
  deliberately. The V store (KV-06) and the decode read path
  (KV-07) are not built, so no end-to-end q8_0 session can start
  and no consumer can read a half-converted cache. The refusal
  message now says exactly that. f16 behavior is unchanged (the
  f16 store branch is the old code, moved under an else).
- Test: `tests/kv_cache_format_test.cpp` extended (same registered
  `gemma4.kv_cache_format` test): fp16 bit roundtrips, block
  layout pins (constant block -> codes 127, scale fp16(1/127) at
  bytes 32..33; zero block exact), and the K-row roundtrip below.

Verification (2026-11-17, device):
- Standalone compile of the registered test (g++ -std=c++20
  -Wall -Wextra, headers only) -> PASS. Roundtrip of a 2048-value
  sliding K row (deterministic pseudo-random values in [-2, 2],
  one outlier block with amax 8): **max abs error 0.010000**
  (the outlier block's own small values, quantized at its scale
  8/127), **max abs error outside the outlier block 0.008366**,
  **SNR 48.25 dB**. Bounds asserted in the test: max <= 0.04,
  calm blocks <= 0.012, SNR >= 30 dB. These are the recorded
  error bounds for the Phase 2 gate's K half; KV-06 adds V.
- nix build of the KV-05 tree: clean (detached, ~4 min wall;
  store path qsdj2l6...-gufo-0-unstable-4938016). clang-format
  clean on all touched files.
- f16 spot check on the KV-05 binary: FACT-1 probe answers
  `Paris`, identical to the KV-02 baseline. `--kv-cache-type
  q8_0` prints the format report line, then the session refusal
  naming KV-05/KV-06/KV-07 -- no silent q8_0 mode.
- Installed profile and gufo-resident.service untouched;
  Ollama (:11434) untouched.

Next (KV-06): V rows quantize-on-write + the ring-buffer layout
proof for both stores; the Phase 2 gate closes there (no write
path still writing raw bf16 in q8_0 mode). KV-07 then builds the
decode read path, which is what may lift the CreateCache refusal.

## Phase 2 second slice (KV-06) -- q8_0 V quantize-on-write + ring

What landed:
- `hip/kernels.hpp` + `hip/kernels.hip.cpp`: `QkvPostArgs` gains
  `v_cache_q8`, mirroring `k_cache_q8`. When set (and
  rotated_pairs == 0), QkvPost stores the V row for a slot as q8_0
  blocks at slot * KvCacheRowBytes(kv_heads * head_dim, kQ8_0) --
  the same ggml block layout, the same one-block-per-wave WaveMax
  store, and the same existing slot/ring computation as the KV-05
  K store, so linear and ring slots are addressed identically.
  The f16 V store is the old code, moved under an else.
- `hip/executor.cpp`: Forward passes `v_cache_q8` for sliding
  layers when `cache.format == kQ8_0`, beside `k_cache_q8`.
  CreateCache still deliberately refuses q8_0 end-to-end -- only
  the decode read path (KV-07) is missing now, and the refusal
  message says exactly that.
- Write-path audit (the Phase 2 gate's second half): QkvPost is
  the only kernel that stores projected K/V rows. The attention
  kernels only read the caches; MoveKey and CopyRuns relocate
  already-stored rows byte-wise and never write bf16 from a
  projection (their re-proof over the quantized layout is KV-13).
  Global layers still store V as bf16: their V is the normalized
  K projection (rotated_pairs > 0), the derived-K design is
  Phase 4 (KV-09), and the executor passes `v_cache_q8` only for
  sliding layers. So in q8_0 mode no sliding-layer write path
  still writes raw bf16.
- Tests: `tests/kv_cache_format_test.cpp` extended (same
  registered `gemma4.kv_cache_format` test) with the V-row
  roundtrip and the ring-layout proof below.

Verification (2026-11-17, device):
- Standalone compile of the registered test (g++ -std=c++20
  -Wall -Wextra, headers only) -> PASS. V row (2048 values,
  deterministic pseudo-random in [-2, 2], one outlier block with
  amax 6): **max abs error 0.010000**, **max abs error outside
  the outlier block 0.008249**, **SNR 48.15 dB** -- inside the
  same asserted bounds as K (max <= 0.04, calm <= 0.012,
  SNR >= 30 dB). Ring proof: 20 positions stored at
  slot = position % 8 over two full wraps roundtrip their last
  writers with **max abs error 0.006234**, and a constant row at
  a wrapped position lands at the exact byte offsets of its slot
  (codes at slot*2176 + block*34, fp16 scale at +32, neighbouring
  slots untouched).
- nix build of the KV-06 tree: clean (detached, ~4 min wall;
  store path 7sb5c12c...-gufo-0-unstable-4938016). clang-format
  clean on all touched files.
- f16 spot check on the KV-06 binary: FACT-1 probe answers
  `Paris`, identical to the KV-02 baseline. `--kv-cache-type
  q8_0` prints the format report line, then the session refusal
  naming KV-07 only -- no silent q8_0 mode.
- Installed profile and gufo-resident.service untouched;
  Ollama (:11434) untouched.

**Phase 2 gate: PASSED** 2026-11-17 -- roundtrip bounds recorded
for both halves (K: max 0.010000, calm 0.008366, SNR 48.25 dB;
V: max 0.010000, calm 0.008249, SNR 48.15 dB; ring rows max
0.006234), and no sliding-layer write path still writes raw
bf16 in q8_0 mode. Next (KV-07): the decode read path
(dequant-in-decode for sliding layers), which is what may lift
the CreateCache refusal.

## Phase 3 first slice (KV-07) -- dequant-in-decode, sliding

What landed:
- `hip/kernels.hpp`: `AttentionArgs` gains `k_cache_q8` /
  `v_cache_q8`, mirroring `QkvPostArgs` -- set only for sliding
  layers of a q8_0 cache (never with derived keys).
- `hip/kernels.hip.cpp`: in-kernel dequantization on every decode
  read path for sliding layers. `DequantQ8Lane` (value = int8 code
  x fp16 block scale; a lane's P values always lie in one 32-value
  block) serves the single-pass `AttendKeys` (dequantizes to FP32)
  and the split-K kernel's own-key step. The split-K `load()`
  dequantizes into its binary16 transport, so the score/value
  loops consume q8_0 rows exactly as f16 rows (dequantized values
  land at the cache's native binary16 precision there). At
  prefill widths a q8_0 sliding layer takes the single-pass
  kernel instead of the WMMA kernel, which reads binary16 tiles
  (the WMMA dequant-tile stage is KV-12); the executor gives that
  call its FP32 output and the existing narrow step fills the
  binary16 copy the prefill projection consumes.
- Mixed cache, made explicit: `KvCacheLayerFormat` in
  `kv_cache_format.hpp` -- sliding layers take the cache format,
  global layers stay binary16 until KV-09. `CacheBytes` and the
  `CreateCache` carve both size per layer through it (new
  `KvCacheStorageBytesPerLayer`); at 1 slot the mixed figure is
  130,560 B/token (25 sliding q8_0 + 5 global f16), pinned in the
  extended `gemma4.kv_cache_format` test. This also fixes the
  Phase 1 sizing, which would have carved global regions at q8_0
  sizes under their binary16 stores. `CreateCache` no longer
  refuses q8_0; q4_0 stays a named reserved refusal.
- `hip/executor.cpp`: Forward and DraftChains pass the q8_0
  pointers for sliding layers (draft attention reads the shared
  source layer). `MoveKey` now copies the STORED row -- widths in
  uint16 units of stored bytes (KvCacheRowBytes / 2) -- so a
  sibling commit moves a q8_0 row's exact 2,176 bytes; binary16
  layers move exactly what they moved before.
- NOT in this slice, by the roadmap's own staging: continuation
  snapshots still size rows binary16 (engine.cpp VisitBody /
  SnapshotBytes) -- q8_0 sessions must not be snapshotted until
  KV-13 re-proves that machinery; the WMMA prefill dequant-tile
  stage is KV-12; the Phase 3 gate (probe/logit comparison +
  decode matrix vs baseline) is KV-08.

Verification (2026-11-17, device):
- Standalone compile of the registered test (g++ -std=c++20
  -Wall -Wextra, headers only) -> PASS, including the new mixed
  storage pins (130,560 B/token mixed; per-layer road reproduces
  the uniform f16/q8_0 figures).
- nix build of the KV-07 tree: clean (detached unit, ~4 min wall;
  store path 8mk12m3f...-gufo-0-unstable-4938016). clang-format
  clean on all touched files.
- f16 spot check on the KV-07 binary: FACT-1 answers `Paris`,
  identical to the KV-02 baseline.
- q8_0 end to end on the KV-07 binary (`GUFO_KV_CACHE_TYPE=q8_0`,
  full harness, results/kv07-q8/ device-side): FACT-1 `Paris`,
  FACT-2 `Au`, FACT-3 `366`, LONG-1 `47-12-89` (15,739 prompt
  tokens through serve) -- all byte-identical to the f16
  baseline. COH-1 coherent and on-prompt, wording drifted from
  the baseline (expected quantization drift on an open-ended
  greedy path; the formal drift measurement is KV-08's gate).
- Bench smoke (single run, NOT a gate measurement): q8_0 pp512
  1851.89 t/s, tg32 42.14 t/s (f16 baseline tg128 @ ~512 ctx:
  42.69). Prefill runs the single-pass sliding path until
  KV-11/KV-12; decode is in the baseline's ballpark. The speed
  gate itself is KV-08.
- Error paths re-checked: env `bogus` -> named error; flag
  `q4_0` -> reserved error. Installed profile and
  gufo-resident.service untouched; Ollama (:11434) untouched.

Next (KV-08): the Phase 3 gate -- probe set + logit comparison
vs the Phase 0 reference and the decode matrix vs baseline at
512 / 4k / 16k.

## Phase 3 gate slice (KV-08) -- decode correctness + speed gate

What was measured (2026-11-17, device, KV-07 binary
`result` -> /nix/store/bczcs2gx...-gufo-0-unstable-4938016;
MemAvailable ~91 GB at run start):

- Full harness matrix run twice on the same binary: label
  kv08-q8 (GUFO_KV_CACHE_TYPE=q8_0, 07:52) and label kv08-f16
  (08:01); 1 warmup + 5 measured per config, medians below.
  Raw logs device-side in results/kv08-q8/ and results/kv08-f16/.
- Probes (greedy, seed 42): FACT-1 Paris, FACT-2 Au,
  FACT-3 366, LONG-1 47-12-89, and COH-1 -- the q8_0 answers
  are byte-identical to the same-binary f16 run AND to the
  Phase 0 baseline, all five, including the open-ended COH-1
  (KV-07 smoke run had shown COH-1 wording drift; the full
  gate run shows none).
- Logit comparison: no logit-dump tool ships (KV-01 note), so
  the comparison is the frozen greedy argmax paths above.
  --validate-prefill 512 was run on this binary
  (results/kv08-f16/validate-prefill.log): core bench parses
  the flag but only the qwen drivers consume it -- for gemma4
  it is inert (plain bench table, no validation output), so
  no claim is made from it.
- Decode medians (tok/s), q8_0 vs same-binary f16: tg512
  42.70 vs 42.66 (+0.09%), tg4096 41.32 vs 41.21 (+0.27%),
  tg16384 39.84 vs 39.80 (+0.10%). Vs the Phase 0 baseline
  (42.69 / 41.18 / 39.90): +0.02% / +0.34% / -0.15%. All
  within run-to-run noise -- no decode regression at
  512 / 4k / 16k.
- Long-context, recorded either way as the gate requires:
  NO decode speed improvement at 16k (tg16384 +0.10% vs
  same-binary f16). The q8_0 win at this phase is memory,
  not speed: mixed cache 130,560 B/token vs 220,160 f16
  (-40.7%; KV-07). Prefill on this pair (Phase 5 owns that
  gate): pp512 2224.94 vs 2200.08, pp4096 2915.90 vs
  2919.11, pp16384 2529.54 vs 2536.48 -- noise.
- Installed profile and gufo-resident.service untouched;
  Ollama (:11434) untouched.

**Phase 3 gate: PASSED** 2026-11-17 -- probes pass
byte-identical; no decode regression beyond noise at
512 / 4k / 16k; long-context effect measured and recorded
(speed-neutral, memory -40.7%). Next (KV-09): global
layers, the derived-K design (Phase 4).

## Phase 4 first slice (KV-09) -- q8_0 global layers, derived-K

Design (the derived-K question, answered from the code):
- A global layer stores two rows per token: the V row (2 heads x
  512 values = 1,024 values; the RMS-normalized K projection, no
  weight, no rope) and the rotated-K row (2 heads x 128 packed
  values = 256 values: per head the rotated dims [0, 64) then
  [256, 320) of K = rope(k_norm . V); pairs = 64).
- Feasibility: rope is applied BEFORE quantization in both stores
  and is orthogonal, so reconstruction adds no error of its own:
  rotated dims carry only their own store's quantization error;
  unrotated dims are the dequantized V (the query carries k_norm,
  as in the f16 design). No re-rotation of quantized values ever
  happens. The host reconstruction proof below lands in the same
  error family as the sliding stores (SNR 47.14 dB vs 48.25/48.15),
  so the design is feasible within tolerance -- no fallback is
  needed from this slice; KV-10 measures it through the real path.

What landed:
- `kv_cache_format.hpp`: `KvCacheLayerFormat` now returns the
  cache format for every layer (the KV-07 mixed cache is retired);
  q8_0 storage is the full 125,440 B/token (was 130,560 mixed;
  f16 stays 220,160), pinned in the extended test.
- `hip/kernels.hip.cpp` QkvPost: the V store branch now serves
  derived-key layers too (their V row quantizes exactly like a
  sliding V row), plus a new rotated-K branch quantizing the
  packed rotated dims after rotation -- the segment/wave mapping
  puts one 32-value block in one wave (WaveMax amax, lane 0 writes
  the fp16 scale, the KV-05 pattern). `hip/executor.cpp` passes
  the q8_0 pointers for every layer of a q8_0 cache (Forward and
  DraftChains), and gives q8_0 layers their FP32 output at
  prefill widths.
- Decode read path (split-K, staged tiles): `stage_load` stages
  the V tile by dequantizing the q8_0 V store into the same
  binary16 tile the f16 path stages; `rot_load` dequantizes the
  packed rotated store into the rotated tile; the own-key step
  reconstructs from the two stores via the new
  `DequantQ8PackedLane`. Prefill: a q8_0 global layer takes the
  single-pass kernel like a q8_0 sliding layer, where
  `AttendKeys` reconstructs derived keys from the two stores;
  the WMMA dequant-tile stage remains KV-12, and f16 global
  prefill still takes WMMA exactly as before.
- `MoveKey` needed no change: it already moves stored-row
  widths through `KvCacheLayerFormat`, so global rows now move
  their exact q8_0 bytes (272 B rotated-K row, 1,088 B V row).
  Guard: the Executor refuses a q8_0 cache by name when a
  derived-key layer's rotated pairs are not the production 64
  (the geometry the WMMA path and the staged tiles hardcode).
- NOT in this slice: KV-10's validation gate (reconstruction
  error through the real attention path + matrix re-run), and
  KV-13's snapshot re-proof -- continuation snapshots still size
  rows binary16 in engine.cpp, as recorded at KV-07, so q8_0
  sessions must not be snapshotted until KV-13.

Verification (2026-11-17, device):
- Standalone compile of the registered test -> PASS, including
  the new derived-K reconstruction check (one global head,
  K = rope(k_norm . V) rebuilt from the two q8_0 stores):
  **max abs error 0.015822** (bound 0.04), **SNR 47.14 dB**
  (bound 30 dB) -- the sliding stores measured 48.25/48.15 dB.
- nix build of the KV-09 tree: clean (detached unit; store path
  6h60fbr9...-gufo-0-unstable-4938016). clang-format clean on
  all touched files.
- Probes on the KV-09 binary (full harness, labels kv09-f16 /
  kv09-q8, results device-side): all five answers byte-identical
  between q8_0 and same-binary f16, and f16 byte-identical to
  the KV-08 baseline (timing lines excluded) -- FACT-1 `Paris`,
  FACT-2 `Au`, FACT-3 `366`, COH-1 identical text, LONG-1
  `47-12-89` (15,739 prompt tokens through serve; prefill and
  decode both exercised the global q8_0 paths). Error paths
  unchanged: env `bogus` -> named error; flag `q4_0` ->
  reserved error.
- Installed profile and gufo-resident.service untouched;
  Ollama (:11434) untouched.

Next (KV-10): the Phase 4 validation gate -- reconstruction
error through the real attention path, probes + matrix re-run,
and the gate call (full q8_0 vs the mixed-cache fallback) on
that evidence.

## Phase 4 gate slice (KV-10) -- global validation gate

Run 2026-11-17 09:04-09:35 EST on purpleraider, KV-09 binary
(/nix/store/6h60fbr9...-gufo-0-unstable-4938016/bin/gufo),
MemAvailable 124,560,356 kB at start. Detached /tmp/kv10_run.sh:
matrix + probes re-run for kv10-q8 and kv10-f16, KV10_RUN_DONE,
lock self-removed on completion. Collected and banked by the
09:44 keeper tick (no lock, no live bench at collection).

Reconstruction through the real attention path (KV-09 evidence,
re-validated here end-to-end): max_abs_err 0.015822, SNR 47.14 dB
on the global derived-K reconstruction.

Probes (greedy, seed 42): all 5 byte-identical q8_0 vs f16
excluding the [Model Load] timing line -- FACT-1 Paris,
FACT-2 Au, FACT-3 366, COH-1 identical text, LONG-1 47-12-89
(15,739 prompt tokens; global layers exercised at length).

Matrix medians (tok/s, 5 runs, 1 warmup discarded):
- pp512: q8_0 1713.89 vs f16 2189.33 (-21.7 pct)
- pp4096: q8_0 1029.10 vs f16 2906.45 (-64.6 pct)
- pp16384: q8_0 534.10 vs f16 2526.15 (-78.9 pct)
- tg512: q8_0 37.97 vs f16 41.92 (-9.4 pct)
- tg4096: q8_0 36.78 vs f16 40.53 (-9.3 pct)
- tg16384: q8_0 35.38 vs f16 39.15 (-9.6 pct)
Speed recorded honestly, not fudged: prefill in q8_0 mode is
currently far slower -- that is exactly Phase 5 work (KV-11
non-WMMA prefill, KV-12 WMMA / dispatch rule, whose gate is
no-regression vs f16). Decode is ~9-10 pct slower with global
layers also quantized (KV-08 sliding-only was noise-level).
The KV-10 gate itself is the correctness/reconstruction gate
in the roadmap text, and it passes on the numbers above.

**Phase 4 gate: PASSED** 2026-11-17 -- global layers quantized
within tolerance (reconstruction + all probes identical through
the real attention path); the mixed-cache fallback is NOT
needed. Full q8_0 ships forward to Phase 5.

Next (KV-11): Phase 5 -- non-WMMA prefill reads/writes q8_0.

## Phase 5 first slice (KV-11) -- non-WMMA prefill q8_0

Audit (2026-11-17, device): prefill writes already quantize at
every width -- QkvPost is width-agnostic and the executor passes
the q8_0 store pointers for every layer (KV-05/KV-06/KV-09).
The read side was the interim KV-07/KV-09 routing: at prefill
widths a q8_0 layer took the single-pass AttentionKernel, whose
AttendKeys dequantizes every key once per (row, head) -- the
KV-10 matrix priced that at pp16384 534.10 t/s.

What landed:
- `hip/kernels.hip.cpp` LaunchAttention: at prefill widths
  (rows > kSplitRows) a q8_0 layer now runs the split kernel in
  kSplitRows-row blocks (per-block AttentionArgs: q/out and
  key_ends offset, first_position advanced, pack_row0 offset,
  out_half narrowed once over the full rows afterwards). The
  split kernel's staged tiles dequantize each q8_0 cache tile
  once for the rows and query heads sharing the block -- at
  hd512 the tile is staged in shared memory and serves all
  eight query heads of the KV head -- and its per-row
  arithmetic is batch-invariant, so a prefill row computes
  exactly what the decode path computes for it (the KV-07/KV-09
  read paths, derived-key reconstruction included; chunked
  prefill via prefill_chunk and multi-chunk sessions ride the
  same road). Shapes the split kernel does not take (non-
  production head grouping, shared-position batches) keep the
  single-pass AttendKeys road as fallback. The f16 route is
  untouched: WMMA first, single-pass on decline, exactly as
  before. The WMMA q8_0 dequant-tile stage / dispatch rule is
  KV-12, not this slice.
- `hip/executor.cpp`: comment updated (KV-07/KV-11 non-WMMA
  path). No behavior change there.

Verification (2026-11-17, device; binary result ->
/nix/store/95kf8hkd...-gufo-0-unstable-4938016;
MemAvailable ~124.5 GiB at start):
- nix build: clean (detached, ~4 min wall). clang-format clean
  on both touched files. Standalone compile of the registered
  kv_cache_format test -> PASS (roundtrip/reconstruction
  numbers unchanged: K 0.010000 / 48.25 dB, V 0.010000 /
  48.15 dB, ring 0.006234, derived-K 0.015822 / 47.14 dB).
- Probes (full harness, labels kv11-q8 / kv11-f16, results
  device-side): all five byte-identical q8_0 vs same-binary
  f16 excluding the [Model Load] timing line, and both
  identical to the KV-10 runs -- FACT-1 `Paris`, FACT-2 `Au`,
  FACT-3 `366`, COH-1 identical text, LONG-1 `47-12-89`
  (15,739 prompt tokens through serve; prefill_chunk 2048, so
  eight prefill chunks read earlier chunks' q8_0 cache through
  the new path, sliding window/ring and global derived-K
  layers alike).
- Matrix medians (tok/s, 5 runs, 1 warmup discarded), KV-11
  binary, q8_0 vs same-binary f16 (KV-10 q8 in parens):
  pp512 1610.81 vs 2201.36 (1713.89; -6.0 pct vs KV-10 q8 --
  a small short-context cost of the block-launch road),
  pp4096 1447.54 vs 2917.83 (1029.10; +40.7 pct),
  pp16384 1088.75 vs 2527.32 (534.10; +103.8 pct),
  tg512 37.93 vs 42.02, tg4096 36.82 vs 40.61,
  tg16384 35.43 vs 39.26 (decode within +-0.1 pct of KV-10
  q8 -- the decode path is untouched). f16 vs KV-10 f16:
  within +0.05..+0.55 pct on all six configs -- noise, f16
  unchanged. q8_0 prefill still trails the WMMA f16 path
  (-26.8 / -50.4 / -56.9 pct); closing that, or dispatching
  around it, is the KV-12 gate's work and is NOT claimed here.
- Error paths unchanged: env `bogus` -> named error; flag
  `q4_0` -> reserved error. Installed profile and
  gufo-resident.service untouched; Ollama (:11434) untouched.

Next (KV-12): WMMA matrix-core prefill dequant-tile stage, or
the measured dispatch rule (q8_0 prefill takes the faster path
per context size, crossover measured); the Phase 5 gate
(prefill matrix vs the matrix-core f16 baseline, no regression
beyond noise) is decided there.

## Phase 5 gate slice (KV-12) -- WMMA prefill dequant-tile stage

What landed (working tree built and measured 2026-11-17 by the
prior builder; collected, verified and banked by the 11:54
keeper tick -- no lock and no live bench at collection, the
10:43 lock was stale, matrices had finished 11:52):
- `hip/attention_wmma.hip.cpp` + `.hpp`: the WMMA matrix-core
  prefill kernel gains a q8_0 dequant-tile stage -- each K/V
  tile dequantizes once into LDS binary16 and feeds the matrix
  cores, exactly the tile the f16 road stages. Shapes the stage
  does not stage exactly return false.
- `hip/kernels.hip.cpp` LaunchAttention: at prefill widths a
  q8_0 layer tries the WMMA dequant-tile road first; a declined
  shape falls through to the KV-11 split-kernel road.
- `hip/executor.cpp`: a q8_0 layer at prefill widths again keeps
  the null FP32 output for a binary16-only output projection
  (the WMMA road writes the binary16 copy directly, like f16);
  the non-WMMA fallback throws for its FP32 output there,
  exactly as the f16 fallback does.

Verification (2026-11-17, device; binary result ->
/nix/store/g2mwszad...-gufo-0-unstable-4938016, built from
this tree 11:34; MemAvailable ~124 GiB at both runs):
- clang-format clean on all four touched files (keeper
  re-checked). Standalone compile of the registered
  kv_cache_format test -> PASS, numbers unchanged (K 0.010000 /
  48.25 dB, V 0.010000 / 48.15 dB, ring 0.006234, derived-K
  0.015822 / 47.14 dB).
- Probes (full harness, labels kv12-q8 / kv12-f16, results
  device-side): all five answers byte-identical q8_0 vs
  same-binary f16 -- FACT-1 Paris, FACT-2 Au, FACT-3 366,
  COH-1 identical text, LONG-1 47-12-89. The .err diffs are
  only the expected kv-cache-type report line and timestamps.
- Prefill matrix medians (tok/s, 5 runs, 1 warmup discarded),
  KV-12 binary, q8_0 vs same-binary f16 (KV-11 q8 in parens):
  pp512 2208.62 vs 2192.60 (+0.73 pct; KV-11 q8 1610.81),
  pp4096 2884.23 vs 2918.58 (-1.18 pct; KV-11 q8 1447.54),
  pp16384 2448.47 vs 2523.66 (-2.98 pct; KV-11 q8 1088.75).
  The WMMA stage closes almost all of the KV-11 gap
  (-26.8 / -50.4 / -56.9 pct at KV-11). Decode unchanged in
  family: tg512 37.98 vs 42.08, tg4096 36.77 vs 40.59,
  tg16384 35.44 vs 39.31 (the ~-9.5 pct global-q8 decode cost
  recorded at KV-10; decode is not this phase gate).
- Gate call, recorded honestly: intra-run spread on pp16384
  is under 0.4 pct on both labels, so the -2.98 pct at 16k is
  a REAL small regression, not noise. The Phase 5 gate text
  (no regression beyond noise vs the matrix-core f16
  baseline) is therefore NOT yet passed. pp512 and pp4096
  are within ~1 pct. Installed profile and
  gufo-resident.service untouched; Ollama (:11434) untouched.

Next: close the pp16384 gap (dequant-tile staging cost at
long context is the named suspect) and re-run the pp16384
pair; the Phase 5 gate is decided on that evidence. KV-13
(Phase 6 machinery re-proof) does not start until the gate
call is made.

## KV-12b -- packed-load staging tweak (measured on the superseded design)

Landed by the 12:04 keeper tick (its first builder dispatch
stopped when a chunked patch transfer was removed mid-flight;
the keeper worked the slice directly). NOTE: this was built
and measured against the KV-12 staging design that the Phase
5 REDO (above, 3ca9f6a) supersedes; it is banked as measured
history and because DequantQ8Chunk carries into the redo's
R2 arithmetic unchanged.
- `hip/attention_wmma.hip.cpp` DequantQ8Chunk: the chunk's
  eight int8 codes now come from ONE packed 8-byte load
  (memcpy; q8_0 block starts are 34-byte multiples, not
  8-aligned) instead of eight byte loads. Codes, scale and
  products unchanged -- staged tiles bit-for-bit identical;
  clang-format clean.
- Verification (2026-11-17, device; binary result ->
  /nix/store/kbwmhx1m...-gufo-0-unstable-4938016; MemAvailable
  ~121 GiB): probes kv12b-q8 vs kv12b-f16 -- all five answers
  identical, load-timing lines excluded (Paris / Au / 366 /
  COH-1 same text / 47-12-89). Prefill medians (tok/s, 5
  runs): pp512 2204.25 vs 2190.02 (+0.65 pct), pp4096
  2912.51 vs 2895.29 (+0.59 pct, was -1.18), pp16384
  2463.46 vs 2525.32 (-2.45 pct, was -2.98; run spreads
  ~0.5 pct, so still real). Decode q8 unchanged in family
  (37.93 / 36.76 / 35.46). Installed profile, resident
  service and Ollama (:11434) untouched.
- Standing: the old Phase 5 gate was not passed by this
  tweak and the redo's raised gate now governs. Next per
  the redo plan: R1 layout audit, then R2-global.

## Phase gates

- Phase 0 gate: PASSED 2026-11-17 — harness committed (KV-01),
  baseline table + probe answers banked (KV-02).
- Phase 1 gate: PASSED 2026-11-17 — evidence below.
- Phase 2 gate: PASSED 2026-11-17 — evidence in the KV-06 section above.
- Phase 3 gate: PASSED 2026-11-17 -- evidence in the KV-08 section above.
- Phase 4 gate: PASSED 2026-11-17 -- evidence in the KV-10 section above.
- Phase 5 gate: OPEN, REDO R5 RUN FAILED 2026-11-17 -- pp512 +0.78 pct, pp4096 -0.15 pct (not faster), pp16384 -3.24 pct vs same-binary f16; probes byte-identical, host test PASS. R7 loop law: new approach round starts; Phase 6 does not start.

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

## Phase 5 REDO — commissioned 2026-11-17 (Volmarr's order)

Volmarr's word, 2026-11-17 12:20 EST: Phase 5 is redone in
full on a researched new approach; 12:25 EST: the loop repeats
with new approaches until success. The KV-12 dequant-tile
STAGING approach above is SUPERSEDED — its commit (81190ea)
stays as history, its gate stays OPEN and unpassed.

Redo plan: `campaign/KV_PHASE5_REDO.md` (research report +
ChatGPT consultation synthesis; both converged on the same
first move). In brief: R1 layout audit (separate scale plane,
per-head-contiguous tile-major, only if reads do not already
coalesce) → R2 fused dequant-on-load into the LDS tile,
GQA-group-shared, manual fragment construction, global d512
first then sliding d256, single- vs double-buffer measured →
R3 current-chunk round-trip bypass with the quantize->
dequantize transformation reproduced in-register (raw
unquantized values FORBIDDEN) → R4 derived-K globals loaded
+ dequantized once per tile. R6 bounded fallbacks (E1 fused-V
scale-folding, E2 int8-WMMA QK spike) only if R1–R4 land
short, each with its decision rule recorded here first.

Redo gate (raised bar): q8_0 STRICTLY FASTER than same-binary
f16 beyond noise at pp512 AND pp4096 AND pp16384, all 5
probes byte-identical, kv_cache_format test PASS. Redo slices
track under KV-11/KV-12. Next step for the next builder/keeper
tick: R1 layout audit of the QkvPost write path and the WMMA
tile reads, recorded here, then R2-global.

## Phase 5 REDO R1 -- layout audit (2026-11-17, keeper tick 12:44 EST)

Audited on device at banked head 6bab2c8. MemAvailable ~124 GiB
at tick open; no build/bench this tick (audit only); Ollama
(:11434) and gufo-resident.service untouched.

Writer (hip/kernels.hip.cpp QkvPostKernel, KV-05/06/09):
- K, V and the packed rotated-K row are all written in the
  interleaved ggml q8_0 block layout: row stride
  kv_heads * (D/32) * 34 bytes (rotated row:
  kv_heads * 2 * (pairs/32) * 34); block index
  kvh * (D/32) + dim/32; each 34-byte block holds 32 int8
  codes at bytes [0,32) and the fp16 scale at bytes [32,34).
  There is NO separate scale plane and NO tile-major order:
  the audit precondition in KV_PHASE5_REDO.md R1 is met --
  the layout IS the interleaved one.

WMMA reader (hip/attention_wmma.hip.cpp DequantQ8Chunk):
- Same block arithmetic: block = cache + slot*row_bytes +
  (kvh*(per_head/32) + dim0/32) * 34; scale read from
  block+32; the chunk's 8 codes come from one packed 8-byte
  memcpy at block + dim0%32 (KV-12b).
- Coalescing verdict: reads do NOT coalesce cleanly. In the
  tile fetch, consecutive lanes cover different keys, so a
  wave's chunk loads are 8-byte loads strided by the full
  row_bytes; the four 8-value chunks of one 32-value block
  are issued by different lanes, and the block's fp16 scale
  is re-read once per chunk (4x per block). Useful bytes per
  block are eventually all consumed, but no single wave
  request spans contiguous addresses, unlike the f16 road's
  16-byte uint4 loads. Recorded honestly: this is strided
  gather, not coalesced streaming.

Blast radius of a layout change (why R1 is not a one-file
edit): every consumer of the KV q8_0 layout must agree --
QkvPostKernel writer (3 stores), the WMMA reader, the
KV-11 split-kernel readers and the decode dequant helpers
in hip/kernels.hip.cpp (lines ~797, ~825, ~1092, ~1185,
~1256, ~1320), kv_cache_format.hpp row-byte math
(KvCacheRowBytes / kQ8_0BlockBytes), the host
QuantizeQ8_0Row reference, and the kv_cache_format test's
pinned byte figures. Phases 2-4 gates passed on THIS
layout, so a layout switch must re-run the decode and
validation spot gates, not only the Phase 5 matrix.

Decision (recorded before R2 starts, per the redo plan):
R1 layout change is WARRANTED by this audit, and it is
sequenced as the first step of the R2-global slice, under
a versioned layout flag, so the fused dequant-on-load
loader is written once against the new separate-scale-
plane, per-(head, token-tile, d)-contiguous layout instead
of twice. Next tick: R2-global step 1 -- layout flag +
writer + scale plane, decode/split readers switched in
the same slice, kv_cache_format test updated and PASS,
probes byte-identical before any speed claim.

## Phase 5 REDO R2-global step 1 -- versioned split-scale-plane layout (2026-11-17, builder tick 13:03 EST)

Layout v2 landed, writer and ALL readers switched in one slice,
per the R1 decision. No fused loader yet (that is step 2, written
against this layout); no speed claim this step -- the gate is
byte identity, and it passes.

- Versioning: `KvQ8Layout` in kv_cache_format.hpp -- v1
  kInterleavedV1 (superseded ggml 34-byte blocks, scale inline at
  bytes [32,34)), v2 kSplitScaleV2 = kKvQ8LayoutCurrent;
  `KvCacheQ8LayoutVersion(format)` reports 2 for q8_0, 0 else.
  Caches never outlive the process, so no migration exists; the
  flag makes any future layout change an explicit version bump.
- New layout (per row, per token): int8 payload contiguous --
  32 bytes per 32-value block, blocks in row order (per head,
  then d_block) -- followed by a separate fp16 scale plane,
  2 bytes per block indexed [token][head][d_block]. Host helpers
  Q8_0PayloadOffset / Q8_0ScaleOffset / Q8_0BlockCount;
  QuantizeQ8_0Block now emits codes and scale separately;
  QuantizeQ8_0Row / DequantizeQ8_0Row place them in the planes.
  Row totals UNCHANGED (blocks * 34): every KvCacheRowBytes /
  storage figure in the test stands as pinned.
- Device: hip/kernels.hip.cpp gains Q8RowBytes / Q8BlockCodes /
  Q8BlockScale mirrors. Switched: QkvPost writer (K, V and the
  packed rotated-K stores), DequantQ8Lane / DequantQ8PackedLane
  (decode AttendKeys + own-key step), the KV-11 split-kernel
  readers (non-staged load lambda, staged V tile, staged
  rotated-K load). hip/attention_wmma.hip.cpp DequantQ8Chunk
  reads codes from the payload plane and the scale from the
  scale plane (row block count = row_bytes / 34). kernels.hpp
  layout docs updated. MoveKey / snapshots copy whole rows
  opaquely and needed no change.
- Verification (2026-11-17, device; binary result ->
  /nix/store/j9gncxvl...-gufo-0-unstable-4938016, derivation
  pdalz5ri... from this tree; MemAvailable ~124 GiB):
  clang-format clean on all five touched files. Host
  kv_cache_format test PASS with roundtrip numbers IDENTICAL
  to the old layout (K 0.010000 / 48.25 dB, V 0.010000 /
  48.15 dB, ring 0.006234, derived-K 0.015822 / 47.14 dB) --
  the arithmetic is unchanged, only the byte arrangement.
  Probes (full harness, labels r2s1-q8 / r2s1-f16, results
  device-side, untracked per convention): all five answers
  byte-identical q8_0 vs same-binary f16, model-load timing
  line excluded as before -- FACT-1 Paris, FACT-2 Au, FACT-3
  366, COH-1 identical text, LONG-1 47-12-89 -- and identical
  to the KV-12b answers recorded on layout v1. This re-proves
  the Phases 2-4 decode/validation spot gates on layout v2.
  Installed profile and gufo-resident.service untouched;
  Ollama (:11434) untouched.

Next: R2-global step 2 -- the fused dequant-on-load loader
written against this layout (global d512 first), then the R5
gate run under the raised bar.

## Phase 5 REDO R2-global step 2 -- fused dequant-on-load loader, global d512 (2026-11-17, builder tick 13:28 EST)

The fused loader landed for the global (d512, kGlobal) WMMA
prefill path, written against layout v2. Two designs were
built and measured this tick; only the second is banked.

Banked design (fetch-side fusion, proven stores untouched):
- `DequantQ8Packed(packed, scale)` factors the 8-value dequant
  out of `DequantQ8Chunk` (same bf16(int8) * bf16(scale)
  arithmetic, unchanged); `DequantQ8Chunk` now delegates to it,
  so every existing caller stages bit-identical values.
- `LoadQ8BlockRaw` loads one full 32-value block: codes as two
  uint4 words from the contiguous payload plane, the fp16
  scale bits once from the scale plane.
- In the global full-tile q8 fetch, the four chunks of a block
  sit in lanes tid, tid+8, tid+16, tid+24 (same key pair), so
  the chunk-0 lane loads the whole block for both keys of its
  pair and the raw words + scale bits are distributed inside
  the wave with __shfl; each lane dequantizes exactly its own
  chunk into the same v_next registers the KV-12 road filled.
  Per block: one 32-byte code load + one scale load, where the
  per-chunk road issued four 8-byte gathers and re-read the
  scale four times. No global bf16 scratch, no separate
  staging pass; the LDS store paths, register footprint and
  f16 road are byte-for-byte the KV-12 ones.
- Scope kept: partial tiles keep the per-chunk road (their
  bounds logic is unchanged); the rotated-K chunks keep the
  per-chunk road (their chunk lanes span waves, so no in-wave
  distribution exists); sliding d256 is untouched (its step
  follows R2-global). R4's rule holds: the V block's values
  still serve both the K-from-V dims and the V tile from the
  same registers -- nothing is read or dequantized twice.
  Single- vs double-buffer measurement was NOT taken this
  step: the register-prefetch structure (fetch next tile
  before the current tile's stores) is unchanged from KV-12,
  so there was no new buffering choice to measure.

Rejected design (built, measured, reverted before banking):
a wider fusion -- each thread dequantizing a full V block
into new fused registers, with new K/V^T LDS store paths
(scalar transposed stores, DPP full-block exchange) -- was
bit-correct (all probes identical) but SLOW: same-binary
medians pp512 2201.68 vs f16 2196.65 (+0.23 pct), pp4096
2713.67 vs 2932.04 (-7.45 pct), pp16384 2020.83 vs 2523.04
(-19.90 pct). The regression grew with context, i.e. with
attention share, indicting the new store paths / wider live
registers, not the loads. It was reverted in full; the
banked design keeps the proven stores. Recorded so the
approach is not retried blind.

Verification (2026-11-17, device; binary result ->
/nix/store/vc1p846n...-gufo-0-unstable-4938016; MemAvailable
~124 GiB throughout; Ollama :11434 and gufo-resident.service
untouched):
- nix build clean (detached, exit 0); clang-format clean on
  the one touched file (hip/attention_wmma.hip.cpp).
- Host kv_cache_format test PASS, roundtrip numbers
  identical to step 1 (K 0.010000 / 48.25 dB, V 0.010000 /
  48.15 dB, ring 0.006234, derived-K 0.015822 / 47.14 dB).
- Probes (labels r2s2b-q8 / r2s2b-f16): all five answers
  byte-identical q8_0 vs same-binary f16, model-load timing
  line excluded (Paris / Au / 366 / COH-1 same text /
  47-12-89), and identical to the step-1 answers.
- Prefill medians (tok/s, 5 runs, spreads ~0.3-0.5 pct):
  pp512 2218.59 vs f16 2193.49 (+1.14 pct), pp4096 2866.38
  vs 2906.20 (-1.37 pct), pp16384 2434.77 vs 2521.03
  (-3.42 pct). Decode spot q8 36.69 / 33.29 / 32.2, in the
  family of the previous runs. NO Phase 5 gate pass: the R5
  raised bar (strictly faster at all three sizes) is not
  met, as expected this step -- the fused loader removes
  the redundant scale reads and narrow gathers without a
  regression, but the remaining pp16384 gap lives elsewhere
  (the LDS bf16 round-trip itself and the still per-chunk
  rotated-K / sliding roads are the named suspects for the
  next rounds).

Next: R2 sliding (d256) -- the same block-load + in-wave
distribution applied to the sliding WMMA path, whose chunk
lanes must be audited first (its thread mapping differs);
then R3 current-chunk bypass; then the R5 gate run.

## Phase 5 REDO R2-sliding -- fused dequant-on-load, sliding d256 (2026-11-17, builder tick 14:29 EST)

Audit (hip/attention_wmma.hip.cpp, sliding instantiation
Launch<256, 2, 4>): D=256, kHeads=2, kQueryBlocks=4 gives
kThreads=512, kWavesPerRow=2, kChunks=512, kStage=1 -- each
thread stages exactly ONE 8-value chunk per tile, with
chunk_key = tid % 16 and chunk_d8 = (tid / 16) * 8. The four
chunks of one 32-value block for one key therefore sit in
lanes tid, tid+16, tid+32, tid+48: TWO waves, not one (the
global path's lanes tid, tid+8, tid+16, tid+24 all live in
one wave). A full-block in-wave distribution like
R2-global's does NOT exist for sliding. What does exist:
inside one wave, lanes l and l+16 hold chunks 2c and 2c+1
of the same block for the SAME key (a wave covers two
consecutive chunk indices for all 16 keys), so a half-block
(16-value) in-wave distribution is available, and the pair
shares one key, hence one slot and one validity flag.

Banked design (closest correct fused variant, per the redo
plan's audit-first rule): `LoadQ8HalfBlockRaw` loads one
half block -- 16 codes as one uint4 from the contiguous
payload plane, the block's fp16 scale once from the scale
plane. In the sliding full fetch, the chunk-2c lane loads
the half block for both K and V and the raw words + scale
bits reach lane l+16 by __shfl (executed unconditionally
for the whole wave); each lane dequantizes exactly its own
chunk with DequantQ8Packed (bit-identical bf16(int8) *
bf16(scale) arithmetic). Per block per cache: two 16-byte
code loads + two scale loads replace four 8-byte gathers +
four scale re-reads. Proven LDS store paths unchanged;
partial tiles keep the exact per-key bounds (the pair's
shared key gives one validity flag; out-of-range pairs
stage zeros exactly as the per-chunk road did). Scope:
D==256 non-global q8 only; the D512 non-global fallback
and all f16 roads are untouched.

Verification (2026-11-17, device; binary result ->
/nix/store/3137nkac...-gufo-0-unstable-4938016; MemAvailable
~124 GiB throughout; Ollama :11434 and
gufo-resident.service untouched):
- nix build clean (detached transient unit, exit 0;
  re-run confirms the same store path from cache);
  clang-format clean on the one touched file
  (hip/attention_wmma.hip.cpp).
- Host kv_cache_format test PASS, roundtrip numbers
  identical to R2-global step 1 (K 0.010000 / 48.25 dB,
  V 0.010000 / 48.15 dB, ring 0.006234, derived-K
  0.015822 / 47.14 dB).
- Probes (labels r2sl-q8 / r2sl-f16): all five answers
  byte-identical q8_0 vs same-binary f16, model-load
  timing line excluded (Paris / Au / 366 / COH-1 same
  text / 47-12-89), and identical to the R2-global
  step-2 answers.
- Prefill medians (tok/s, 5 runs): pp512 2209.99 vs f16
  2101.37 (+5.17 pct), pp4096 2860.55 vs 2807.72
  (+1.88 pct), pp16384 2438.34 vs 2524.45 (-3.41 pct).
  Honest caveats: the f16 pp512 set this run is ragged
  (runs 1336.31 / 2072.60 / 2101.37 / 2132.49 / 2154.70 --
  one deep outlier and a median below the ~2190 band of
  the R2-global runs), so the pp512 margin is overstated
  by an unknown amount; q8 spreads were ~0.7-2.7 pct.
  Decode spot q8 36.63 / 33.29 / 32.20 vs f16 42.04 /
  40.65 / 39.19 (decode is not this phase's gate).
  NO Phase 5 gate pass: pp16384 remains ~3.4 pct behind
  f16, the same gap as after R2-global step 2 -- the
  sliding fusion removes its redundant loads without a
  regression but does not move the 16k gap, which keeps
  pointing at the LDS bf16 round-trip itself and the
  still per-chunk rotated-K road on the global layers.

Next: R3 current-chunk bypass (quantize->dequantize
transformation reproduced in-register; raw unquantized
values FORBIDDEN); then the R5 gate run.

## Phase 5 REDO R3 -- current-chunk round-trip bypass: AUDIT-ONLY (2026-11-17, builder tick 15:13 EST)

Conclusion first: a correct byte-identical R3 bypass CANNOT be
built inside the current kernel architecture without a large
executor rework. Per the redo plan's audit-first rule and this
slice's hard rule (raw unquantized values FORBIDDEN), nothing
was implemented and nothing is faked. This entry is the audit
evidence; no code changed, no build or bench was run, and no
speed or probe claim is made for R3.

Evidence (all line numbers on HEAD da5cded):

1. Two separate kernel launches, no shared registers or LDS.
   Executor::Forward runs QkvPost(post, stream_) at
   hip/executor.cpp:1013 and only then Attention(att, stream_)
   at hip/executor.cpp:1058, per layer per segment. QkvPost is
   its own kernel (QkvPostKernel, hip/kernels.hip.cpp:464,
   launched at hip/kernels.hip.cpp:2156). The redo plan's
   in-register quantize->dequantize reproduction would have to
   happen inside WmmaPrefillAttentionKernel
   (hip/attention_wmma.hip.cpp:240), a different launch: values
   cannot flow from QkvPost's registers into it. In-register
   therefore means the attention kernel RECOMPUTING the whole
   transformation from new inputs (points 2-4).

2. The WMMA prefill kernel has no access to bf16-side (or any
   pre-cache) current-chunk K/V values. AttentionArgs
   (hip/kernels.hpp:139) carries exactly: q (the packed FP32
   queries), k_cache / v_cache (bf16), k_cache_q8 /
   v_cache_q8, out / partials and shape scalars. There is no
   projection pointer, and none of the transformation inputs:
   no k_norm, theta_scale, freq_factors or eps -- those exist
   only in QkvPostArgs (hip/kernels.hpp:39). Every fetch path
   in the WMMA kernel reads solely from the caches: the R2
   fused loaders and DequantQ8Chunk take a.k_cache_q8 /
   a.v_cache_q8, the f16 road takes a.k_cache / a.v_cache
   (hip/attention_wmma.hip.cpp fetch lambda, ~lines 424-640).
   The executor never passes its projection buffers onward:
   q_source / k_source / v_source (hip/executor.cpp:940-942)
   feed only post.q / post.k / post.v (line 979 ff.).

3. The raw projection buffers would not be correct inputs
   anyway. k_ / v_ (scratch, hip/executor.cpp:305-306) hold
   PRE-transformation projections. The attended values are
   created inside QkvPostKernel: K = rope(rms(k) * k_norm)
   (norm_rope at hip/kernels.hip.cpp:569, after load at :565),
   V = rms(v) with no weight and no rope (load at :651,
   norm_rope(v, nullptr, false)), and only then quantized per
   32-value block (amax/127 scale via WaveMax, roundf codes,
   fp16 scale; K store ~:577, V store ~:658, derived-K store
   ~:609). Feeding the attention kernel the raw buffers would
   feed un-normalized, un-rotated values -- a worse violation
   than the forbidden raw-value feed, not a variant of it.

4. Recomputing the transformation inside the WMMA kernel is a
   large rework, not a tile-loader switch:
   - New AttentionArgs fields (FP32 K/V projection pointers
     with the fused qkv_stride / separate_v layouts the
     executor juggles at hip/executor.cpp:940-975, plus
     k_norm, theta_scale, freq_factors, eps) and per-segment
     plumbing in Executor::Forward (segments share the same
     scratch rows at row0 offsets).
   - A duplicate of QkvPost's norm_rope in the attention
     kernel: per-head RMS over the FULL head dim (for global
     d512 the norm couples all 16 q8 blocks of a token -- the
     same coupling R4 records), NEOX rope at each key's own
     position, then per-32-block amax. QkvPost computes amax
     with one wave holding exactly one 32-value block;
     the WMMA chunk mapping (chunk_key / chunk_d8,
     hip/attention_wmma.hip.cpp ~:395-402) distributes a
     block's four 8-value chunks across different lanes and,
     for sliding, different waves (R2-sliding audit), so the
     amax/scale would need cross-lane and cross-wave
     reductions plus new synchronization in the fetch path,
     for both the K and V stores, before any dequantize.
   - Derived-K globals (R4's domain, interleaved with
     R2-global in the same kernel): the current chunk's K
     tile is built from v_next registers plus the rotated-K
     store, i.e. TWO different quantizations (the V row and
     the packed rotated row, different block scales) of one
     normalized projection, both reproduced, plus the
     K-from-V reconstruction. An R3 that covered only sliding
     layers would leave the global layers -- essentially all
     quadratic cost at pp16384 -- on the round trip, so R3
     cannot be banked honestly without R4's construction.
   - Boundary definition: the engine's prefill chunk is
     options_.prefill_chunk (engine.cpp:428), cut further by
     max_rows, image spans and remainder merging, NOT a
     multiple of the kernel's 16-key tile (kKeys,
     hip/attention_wmma.hip.cpp:38; tile loop at :727) in
     general. The current-chunk keys [first_position,
     first_position + rows) are well defined per forward, but
     the boundary tile is a partial tile whose per-key source
     switch would ride on top of all the plumbing above.

Rejected alternative, recorded so it is not retried as R3:
QkvPost writing a bf16 side copy of the current chunk for the
attention kernel to read instead of the q8 cache. That is
still a global-memory write+read round trip -- wider bytes
than the q8 round trip it replaces -- so it is not the plan's
bypass and would predictably lose, not win.

What a real R3 would require (future round, Volmarr's word):
either (a) fuse QkvPost into the prefill attention kernel for
the current chunk's rows (one launch owns projection-post and
attention; the cache write for later forwards still happens),
or (b) the full plumbing + duplicated norm/rope/quantize of
point 4. Both are executor-level redesigns beyond one slice.

Next: R5 gate run on the banked R1 + R2-global + R2-sliding
state (the raised bar: strictly faster than same-binary f16
at pp512 / pp4096 / pp16384, five probes byte-identical,
host test PASS). If the gate fails, R7's loop law starts a
new approach round; this audit is one of its inputs (the
remaining pp16384 gap after R2 was ~-3.4 pct, and R3 as
designed cannot be counted toward closing it).

## Phase 5 REDO R5 -- gate run (2026-11-17, run 15:22-15:41 EST, banked by keeper tick 15:44 EST)

Gate protocol per KV_PHASE5_REDO.md R5: unchanged harness,
median of 5, warmup discarded, same-binary f16 comparison,
raised bar (q8_0 strictly faster than f16 at pp512 / pp4096 /
pp16384). Binary: harness default gufo (labels r5-q8 /
r5-f16, results device-side under campaign/kv-harness/
results/r5-q8 and r5-f16). MemAvailable ~91 GiB at both
matrix starts (the 124 GiB seen at tick start had settled;
well above the 40 GiB floor). Ollama :11434 and
gufo-resident.service untouched.

Prefill medians (tok/s, 5 runs):
- pp512: q8 2207.91 vs f16 2190.91 (+0.78 pct)
- pp4096: q8 2882.42 vs f16 2886.76 (-0.15 pct)
- pp16384: q8 2446.69 vs f16 2528.64 (-3.24 pct)
Spreads: pp16384 tight (q8 2440.83-2452.60, f16
2521.49-2531.62), so the 16k gap is real, not noise.
pp4096 is inside its wider spread and is recorded as
not-faster, not as a proven regression.

Probes: all five byte-identical q8_0 vs same-binary f16
(Model Load timing line excluded) -- FACT-1 Paris,
FACT-2 Au, FACT-3 366, COH-1 identical text, LONG-1
47-12-89. Verified by direct diff this tick.

Host kv_cache_format test: PASS, standalone compile this
tick, roundtrip numbers identical to R2 (K 0.010000 /
48.25 dB, V 0.010000 / 48.15 dB, ring 0.006234,
derived-K 0.015822 / 47.14 dB).

Decode spot (recorded, not this gate): tg512 36.70 vs
42.14, tg4096 33.37 vs 40.56, tg16384 32.27 vs 39.21.

VERDICT: Phase 5 gate FAILS on the raised bar. pp512
passes, pp4096 is not faster, pp16384 remains ~3.2 pct
behind f16 -- the same LDS bf16 round-trip gap R2 left,
with R3 proven unbuildable in the current architecture
(audit above) and R4 not yet attempted as its own
construction. Per R7 loop law, a new approach round
starts from this evidence; candidates on record: R4
derived-K globals loaded once (interleaved with the
R2-global kernel), R6 bounded experiments (E1 fused-V
scale-folding with its decision rule recorded before it
starts; E2 int8-WMMA QK spike, microbenchmark only), or
the executor-level R3 redesign, which needs Volmarr word
as a future round. Phase 6 (KV-13) does NOT start.
No quantized mode becomes any default.

## Phase 5 REDO R7 round 2 -- R4 rotated-K audit (2026-11-17, keeper tick 15:54 EST)

Conclusion first: the one part of R4 not yet fused -- the
rotated-K store of the global layers -- CAN take a fused
dequant-on-load loader, in the half-block form already proven
for sliding in R2-sliding. Nothing was implemented this tick;
this entry is the audit evidence and the implementation spec
for the next tick. No build, bench, speed or probe claim is
made here.

Evidence (all on HEAD 042e0da, hip/attention_wmma.hip.cpp):

1. What R4 already holds. In the global full-tile q8 fetch the
   V store (512 values per head) is fused (R2-global step 2,
   LoadQ8BlockRaw + in-wave distribution), and the V block
   values serve both the K-from-V dims and the V tile from the
   same registers -- nothing is read or dequantized twice on
   that road. The rotated-K chunks are the named exception,
   kept on the per-chunk road because their chunk lanes span
   waves (fetch comment, ~lines 435-439; call at ~line 486:
   k_rot = DequantQ8Chunk(a.k_cache_q8, rot_row_bytes_q8,
   key0 + tid % kKeys, kv_head, 2 * kGlobalPairs,
   (tid / kKeys) * 8)).

2. Rotated-K geometry. kGlobalPairs = 64 (line 194), so the
   rotated row is 128 values per head = 4 q8 blocks of 32.
   kRotChunks = kKeys * 2 * kGlobalPairs / 8 = 16 * 128 / 8
   = 256 chunks per tile; rot_wave = wave < 8. Chunk r = tid
   (tid < 256) is key tid % 16, dim group c = tid / 16 in
   0..15, dims 8c..8c+7 of the rotated row. The store side
   (store_keys, ~lines 637-646) writes k_rot to LDS dims
   c * 8 for c < 8 and D/2 + (c - 8) * 8 for c >= 8.

3. The half-block opening (same shape as R2-sliding). For one
   key and one 32-value block b, the four chunks sit at tids
   key + 16*(4b), +16*(4b+1), +16*(4b+2), +16*(4b+3) -- two
   waves. Inside ONE wave, lanes l and l+16 hold chunks 2c
   and 2c+1 of the same block for the SAME key (a wave covers
   two consecutive dim groups for all 16 keys). So lane l can
   load the half block (16 codes as one uint4 from the
   payload plane, the block fp16 scale once from the scale
   plane, via the existing LoadQ8HalfBlockRaw with
   per_head = 2 * kGlobalPairs) and the raw words + scale
   bits reach lane l+16 by __shfl, executed unconditionally
   for the whole wave; each lane then dequantizes exactly its
   own chunk with DequantQ8Packed -- bit-identical arithmetic,
   same as every fused loader banked so far.

4. Expected saving, bounded honestly. Today each rotated
   block per key costs four 8-byte code gathers + four scale
   re-reads (one per chunk). Fused: two 16-byte code loads +
   two scale loads per block per key (one pair per wave).
   But the rotated row is only 128 of the 640 values fetched
   per head per key (V 512 + rotated 128), so even a perfect
   fusion touches at most about a fifth of the global fetch
   traffic, and the fetch is only part of tile cost. This
   slice alone is therefore NOT expected to close the full
   pp16384 gap (-3.24 pct at the R5 gate); the LDS bf16
   round-trip remains the other named suspect. The slice is
   still worth banking: it removes the last per-chunk road
   in the global path, with a proven pattern and a bit-
   identity proof obligation, and it prices the rotated-K
   share exactly for the round after it.

5. Scope and risks for the implementation tick.
   - Full tiles only: the fused call sits in the same
     key0 + kKeys <= block_hi branch as the V fusion; the
     partial-tile road (~line 602 ff.) keeps the exact
     per-chunk DequantQ8Chunk call and its bounds logic.
   - rot_wave lanes only produce k_rot, but __shfl must run
     unconditionally across each participating wave, as in
     R2-sliding; waves >= 8 do not participate.
   - Half-block alignment: dim groups pair as (0,1), (2,3),
     ... so d8 % 32 is 0 or 16 for the loading lane, which
     is exactly LoadQ8HalfBlockRaw precondition.
   - Proof obligations, same as R2: nix build clean, host
     kv_cache_format PASS with identical roundtrip numbers,
     all five probes byte-identical q8 vs same-binary f16,
     prefill medians at pp512 / pp4096 / pp16384 recorded
     whether or not they improve. No gate pass is claimed
     unless the R5 raised bar is met in a separate gate run.

Next: implement the R4 rotated-K half-block fusion per this
spec, bank it with its evidence, then re-run the R5 gate
protocol. If the gate still fails, the next R7 round takes
on the LDS bf16 round-trip itself.

## Phase 5 REDO R7 round 2 -- R4 rotated-K half-block fusion IMPLEMENTED (2026-11-17, keeper tick 16:04 EST)

Implementation (hip/attention_wmma.hip.cpp, global full-tile
q8 fetch only, per the R7 round-2 audit spec above):
- The rotated-K per-chunk DequantQ8Chunk call in the
  key0 + kKeys <= block_hi branch is replaced by the fused
  half-block loader: lane l (< 16) of each participating
  wave loads the half block via LoadQ8HalfBlockRaw (16 codes
  as one uint4 from the payload plane, the block fp16 scale
  once from the scale plane, per_head = 2 * kGlobalPairs);
  the raw words + scale bits reach lane l + 16 by __shfl,
  executed unconditionally across each participating wave;
  each lane then dequantizes exactly its own chunk with
  DequantQ8Packed (loader pack2(rw[0], rw[1]), partner
  pack2(rw[2], rw[3])) -- bit-identical arithmetic, so the
  staged tile is bit-for-bit the per-chunk road's.
- rot_wave lanes only produce k_rot; waves >= 8 do not
  participate. The partial-tile road keeps the exact
  per-chunk DequantQ8Chunk call and its bounds logic,
  untouched. The stale fetch comment claiming the rotated-K
  chunks span waves (no in-wave distribution) is removed.
  clang-format clean on the one touched file.

Verification (2026-11-17, device; binary result ->
/nix/store/csnxpnmq...-gufo-0-unstable-4938016;
MemAvailable ~124 GiB at build and at both matrix starts;
Ollama :11434 and gufo-resident.service untouched):
- nix build clean (detached, exit 0).
- Host kv_cache_format test PASS (standalone compile),
  roundtrip numbers identical to R2: K 0.010000 / 48.25 dB,
  V 0.010000 / 48.15 dB, ring 0.006234, derived-K 0.015822 /
  47.14 dB.
- Probes (labels r7r4-q8 / r7r4-f16, same binary
  result/bin/gufo, GUFO_KV_CACHE_TYPE per label): all five
  byte-identical q8_0 vs same-binary f16, Model Load timing
  line excluded (Paris / Au / 366 / COH-1 same text /
  LONG-1 47-12-89), verified by direct diff this tick.
- Prefill medians (tok/s, 5 runs, warmup discarded):
  q8 pp512 2221.29 (2212.78-2233.50), pp4096 2852.22
  (2793.37-2880.65), pp16384 2401.79 (2377.53-2441.58);
  f16 pp512 2135.79, pp4096 2597.03, pp16384 2140.55.
- HONEST CAVEAT: this session's f16 set sits far below its
  R5 band (R5 f16 medians 2190.91 / 2886.76 / 2528.64) and
  q8 pp16384 (2401.79) sits below R5's q8 median 2446.69 --
  the machine ran slower overall this session, so the
  apparent q8 margins (+4.0 / +9.8 / +12.2 pct) are
  variance-inflated and are NOT evidence about the raised
  bar. NO gate pass is claimed. Decode spot: q8 tg 36.42 /
  32.98 / 31.90 vs f16 41.85 / 40.41 / 38.72.

Next: re-run the R5 raised-bar gate protocol as a separate
step on this banked state (strictly faster than same-binary
f16 at pp512 / pp4096 / pp16384, five probes byte-identical,
host test PASS). If the gate still fails, the next R7 round
takes on the LDS bf16 round-trip itself. No quantized mode
becomes any default.

## Phase 5 REDO R7 round 2 -- R5 gate re-run on R4 state (2026-11-17, gate run 17:09-17:28 EST, banked by keeper tick 17:34 EST)

Gate protocol per KV_PHASE5_REDO.md R5, on banked HEAD
4209ef2 (R4 rotated-K fusion). Labels r7gate-q8 / r7gate-f16,
same binary result/bin/gufo. MemAvailable ~124 GiB at both
starts. Host kv_cache_format test PASS, roundtrip numbers
identical to R2 (K 0.010000 / 48.25 dB, V 0.010000 / 48.15 dB,
ring 0.006234, derived-K 0.015822 / 47.14 dB).

Prefill medians (tok/s, 5 runs, warmup discarded):
- pp512: q8 2178.59 vs f16 2208.61 (-1.36 pct)
- pp4096: q8 2692.32 vs f16 2896.93 (-7.06 pct)
- pp16384: q8 2421.44 vs f16 2537.17 (-4.56 pct)
Spreads: f16 sets tight; q8 pp16384 2402.98-2441.70 vs f16
2534.59-2540.34, so the 16k gap is real, not noise.

Probes: all five byte-identical q8_0 vs same-binary f16
(Model Load timing line excluded), verified by direct diff
this tick -- FACT-1 Paris, FACT-2 Au, FACT-3 366, COH-1
identical text, LONG-1 47-12-89.

Decode spot (recorded, not this gate): q8 tg 36.73 / 33.33 /
32.32 vs f16 42.13 / 40.62 / 39.24.

VERDICT: Phase 5 gate FAILS on the raised bar again -- q8 is
slower than same-binary f16 at all three prefill lengths on
this run. The R4 fusion did not close the gap; the LDS bf16
round-trip remains the named suspect, as the R4 audit
predicted (rotated-K is at most about a fifth of global fetch
traffic). Per R7 loop law, the next round takes on the LDS
bf16 round-trip itself. Phase 6 (KV-13) does NOT start. No
quantized mode becomes any default.

## Phase 5 REDO R7 round 3 -- LDS bf16 round-trip: AUDIT-ONLY (2026-11-17)

AUDIT-ONLY. No code change, no build, no bench, no gate
claim this tick. Evidence verified on device this tick in
hip/attention_wmma.hip.cpp (988 lines).

### What the LDS round-trip is

- LDS arrays (lines 255-263): k_lds[kKeys * kKStride] and
  vt_lds[D * kVtStride] as __half, plus s_lds (260),
  p_lds (261), row_scale / row_sum (262-263).
  Constants: kKeys = 16 (line 38), kSlice = 128 (39),
  kSliceSteps = kSlice / 16 (40), kKStride = D + 8 (251),
  kVtStride = kKeys + 8 (252), kWavesPerRow = D / kSlice
  (247).
- Sizes: global D = 512: k_lds 16 * 520 * 2 = 16,640 B,
  vt_lds 512 * 24 * 2 = 24,576 B (~41 KiB K+V alone).
  Sliding D = 256: k_lds 16 * 264 * 2 = 8,448 B,
  vt_lds 256 * 24 * 2 = 12,288 B (~20.7 KiB K+V).
- Round-trip shape: the R2 fused loaders dequantize q8
  into registers (k_next / v_next uint4, lines 366-367;
  fetch lambda at 421, dequant paths through ~640);
  store_keys (660) and store_values (685, includes the V
  transpose via DPP/perm) write bf16 into k_lds / vt_lds;
  the QK loop (~775-777) and PV loop (~867-876) LoadFrag
  from LDS (LoadFrag at 162) and Wmma (42). There is NO
  separate global bf16 scratch/staging pass -- the
  round-trip is register -> LDS bf16 -> register
  fragment, per 16-key tile, with 3 __syncthreads per
  tile in the main loop (784, 839, 881, plus the initial
  760), and the next tile is already prefetched into
  registers while the current tile computes (fetch at
  loop top 768, store_keys after scores at 837,
  store_values after PV at 883).

### Why LDS exists here

- The tile is shared by the whole GQA group / all waves
  in the workgroup (kWavesPerRow = D / 128: 4 global,
  2 sliding). V must be transposed for PV WMMA, and WMMA
  fragments are lane-layout specific (Wmma at 42,
  LoadFrag at 162). Each wave reads the same K tile for
  its own 128-dim slice, so dequantizing once into LDS
  is shared; keeping K/V in registers per-wave would
  duplicate dequant per wave.

### Candidate mechanisms (honest assessment)

- (A) int8-resident LDS tile + dequant-on-LoadFrag --
  halves K+V LDS, but repeats dequant per wave/slice
  read, needs manual fragment construction + per-32
  scale handling in fragment layout, large kernel
  rework, gain uncertain (gfx1151 int8 WMMA = bf16 rate
  per redo plan; win only on LDS traffic/capacity).
- (B) Direct register fragments via cross-lane shuffles
  bypassing LDS -- fetch distribution (chunks across
  lanes/waves, see R2/R3 audits) does not match WMMA
  fragment lane ownership; requires cross-lane/cross-wave
  redistribution at least as costly as LDS, similar class
  to R3's proven-large executor/kernel rework.
- (C) Narrow wins: padding/stride trimming (kKStride +8,
  kVtStride +8 exist for bank-conflict avoidance --
  removing risks conflicts, saves little),
  double-buffering (already effectively prefetched in
  registers; AMD guide note in redo plan says measure
  single vs double, do not assume). None is a small slice.

### Conclusion

No small, bit-identical LDS-elimination slice exists in
the current architecture. The next implementation round
must choose ONE of: (A) int8-resident K tile only
(bounded: K is read kSliceSteps times per wave for QK,
V transpose path untouched -- spec it as the recommended
bounded experiment, with decision rule: bank only if
probes byte-identical + host PASS, and claim gate only
via a separate R5 run) or stop and take R6 experiments /
ask Volmarr about executor-level redesign. Do NOT
implement anything this tick. Next step after this audit:
implement (A) as a bounded experiment in a following
tick, or R6 if (A) audit-on-implementation fails.

### Tick state

MemAvailable was ~124,413,940 kB at keeper tick start
(re-checked ~124,409,608 kB this tick), no live
build/bench, Ollama :11434 and gufo-resident untouched,
Phase 6 / KV-13 does NOT start, no quantized default.

## Phase 5 REDO R7 round 3 -- (A) int8-resident K tile IMPLEMENTED (2026-11-17, builder tick 17:59 EST)

Implementation (hip/attention_wmma.hip.cpp only), per the
round-3 audit's bounded experiment (A), all q8 instantiations
(global derived-K, sliding d256, generic plain):
- K stays int8 in LDS: the tile is staged as raw q8 codes
  (1 B/dim, stride D + 16) plus one fp16 scale per
  (key, 32-block), overlaid on the existing bf16 k_lds
  allocation (int8 + scales are under half its bytes, so no
  added LDS). The QK loop dequantizes each 16-dim fragment
  chunk on load with DequantQ8Packed -- each chunk lies
  inside one 32-block, so one scale serves the whole load,
  and each K element is dequantized exactly once (waves
  partition the head dim). V / vt_lds path untouched; the
  f16 road is untouched.
- Raw forms ride through fetch: LoadQ8ChunkRaw (raw twin of
  DequantQ8Chunk) for the generic and global partial-tile
  paths; the sliding half-block loader's existing raw words;
  the global full-tile V block words and the R7-R4 rotated-K
  half-block words (the derived K's non-rotated dims ARE the
  V store's codes, the rotated dims the rotated store's --
  both staged raw with their own block scales, so the
  derived tile is reproduced exactly). Partial tiles keep
  the exact per-chunk bounds logic; invalid chunks stage
  zero codes.
- Bit-identity argument: staged bf16 K was exactly
  code * block-scale rounded to binary16 (DequantQ8Packed);
  the int8 road stores the same codes + scale and applies
  the same DequantQ8Packed at fragment load. No arithmetic,
  accumulation order, or WMMA operand changes.

Verification (2026-11-17, device; binary result ->
/nix/store/hq5zgb1nq3k13gqbnrl5aw39a48r18kd-gufo-0-unstable-4938016;
MemAvailable ~124 GiB at build, probes and both matrix
starts; Ollama :11434 and gufo-resident.service untouched):
- nix build clean (detached, exit 0); clang-format clean on
  the one touched file.
- Host kv_cache_format test PASS, roundtrip numbers
  identical to R2 (K 0.010000 / 48.25 dB, V 0.010000 /
  48.15 dB, ring 0.006234, derived-K 0.015822 / 47.14 dB).
- Probes (labels r7a-q8 / r7a-f16, same binary
  result/bin/gufo, GUFO_KV_CACHE_TYPE per label): all five
  byte-identical q8_0 vs same-binary f16, Model Load timing
  line excluded (FACT-1 Paris, FACT-2 Au, FACT-3 366,
  COH-1 same text, LONG-1 47-12-89), verified by direct
  diff this tick. Banking rule (probes identical + host
  PASS) is met.
- Prefill medians (tok/s, 5 runs, warmup discarded):
  q8 pp512 2195.67 (2178.98-2206.63), pp4096 2654.13
  (2600.91-2709.10), pp16384 2071.58 (2031.48-2099.67);
  f16 pp512 2206.22, pp4096 2906.29, pp16384 2514.42.
  Deltas: pp512 -0.48 pct, pp4096 -8.68 pct,
  pp16384 -17.61 pct. Decode spot: q8 tg 36.59 / 33.24 /
  32.24 vs f16 42.10 / 40.52 / 39.15.
- HONEST VERDICT: (A) as implemented is SLOWER than the
  staged-bf16 road it replaced, and slower than the R4
  state at pp4096 / pp16384 (gate-run q8 2692.32 /
  2421.44). Halving K LDS traffic did not pay for moving
  dequant into the QK fragment path (per-fragment scale
  LDS load + dequant latency in front of every WMMA, on
  the critical path). The numbers are recorded as
  measured; NO Phase 5 gate pass is claimed and no R5
  gate run was attempted on this state.

Next: a separate R5-protocol gate run is NOT the obvious
next step on a state already slower at spot lengths; the
next R7 round should decide from this evidence -- e.g.
revert (A) to the R4 staged road and take R6 experiments,
or attack the fragment-path latency directly -- recorded
before it starts. Phase 6 (KV-13) does NOT start. No
quantized mode becomes any default.

## Phase 5 REDO R7 round 4 -- DECISION RECORD (2026-11-17, keeper tick 18:32 EST)

Decision, recorded BEFORE the next implementation round
starts, per R6 / R7 loop law:

- REVERT experiment (A) (int8-resident K tile, banked
  3d4aa11) to the R4 staged-bf16 road (state 4209ef2 /
  gate-run a2be8fe) as the next implementation slice.
  Reason: (A) measured slower than the staged road it
  replaced at pp4096 (2654.13 vs gate-run 2692.32) and
  pp16384 (2071.58 vs 2421.44), and slower than same-
  binary f16 at all three lengths (-0.48 / -8.68 /
  -17.61 pct). Halving K LDS traffic did not pay for
  dequant latency on the QK fragment critical path.
  The (A) commit stays in history as evidence; the
  revert restores the best-known q8 prefill state.
- After the revert is banked with probes byte-identical
  + host PASS (no gate claim on the revert itself), the
  following round takes R6 bounded experiments, each
  with its decision rule recorded before it starts:
  E1 fused-V scale-folding only in the variant that
  reproduces staged bf16 rounding of dequantized V;
  E2 int8-WMMA QK spike is microbenchmark-only.
- No R5 gate run on the (A) state. Phase 6 (KV-13)
  does NOT start. No quantized mode becomes any default.

Tick state: banked head verified 3d4aa11 == fork
ls-remote, no prior lock, no live build/bench,
MemAvailable ~124 GiB, Ollama :11434 and
gufo-resident.service untouched.
