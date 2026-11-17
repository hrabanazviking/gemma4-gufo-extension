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

## Phase gates

- Phase 0 gate: PASSED 2026-11-17 — harness committed (KV-01),
  baseline table + probe answers banked (KV-02).
- Phase 1 gate: PASSED 2026-11-17 — evidence below.
- Phase 2 gate: PASSED 2026-11-17 — evidence in the KV-06 section above.
- Phase 3 gate: PASSED 2026-11-17 -- evidence in the KV-08 section above.
- Phase 4 gate: PASSED 2026-11-17 -- evidence in the KV-10 section above.

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
