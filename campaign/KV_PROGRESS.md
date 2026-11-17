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
  core CMake preset build — exact preset/flags to be pinned down
  in Phase 1 when the first test lands (KV-03).
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
| KV-02 f16 baseline | IN PROGRESS | harness run label: baseline-f16 |
| KV-03 format field + allocation | PENDING | |
| KV-04 GUFO_KV_CACHE_TYPE plumbing | PENDING | |
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

TO BE FILLED by the harness run. Table: pp/tg at 512 / 4k / 16k
(median of 5), KV bytes/token, probe answers + greedy reference
paths under campaign/kv-harness/results/baseline-f16/.

## Phase gates

- Phase 0 gate: PENDING — baseline table below + harness committed.
- Phase 1 gate: PENDING.
