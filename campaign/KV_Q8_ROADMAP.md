# KV-Cache Quantization Roadmap — q8_0

**Line:** hrabanazviking/gemma4-gufo-extension, branch `yrsa/our-gemma4`
**Upstream base at founding:** franzmoca/gemma4-gufo-extension @ a31ce42
**Charter:** `RUNEFORGEAI_MAD_DATA_SCIENTIST_GEMMA4_EXTENSION.md` (house
document, never donated). This roadmap is a house document too.
**Commissioned:** 2026-11-17 by Volmarr — roadmap first, then build,
guarded by a 10-minute keeper minder.

## The prize

The Gemma4 KV cache is full-precision bf16 end to end: roughly
2.3 GiB of cache at native context for the 26B model, and every
generated token re-reads the whole cache. Storing that cache in
q8_0 — per-block quantized bytes with a scale per block — roughly
halves the bytes. Generation is bound by how fast the cache can be
read, so a halved cache should mean **faster long conversations and
half the cache memory**, at a quality cost too small to see. This
roadmap lands that, production-ready, or proves honestly where it
cannot be landed.

## Governing laws (from the charter and Volmarr's orders)

- **Speed law.** No quantized mode ships as a default unless
  benchmarks prove no decode regression beyond run-to-run noise at
  every tested context length. A mode that cannot meet the law stays
  opt-in via the setting. Numbers come from the benchmark harness,
  never from hope.
- **No limits except by setting.** Cache behavior is governed by
  `GUFO_KV_CACHE_TYPE` (f16 | q8_0 | q4_0) the Ollama way. Invalid
  values are a clear named error, never a silent fallback.
- **Honesty.** Gaps and regressions are recorded as they are. A
  phase that fails its gate is reported failed, with evidence.
- **Machine care.** All building and benchmarking happens on
  purpleraider, serialized, fenced, MemAvailable-checked. The
  resident service and the machine's own Ollama are never disturbed
  outside the integration gates (Phases 7–8).
- **Banking.** Every slice banks to `yrsa/our-gemma4` with an
  explicit-refspec push to the `fork` remote and an ls-remote
  verification. README.md files are never overwritten.

## Design decisions (settled before building)

- **Layout.** ggml-style q8_0 blocks: 32 consecutive values per
  block, stored as 32 int8 values + one fp16 scale = 34 bytes per
  32 values. Sliding layer K or V row (2,048 values) goes from
  4,096 B (f16) to 2,176 B (q8_0); K+V per sliding layer per token
  from 8,192 B to 4,352 B.
- **Format is a property of the cache**, carried in `KvCache` and
  honored by every consumer — allocation, write path, all attention
  paths, MoveKey, snapshots. No consumer may assume bf16.
- **f16 remains a first-class format forever.** Quantization is
  opt-in until Phase 8's default decision, and f16 is the rollback
  at every step.
- **q4_0 is out of scope for this roadmap.** The format field
  reserves its value; the machinery built here is the road it will
  later travel.

## Benchmark & correctness protocol (used by every gate)

- **Harness:** fixed bench driver committed in Phase 0; same
  prompts, warmup run discarded, median of 5 runs, machine
  otherwise idle, one bench at a time.
- **Matrix:** decode tok/s and prefill tok/s at ~512, ~4k, ~16k
  context on the default model (gemma-4-26B-A4B-it-heretic Q8_0
  from the SSD), plus KV bytes/token per layer type.
- **Correctness:** a frozen probe set (factual recall, coherence,
  one long-context recall probe) plus a logit comparison against
  the f16 run of the same build: quantized answers must stay
  correct and coherent; drift is measured and recorded, not
  waved away.

## Phases and slices

### Phase 0 — Baseline and harness
- **KV-01** Benchmark + probe harness committed (bench driver,
  probe set, results format).
- **KV-02** f16 baseline banked: full matrix measured on the
  current build; probe answers and reference logits recorded.
- **Gate:** baseline table in `campaign/KV_PROGRESS.md`,
  reproducible from the committed harness.

### Phase 1 — Format and layout
- **KV-03** `KvCache` gains a format field; `CacheBytes()` and
  `CreateCache()` allocate per format (q8_0 block layout);
  format plumbing from executor creation to the cache.
- **KV-04** `GUFO_KV_CACHE_TYPE` setting reaches the executor
  (extension env/flag road, Ollama-mannered); invalid value is a
  named error; `diagnose`-style reporting shows the active format.
- **Gate:** builds clean; f16 behavior byte-for-byte the old
  behavior; q8_0 allocates the computed sizes (asserted in tests).

### Phase 2 — Quantize-on-write
- **KV-05** QkvPost quantizes K rows on store for sliding layers
  (linear layout), per-block scales.
- **KV-06** Same for V rows and the ring-buffer layout.
- **Gate:** roundtrip tests — dequantized rows vs bf16 reference
  within recorded error bounds (max abs error + SNR in the
  progress file); no write path still writes raw bf16 in q8_0 mode.

### Phase 3 — Dequant-in-decode
- **KV-07** Decode attention (incl. split-K) reads q8_0 with
  in-kernel dequantization for sliding layers.
- **KV-08** Decode correctness + speed gate: probe set and logit
  comparison vs the Phase 0 reference; decode matrix vs baseline.
- **Gate:** probes pass; **no decode regression beyond noise at
  512 / 4k / 16k**; long-context improvement measured and recorded
  either way.

### Phase 4 — Global layers: the derived-K design
- **KV-09** Design + implement quantization of the global-layer
  stores: the V-store (which is the normalized K projection) and
  the rotated-K dims store, through the on-the-fly key
  reconstruction rope(k_norm . V).
- **KV-10** Validation gate: reconstruction error measured through
  the real attention path; probes + matrix re-run.
- **Gate:** global layers quantized within tolerance — **or** the
  honest fallback ships: global layers stay f16, sliding layers
  quantized (a mixed cache), with the reason and the numbers
  recorded. A failed design is a result, not a disgrace.

### Phase 5 — Prefill paths
- **KV-11** Non-WMMA prefill reads/writes q8_0 (dequant where the
  kernel consumes cache tiles).
- **KV-12** WMMA matrix-core prefill: dequant-tile stage feeding
  the matrix cores, **or** a measured dispatch rule (q8_0 prefill
  takes the faster path per context size, crossover measured).
- **Gate:** prefill matrix vs the matrix-core f16 baseline — no
  regression beyond noise; the dispatch rule, if any, is in code
  and in the progress file.

### Phase 6 — Machinery re-proof
- **KV-13** MoveKey (speculative decoding), continuation
  snapshots/rollback, and the MTP draft chain re-proven over the
  quantized layout; the extension's existing tests/smoke pass.
- **Gate:** all green, or failures named with owners and fixes
  scheduled — never silently skipped.

### Phase 7 — Integration
- **KV-14** Full stack rebuilt (extension + Gufo core), installed
  on purpleraider; the resident supervisor passes
  `GUFO_KV_CACHE_TYPE` through to its engine child and reports the
  active format on its meta surface.
- **KV-15** Resident service end-to-end re-proof with q8_0 active:
  resting state, first-call load, keep_alive rules, idle unload,
  malformed-request manners, thermal throttle untouched.
- **Gate:** service behavior indistinguishable from before, except
  the cache is quantized and the numbers say so.

### Phase 8 — Soak, decide, close
- **KV-16** Long-context soak (16k+ conversation through the
  resident service), final benchmark table (baseline vs q8_0, all
  lengths, KV bytes/token), **default decision**: q8_0 becomes the
  service default if it ties or wins everywhere the law requires;
  otherwise f16 stays default and q8_0 ships as a proven opt-in.
  Closure record in `campaign/STATE.md`; completion report to
  Volmarr.
- **Gate:** the table, the decision, and the evidence agree.

## Risks named in advance

- **WMMA prefill** is the hardest speed fight: matrix cores eat
  bf16 tiles directly; a dequant stage may cost more than the
  cache saving at short contexts. The Phase 5 dispatch rule is the
  planned answer, not an improvisation.
- **Derived-K error** may exceed tolerance on global layers;
  Phase 4's mixed-cache fallback keeps the campaign shippable.
- **Snapshot incompatibility**: quantized rows change byte sizes;
  Phase 6 exists because this will bite if ignored.
- **Build time** on the ROCm stack is long; slices are sized so a
  keeper tick can always make progress between builds.

## Definition of done

q8_0 KV cache available via `GUFO_KV_CACHE_TYPE`, proven correct
on probes, proven no-slower (and faster where the cache binds) on
the full matrix, integrated into the resident service with its
default decided by the numbers, all banked and verified on
`yrsa/our-gemma4`, with the closure record written. Anything less
is reported as exactly what it is.
