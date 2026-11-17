# Phase 5 REDO — fused dequant-on-load prefill (supersedes the KV-12 staging design)

Commissioned by Volmarr, 2026-11-17 12:20 EST: Phase 5 is redone
in full on a researched new approach. Bar, raised by his order:
q8_0 prefill must be **faster than the same-binary f16 baseline**
at pp512, pp4096 and pp16384 — not merely within noise — with
all five probes byte-identical and the kv_cache_format host test
green. His loop law (12:25 EST): a failed gate starts another
approach round; the phase is never surrendered and a gate is
never quietly passed.

Design basis: the deep-research report of 2026-11-17
(`q8-kv-prefill-faster-than-bf16`, VM research notes) and Yrsa's
own ChatGPT consultation the same day. Both converged on the
same first move. The hardware fact that shapes everything: on
gfx1151 int8 WMMA runs at the SAME rate as bf16 WMMA (512
ops/clk/CU), so q8_0 can only win on bytes moved, LDS capacity,
and work eliminated — never on matrix-core throughput.

## What is being replaced

KV-12 (banked @81190ea, gate OPEN, kept as history): the WMMA
prefill kernel in `hip/attention_wmma.hip.cpp`
(`LaunchWmmaPrefillAttention` / `Launch(a, stream, q8)`)
dequantizes each K/V tile via `DequantQ8Chunk` into an LDS
bf16 tile — the same tile the f16 road stages — and
`hip/kernels.hip.cpp` `LaunchAttention` tries that road first,
falling back to the KV-11 split-kernel road. Measured:
pp512 +0.73%, pp4096 −1.18%, pp16384 −2.98% vs same-binary f16.
The staging traffic and its LDS footprint are the named cost.

## R1 — Layout precondition (audit first, change only if needed)

Audit the physical KV layout written by the quantize-on-write
path (`hip/executor.cpp` QkvPost; `hip/executor.hpp`
`struct KvCache`, row strides `row_bytes_q8` /
`rot_row_bytes_q8` in the WMMA kernel). If rows are the
interleaved ggml 34 B block layout (2 B scale + 32 B data),
tile loads cannot coalesce cleanly. The redo layout: int8
payload contiguous per (head, token-tile, d), fp16 scales in a
separate plane indexed [token][head][d_block], tiles stored in
consumption order. Pure layout — zero arithmetic change — but
writer, both attention paths and decode must agree; keep a
versioned layout flag. If the audit shows reads already
coalesce, record that and skip R1.

## R2 — Fused dequant-on-load (the core change; replaces staging)

- The tile loader reads q8_0 from global and dequantizes
  **once, into the LDS tile, shared by the whole GQA group**
  the workgroup covers — never per query head (the per-head
  redundant dequant is the known llama.cpp HIP failure mode).
- No global bf16 scratch anywhere; no separate staging pass.
- bf16 WMMA math, the dequant arithmetic
  (`bf16(int8) * bf16(scale)`, same intermediate precision as
  `DequantQ8Chunk` today) and the accumulation order are
  UNCHANGED — the design is bit-identical capable, which the
  probe gate requires.
- rocWMMA `load_matrix_sync` cannot dequantize: fragments are
  constructed manually from the LDS tile (the AMD GEMM guide's
  own pattern), or the LDS tile is written bf16 and loaded
  typed. Budget the manual fragment construction as the main
  implementation work.
- Order: **global d512 layers first** (they carry essentially
  all quadratic cost at 16k), then sliding d256.
- gfx1151 LDS is 64 KB/workgroup. An int8-resident tile halves
  footprint vs the bf16 staged tile; spend the freed budget on
  tile geometry (Bc) only with a byte-identity re-proof.
  **Measure single- vs double-buffered LDS; do not assume**
  (AMD's gfx1151 GEMM measurements found single-buffer +
  interleaved prefetch FASTER: +15% / +38% on the analogous
  GEMM). Interleave next-tile global loads between WMMA
  instructions per that guide.

## R3 — Current-chunk round-trip bypass

In prefill, the chunk being processed is consumed from the
projection path WITHOUT the global cache write+read round
trip: in the FlashAttention loop the tile loader switches
source at the boundary tile (chunk size aligned to Bc) —
bf16-side values for the current chunk, q8_0 fused loader
(R2) below it. One kernel, one online-softmax state, no
merge pass. **Hard rule:** the attended values must be
IDENTICAL to the staged path, so the quantize→dequantize
value transformation is reproduced in-register for the
current chunk. Feeding raw unquantized projection values is
FORBIDDEN — it changes the arithmetic and voids the probe
gate. (For a fresh prompt this bypass covers 100% of tokens;
in chunked/extend mode its share is C/L of tile visits —
necessary hygiene either way, and proportionally large for
sliding layers whose window is half current chunk.)

## R4 — Derived-K global layers (d512, k_eq_v)

Per Phase 4's design a global layer stores two rows per
token: the V row (2 heads × 512; the RMS-normalized K
projection) and the rotated-K row (2 heads × 128 packed
values). The fused kernel loads and dequantizes **both
stores once per tile into LDS** (the full d=512 row per
token must be resident before K derivation — the norm
couples all 16 q8 blocks of a token), builds K fragments
from the derived combination exactly as the f16 road does
(query carries k_norm; rope already applied pre-quantization
in both stores), and builds V fragments from the same LDS
residency. No tensor is read or dequantized twice. Do NOT
attempt int8 WMMA here: the derivation is nonlinear in the
stored values, so an integer dot product on the raw store
computes the wrong quantity.

## R5 — Gate protocol (unchanged harness, raised bar)

Phase 0 harness (`campaign/kv-harness/`), median of 5, warmup
discarded, same-binary f16 comparison, MemAvailable ≥ ~40 GiB,
machine otherwise idle. PASS requires ALL of:
- pp512, pp4096, pp16384: q8_0 strictly faster than f16
  beyond the measured noise band (intra-run spread < 0.4%).
- All 5 probes byte-identical q8_0 vs f16 (FACT-1/2/3, COH-1,
  LONG-1), timing/report lines excluded as before.
- kv_cache_format host test PASS.
- Decode spot-check recorded (not this phase's gate, never
  hidden).

## R6 — Bounded fallback experiments (only if R1–R4 land short)

Each runs only with its decision rule recorded in
KV_PROGRESS.md before it starts; NOTHING from R6 ships past
the probe gate.

- **E1 — Fused-V scale-folding:** fold V block scales into P
  with fp32 accumulation. Only the variant that reproduces
  the staged path's bf16 rounding of dequantized V may pass;
  a variant that changes V's rounding is disqualified by the
  probe gate, not debated.
- **E2 — int8-WMMA QK spike (microbenchmark only):** one
  16×64 score tile at d512 and d256, variants A (bf16
  reference) / B (pre-quantized Q) / C (runtime-quantized Q),
  per-32 partial i32 accumulators flushed into an fp32
  master with the separable scale sQ·sK. Decision rule: if B
  cannot beat the fused R2 path, int8 QK is abandoned
  permanently — the research says there is no int8 rate
  advantage on gfx1151; this spike settles it on silicon
  instead of by citation. (Known bar risk, recorded in
  advance: int8 QK quantizes Q — error the f16 build never
  had — so even a fast spike is expected to fail byte
  identity; it is a measurement, not a shipping candidate.)

## R7 — Loop law

A failed gate starts another approach round (new research
and/or a new mechanism), recorded in KV_PROGRESS.md with the
numbers that failed. The KV-12 staging commit stays as
history. f16 remains the permanent oracle and rollback.
Phase 6 (KV-13) does not start until the Phase 5 gate call
is made on passing evidence.

## Slice tracking

Redo work is tracked under the existing KV-11 / KV-12 slice
ids in KV_PROGRESS.md: R1 audit → R2 global → R2 sliding →
R3 → R4 (interleaved with R2-global as the same kernel) →
R5 gate run. Each slice banks separately with its evidence,
per-slice, ls-remote verified, as everywhere in this campaign.
