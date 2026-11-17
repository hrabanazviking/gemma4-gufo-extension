# Campaign state — RuneForgeAI Gemma4 extension line

House record for hrabanazviking/gemma4-gufo-extension, companion to
the Gufo core line (hrabanazviking/gufo-RuneForgeAI-Hacks-and-Experiments,
branch `yrsa/our-gufo`). House-only file: never donated (see the
charter's rule).

## Founded

- 2026-11-17: Volmarr forked franzmoca/gemma4-gufo-extension to
  hrabanazviking/gemma4-gufo-extension and seated this line under the
  same rules as the core line. Charter:
  `RUNEFORGEAI_MAD_DATA_SCIENTIST_GEMMA4_EXTENSION.md` at repo root.
- Upstream base at founding: a31ce42 ("fix(gemma4): honor reasoning
  budgets past the thought header").

## Branch map

- `main` — mirrors upstream; not our working line.
- `yrsa/our-gemma4` — THE running line: what we build, install, and
  live with. Our features win conflicts here.
- `donate/<feature>` — created only when a mature improvement is
  chosen for donation, based on upstream's current main, reshaped to
  upstream standards, house files absent (verified before push).

## Remotes (dev clone: purpleraider `~/src/gemma4-gufo-extension`)

- `origin` — franzmoca/gemma4-gufo-extension (upstream).
- `fork` — hrabanazviking/gemma4-gufo-extension (ours). Push the
  running line with an explicit refspec:
  `git push fork HEAD:refs/heads/yrsa/our-gemma4`, then verify with
  `git ls-remote fork`.

## House-only files (never donated)

- `RUNEFORGEAI_MAD_DATA_SCIENTIST_GEMMA4_EXTENSION.md`
- `campaign/STATE.md` (this file)

## Upstream sync road

Fetch `origin`, merge into `yrsa/our-gemma4`, resolve conflicts in
favor of our features, re-run the benchmark matrix, record the merge
here.

## Founding commission — KV-cache quantization (ready at Volmarr's word)

Recon on the core line (2026-11-17) established where the cache lives
in this extension and what the work is:

- `hip/executor.hpp` `struct KvCache`: one device allocation, bf16
  (uint16_t) rows, token-major; `CacheBytes()`/`CreateCache()` in
  `hip/executor.cpp`. Model shape: 30 layers; 25 sliding layers
  (8 KV heads x 256, ring buffer); every 6th layer global
  (2 KV heads x 512) with NO separate V projection — the V cache is
  the normalized K projection and keys are synthesized on the fly as
  rope(k_norm . V).
- Scope: (a) format field + block layout in KvCache/CacheBytes,
  (b) quantize-on-write in QkvPost (sliding layers first),
  (c) dequant-in-decode in the attention kernels, (d) a dequant-tile
  stage for the WMMA matrix-core prefill, (e) a designed answer for
  the derived-K global layers, (f) MoveKey + continuation-snapshot
  re-proof, (g) the full 512 / 4k / 16k benchmark matrix against the
  bf16 baseline (decode ~41-43 tok/s warm, pp2048 ~2981.8 t/s).
- Speed law: no decode regression beyond noise at any tested length;
  a mode that cannot meet it stays opt-in, never the silent default.
- Setting: `GUFO_KV_CACHE_TYPE` (f16 | q8_0 | q4_0), the Ollama way.
- Status: house founded; the campaign itself starts at Volmarr's word.

## Roadmap and minder (2026-11-17)

- Volmarr commissioned the build the same day the house was
  founded: full roadmap at `campaign/KV_Q8_ROADMAP.md` (Phases
  0-8, slices KV-01..KV-16, gates per phase).
- Progress ledger: `campaign/KV_PROGRESS.md` on this branch —
  the minder and every builder keep it current in the same
  commits that bank slices.
- A 10-minute keeper minder (job `gufo-kv-q8-keeper`) guards the
  campaign under Volmarr's keeper law: it checks the work is
  flowing, restarts stopped slices, and banks in roadmap order.
  It reports to Volmarr only at phase gates, at completion, or
  on a genuine blocker.
- Build lock: a builder holds `campaign/.kv-build.lock` (device
  side, uncommitted) while working; the minder observes only
  while a fresh lock and a live build/bench process exist.

## Relation to the core line

The resident service (`gufo-resident.service` on purpleraider, core
line) consumes this extension through the build flake in the dev
clone's parent tree. When this line changes, the core line's build
must be re-proven end to end (resting state, first-call load,
unload rules) before the new build is installed.
