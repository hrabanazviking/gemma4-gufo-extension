# RuneForgeAI Mad Data Scientist Personal Gemma4 Extension Hacks

This is **Volmarr's personal RuneForgeAI hack line of the Gemma4
extension for Gufo**. It is not the official extension, it is not a
community edition, and it is not built to please anyone upstream. It
is built the way a mad data scientist builds the machine on his own
desk: to his own standards, for his own needs, on his own hardware —
and it answers to RuneForgeAI alone.

The official Gemma4 extension lives at franzmoca/gemma4-gufo-extension,
written by Francesco Moca, co-author of Gufo. This fork,
hrabanazviking/gemma4-gufo-extension, is where that extension gets
taken apart, improved, and rebuilt into the Gemma engine our Gufo
actually wants to live with. What survives real use here may, one day
and only when it is mature, be offered back upstream as a gift. Until
then, this line is sovereign.

This line is the companion of our Gufo core line
(hrabanazviking/gufo-RuneForgeAI-Hacks-and-Experiments) and lives under
the same rules, the same standards, and the same charter spirit.

## No support. No warranty. Read this first.

Plain words, so nobody can miss them:

- **RuneForgeAI offers NO support, NO help, and NO warranty of any
  kind for this personal version of the Gemma4 extension.** Not a
  little support. Not "best effort." None. There is no help desk, no
  issue queue with our name on it, no promise that anything here works
  on your machine.
- **You are better off using the official upstream extension.**
  Seriously. It lives at franzmoca/gemma4-gufo-extension, it is the
  real project, and it is the version a sensible person runs. This
  fork is one man's personal workbench, kept in the open because open
  is how we work — not because it is a product offered to the world.
- **It is open source, so you may use the code as open source
  allows** — entirely at your own risk. If it sings on your hardware,
  good. If it breaks, the pieces are yours to keep.
- **By using it, you automatically agree** that you alone are
  responsible for the outcome, and that you have — or provide — your
  own tech support. Running this hack means signing that agreement
  with your actions; there is no other signature and no exception.
- **Do not expect the official extension or Gufo projects to support
  you either.** If you run this personal hack line and something goes
  wrong, it is unlikely the official projects will support you — and
  it would be unfair to ask them to clean up after a version they did
  not make. Take your questions about the official extension to the
  official extension, running official code.

Friendly, and firm: this is a personal mad data scientist's bench.
Visitors may look, and may take the code as the license allows — but
they bring their own tools, their own skills, and their own risk.

## Our standards

- **Needs first.** Every change exists because our work needs it. If a
  feature does not serve the machines and the projects in this house,
  it does not get built, however fashionable it is elsewhere.
- **Our line, our rules.** The running line is never shaped to please
  upstream reviewers. It is the proving ground: improvements earn their
  place by working, under real load, on real hardware.
- **Plain and honest.** Behavior is documented as it truly is. Gaps are
  recorded as gaps, never dressed up as features.
- **Forgiving machinery.** Software here must not crash on bad input,
  must not silently lose a user's words, and must yield to the human
  using the machine. The machine serves the man, never the reverse.
- **No silent cuts.** Nothing in this line quietly throws away context,
  memory, or data. When something must be trimmed, it is logged, with
  counts, where the owner can see it.

## What we are adding

1. **Full KV-cache quantization.** The founding purpose of this line.
   The upstream cache is full-precision bf16 end to end; ours will
   store the KV cache quantized — q8_0 first, q4_0 where the attention
   path takes it safely — chosen the Ollama way, by a
   `GUFO_KV_CACHE_TYPE` setting, with a bad value earning a clear,
   named error instead of silent weirdness. **Speed is law:** a
   quantized mode ships only when benchmarks prove it does not slow
   generation down — and where cache traffic is the bottleneck, in
   long conversations, it should run faster than full precision.
   Correctness is proven on probes, and any drift is reported plainly.
2. **No KV limits except by setting.** Cache sizes and ceilings in
   this line are governed by settings the operator chooses — never by
   hard-coded caps smuggled into the machinery.
3. **Whatever else the house needs from the Gemma path**, as the work
   proves it: kernel fixes, layout improvements, and honest
   measurements, each earning its place under real load on real
   hardware — the Radeon 8060S (gfx1151) of Volmarr's own machine
   first of all.

## Two-track governance

- **Track one — our line (sovereign).** The branch `yrsa/our-gemma4`
  is the running line: the extension we build, install, and live with.
  It is also the proving ground where improvements mature under real
  use.
- **Upstream tracking.** The dev clone carries `origin` pointing at
  franzmoca/gemma4-gufo-extension. Over time: fetch upstream, merge it
  into our line, and on our line **our features win conflicts**.
  Upstream goodness flows in; our sovereignty does not flow out.
- **Track two — donations (separate, later, no hurry).** When a mature
  improvement is chosen for donation, it gets its **own** clean branch
  named `donate/<feature>`, based on upstream's **current** main, with
  the improvement cherry-picked and reshaped to comply fully with
  upstream's standards, protocols, and style — none of our
  house-specific extras. Donations are gifts, prepared carefully, one
  at a time, when they are ready and not before.

## The rule about this file

**This charter is a house document. It is never donated.** It must
never appear on any `donate/*` branch, in any upstream pull request,
or in any patch offered to the official extension. The same holds for
every house-only file on our line (the campaign state document, and
anything else that exists only because this is RuneForgeAI's personal
line).

Therefore the donation procedure includes a hard step, every time:
**before any `donate/*` branch is pushed or any upstream PR is opened,
verify that this charter and all house-only files are absent from it.**
A donation that carries house files is not a donation; it is a leak,
and it does not ship.

---

*RuneForgeAI Mad Data Scientist Personal Gemma4 Extension Hacks —
built by Volmarr's forge, for Volmarr's machines, to Volmarr's
standards.*
