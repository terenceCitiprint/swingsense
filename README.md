# SwingSense

Bridge what your golf swing **feels** like and what's **actually happening**.

Core principle: *feel ain't real*. A described feel is a hypothesis about what
the body might be doing — never ground truth. SwingSense translates feels into
candidate mechanics, cross-references them against physics principles and
coaching frameworks, surfaces feel-vs-real contradictions, and proposes cues or
drills to test.

## Phases

- **Phase 0 — text-only feel translator (done):** describe a swing feel, get a
  structured analysis grounded in a curated knowledge base.
- **Phase 1 — video pose extraction (done):** single-camera 2D pose via
  MediaPipe → swing events (address/top/impact) + biomechanics proxies (tempo,
  hip-shoulder separation, head movement) with honest confidence gating.
- **Phase 2 — visual report (done):** `--report out.html` writes a two-tier
  HTML report: plain-language read up top ("the read"), visuals below ("for
  nerds") — a panoramic strip of velocity-colored skeletons with the hand-path
  arc, the kinematic-sequence chart (who fires when), hand-speed profile,
  hip–shoulder separation curve, and an interactive 3D loop of the transition
  window you can rotate and scrub.
- **Phase 3 — corrective & longitudinal visuals (done):** the 3D loop gains a
  **ghost skeleton** — your own recorded motion re-timed so the pelvis leads
  (never invented positions, just fixed timing) — shown beside the actual swing.
  `swingsense compare before.mp4 after.mp4` writes a phase-aligned overlay of
  two swings with metric deltas; `swingsense trends` charts tempo and
  separation across your logged history.
- **Future:** club tracking, quantitative double-pendulum simulation.

## Install

```bash
pip install -e .            # core (text-only)
pip install -e ".[vision]"  # + video analysis (opencv, mediapipe, numpy)
```

## Usage

```bash
export ANTHROPIC_API_KEY=sk-ant-...

# Translate a feel
swingsense feel "felt like I was hanging back and flipping at it" --club 7i

# Analyze a swing video (optionally with the feel)
swingsense analyze swing.mp4 --feel "felt smooth but the ball went right" -c driver

# Full visual report (panorama, sequence chart, 3D loop) — works with or
# without the LLM step
swingsense analyze swing.mp4 --report report.html
swingsense analyze swing.mp4 --features-only --report report.html  # no API cost

# Compare two swings (before/after a drill) — phase-aligned overlay + deltas
swingsense compare before.mp4 after.mp4 --report comparison.html

# Trends across your logged swings (tempo, separation, confidence over time)
swingsense trends --report trends.html

# Just the measurements, no LLM call
swingsense analyze swing.mp4 --features-only

# Compare two swings (before/after a drill), phase-aligned
swingsense compare before.mp4 after.mp4 -o compare.html

# Are the drills working? Tempo + separation across your history
swingsense trends -o trends.html

# History & knowledge base
swingsense history
swingsense show 3
swingsense kb
```

Data lives in `~/.swingsense/` (override with `SWINGSENSE_HOME`). The reasoning
model defaults to Sonnet; override with `SWINGSENSE_MODEL`.

## Knowledge base

`swingsense/kb/` holds plain YAML — physics principles and coaching frameworks
the engine cites by name. The seed content is intentionally general; replace it
with the coaches and philosophies you trust.
