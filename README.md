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
- **Future:** club tracking, multi-swing trends, quantitative double-pendulum
  simulation.

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

# Just the measurements, no LLM call
swingsense analyze swing.mp4 --features-only

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
