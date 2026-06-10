# SwingSense

A command-line tool that bridges the gap between what a golf swing **feels**
like and what is **actually** happening — because *feel ain't real*.

You describe a swing in plain words; SwingSense translates that feel into
candidate body mechanics, cross-references them against a curated physics /
biomechanics knowledge base and coaching frameworks, flags feel-vs-real
contradictions, and suggests cues or drills you can test. Everything is logged
locally so the tool builds up a picture of *your* patterns over time.

> See [`PLAN.md`](PLAN.md) for the full architecture and roadmap. This repo is
> at **Phase 1**: a text feel translator **plus** a video pipeline that extracts
> pose and biomechanics from a swing clip.

## Install

```bash
pip install -e .              # core (feel translator)
pip install -e ".[vision]"    # + video analysis (opencv, mediapipe)
export ANTHROPIC_API_KEY=sk-ant-...
```

On a slim Linux box the MediaPipe runtime may need a system GL library:
`apt-get install -y libgles2 libegl1 libgl1`. The pose model (~9 MB) is
downloaded and cached on first `analyze`.

## Use

```bash
# Analyze a swing video: extract pose + biomechanics, reason over them with feel
swingsense analyze swing.mp4 --feel "felt stuck behind me" --club driver

# Just the measured video features, no LLM call (no API key needed)
swingsense analyze swing.mp4 --features-only

# Translate a feel into mechanics, cross-referenced with physics (no video)
swingsense feel "felt stuck behind me, hands too active" --club driver

# Preview the reasoning prompt WITHOUT calling the API (no key needed)
swingsense feel "trying to feel a flat lead wrist at the top" --dry-run

# Review your logged swings and their one-line reads
swingsense history --last 10

# Re-open the full analysis for a past swing
swingsense show 3

# Inspect the knowledge base the engine reasons over
swingsense kb

# Version + config (model, data dir, whether the API key is set)
swingsense version
```

## How it works (Phase 0)

```
feel text ──▶ prompt (feel + knowledge base + recent history) ──▶ Claude
   ──▶ structured analysis (mechanics, physics refs, contradictions, drills)
   ──▶ rendered to terminal + saved to local SQLite history
```

- **Reasoning model**: Claude via the Anthropic API. Override with
  `SWINGSENSE_MODEL` (defaults to a Sonnet id; point at an Opus id for deeper
  reasoning).
- **Knowledge base**: plain YAML under `swingsense/kb/` — `physics/` and
  `coaching/`. It's *yours*: edit and extend it so suggestions reflect the
  coaches and philosophy you trust. The seed coaching content is intentionally
  general — replace it.
- **Memory**: a local SQLite database in `~/.swingsense/` (override with
  `SWINGSENSE_HOME`).

## Configuration

| Variable            | Purpose                                   | Default              |
|---------------------|-------------------------------------------|----------------------|
| `ANTHROPIC_API_KEY` | Auth for the Claude reasoning layer       | _(required)_         |
| `SWINGSENSE_MODEL`  | Reasoning model id                        | `claude-sonnet-4-6`  |
| `SWINGSENSE_HOME`   | Where history + overrides live            | `~/.swingsense`      |

## Roadmap

Phase 0 (feel translator) ✓ → **Phase 1** video pose + first biomechanics
features (events, tempo, rotation proxies, head movement) ✓ → **Phase 2** deeper
features (true kinematic sequence, X-factor, club tracking, launch-monitor OCR)
→ **Phase 3** richer cross-reference on measured data → **Phase 4** quantitative
physics (double-pendulum → inverse dynamics). Details in [`PLAN.md`](PLAN.md).

### Honesty about Phase 1 limits

Single-camera 2D pose is a **proxy**: depth and true 3D turn are approximate, the
**club is not tracked**, and low frame rates blur the downswing. Every analysis
reports a **confidence** and **reliability flags** — e.g. a swing where the
golfer is small in frame (poor hand tracking) is marked low-confidence, and
tempo measured below the camera's time resolution is flagged approximate. Shoot
**120–240 fps**, fill the frame with the body, and use a plain background for the
best results.
