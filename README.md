# SwingSense

A command-line tool that bridges the gap between what a golf swing **feels**
like and what is **actually** happening — because *feel ain't real*.

You describe a swing in plain words; SwingSense translates that feel into
candidate body mechanics, cross-references them against a curated physics /
biomechanics knowledge base and coaching frameworks, flags feel-vs-real
contradictions, and suggests cues or drills you can test. Everything is logged
locally so the tool builds up a picture of *your* patterns over time.

> See [`PLAN.md`](PLAN.md) for the full architecture and roadmap. This repo is
> currently at **Phase 0: the text-only feel translator** (no video yet).

## Install

```bash
pip install -e .          # or: uv pip install -e .
export ANTHROPIC_API_KEY=sk-ant-...
```

## Use

```bash
# Translate a feel into mechanics, cross-referenced with physics
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

Phase 0 (this) → **Phase 1** video pose extraction → **Phase 2** biomechanics
features (kinematic sequence, X-factor, tempo) → **Phase 3** cross-reference
engine on measured data → **Phase 4** quantitative physics (double-pendulum →
inverse dynamics). Details in [`PLAN.md`](PLAN.md).
