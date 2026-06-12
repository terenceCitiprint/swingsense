# SwingSense — Planning Document

A command-line tool that ingests golf swing videos, lets you describe the
*feel* of a swing in plain language, and cross-references that against
biomechanics/physics and coaching frameworks to help you understand your
movement and suggest mechanical adjustments.

> Design principle: **"Feel ain't real."** The whole point of the tool is to
> bridge the gap between what a swing *feels* like (subjective, unreliable) and
> what is *actually* happening (objective geometry + physics). Everything below
> serves that bridge.

---

## 1. What the tool does (the loop)

```
  ┌──────────────┐     ┌───────────────────┐     ┌────────────────────┐
  │ Swing video  │ ──▶ │ Pose + club       │ ──▶ │ Kinematic features │
  │ (you record) │     │ extraction        │     │ (angles, timing)   │
  └──────────────┘     └───────────────────┘     └─────────┬──────────┘
                                                            │
  ┌──────────────┐                                          ▼
  │ "Feel" text  │ ─────────────────────────────▶ ┌────────────────────┐
  │ (you type)   │                                 │  Reasoning engine  │
  └──────────────┘                                 │  (Claude) maps     │
                                                    │  feel ⟷ mechanics, │
  ┌──────────────┐     ┌───────────────────┐        │  checks physics,   │
  │ Coaching KB  │ ──▶ │ Physics rules /   │ ─────▶ │  KB → suggestions  │
  │ (curated)    │     │ (later) simulation│        └─────────┬──────────┘
  └──────────────┘     └───────────────────┘                  │
                                                              ▼
                                                    ┌────────────────────┐
                                                    │ Insight + drills + │
                                                    │ logged to history  │
                                                    └────────────────────┘
```

---

## 2. The four layers

### Layer 1 — Video → Motion data
Turn a phone video into joint positions over time.

- **Pose estimation**: extract body keypoints (shoulders, hips, elbows, wrists,
  knees, ankles, head) per frame. Options:
  - `MediaPipe Pose` / BlazePose — easiest, runs locally, good enough to start.
  - `MMPose` / `MoveNet` — more accurate, heavier.
- **Club detection** is the hard part — pose models don't track the club. Start
  by having you mark grip + clubhead on a few key frames, or use a simple object
  tracker; upgrade later.
- **Reality check on accuracy**: a single phone gives **2D** pose — no true depth.
  For real biomechanics you eventually want **two synced camera angles**
  (down-the-line + face-on). v1 can work in 2D and be honest about its limits.

### Layer 2 — Motion data → Biomechanics features
Compute the things coaches and physics actually care about:

- **Kinematic sequence** — the proximal-to-distal firing order
  (pelvis → thorax → lead arm → club) and the *timing gaps* between peaks. This
  is the single most diagnostic feature in a golf swing.
- **X-factor** — hip-to-shoulder separation at the top, and the "X-factor
  stretch" early in the downswing.
- **Rotation & sway** — pelvis/thorax rotation, lateral movement, head stability.
- **Wrist conditions** — lead wrist flexion/extension (proxy for clubface),
  lag angle and release timing.
- **Tempo** — backswing:downswing ratio (the classic ~3:1).
- **Center of pressure trace** (later, with force data) — weight shift pattern.

### Layer 3 — Reasoning engine (Claude)
This is the brain that makes the tool *yours*. Using the Claude API:

- **Feel ⟷ mechanics translation**: you say "I feel like my hands are passive
  and I'm dropping it in the slot"; Claude maps that to candidate mechanical
  signatures and checks them against your measured features.
- **Cross-referencing**: compares your measured mechanics against the coaching
  knowledge base AND the physics rules, flags agreements/contradictions
  ("your *feel* of trapping it matches a measured negative attack angle, but
  your shaft lean says otherwise").
- **Suggestion generation**: produces drills/cues tied to *your* body and *your*
  measured gaps, citing which coach/principle it's drawing from.
- **Memory**: learns your recurring patterns and which feels/cues actually
  change your numbers over time.

### Layer 4 — Knowledge base (curated by you)
Structured, version-controlled files (YAML/JSON) so you control the philosophy:

- `physics/` — biomechanics principles as rules (kinematic sequence, angular
  momentum transfer, ground reaction forces, double-pendulum lag/release).
- `coaching/` — frameworks you trust (e.g. AMG-style data norms, TPI screens,
  single-plane vs rotary, specific coach cues), each tagged so suggestions can
  cite their source.
- These are *data*, not code — you can grow the KB without touching the engine,
  and Claude reasons over it (optionally via embeddings/RAG once it's large).

---

## 3. The physics — conceptual now, simulation later

**Phase A (conceptual):** encode biomechanics as *rules and reasoning* Claude
applies to your measured features. Concepts covered:
- Kinematic sequence / proximal-to-distal energy transfer
- X-factor & stretch (elastic energy storage)
- Angular momentum and the "release" as conservation/transfer
- Ground reaction forces conceptually (vertical push, horizontal, torque)

**Phase B (quantitative):** model the swing as a **double/triple pendulum**
(lead arm + club, hinged at the wrist) and compute torques, lag retention, and
release timing numerically (`numpy`/`scipy`). Later, **inverse dynamics** to
estimate joint torques from the motion. This is a real engineering effort — it's
deliberately deferred until the conceptual loop is useful.

---

## 4. Proposed tech stack

| Concern            | Choice                                  | Why |
|--------------------|-----------------------------------------|-----|
| Language           | **Python**                              | Best ecosystem for video, pose, biomechanics, ML |
| CLI framework      | **Typer** (or Click)                    | Clean subcommands (`analyze`, `feel`, `history`) |
| Video I/O          | **OpenCV**                              | Frame extraction, drawing overlays |
| Pose estimation    | **MediaPipe Pose** (start)              | Local, free, fast; upgrade to MMPose later |
| Numerics           | **numpy / scipy**                       | Feature computation, later the simulation |
| LLM reasoning      | **Anthropic Claude API** (`claude-opus-4-8` / `claude-sonnet-4-6`) | Feel↔mechanics reasoning, suggestions |
| Knowledge base     | **YAML/JSON files** (+ embeddings later)| You own and version the philosophy |
| Storage / memory   | **SQLite**                              | Swing history, features, what worked |
| Packaging          | **uv** or `pip` + `pyproject.toml`      | Reproducible CLI install |

---

## 5. CLI shape (target UX)

```bash
# Analyze a swing video and attach how it felt
swingsense analyze driver_2026-06-06.mp4 \
    --feel "felt like I was stuck behind me, hands too active"

# Just log/translate a feel without video
swingsense feel "trying to feel a flat lead wrist at the top"

# Review trends over time
swingsense history --club driver --last 10

# Show what the tool measured for one swing
swingsense show <swing-id>
```

---

## 6. What YOU need to provide

**Hardware / data**
- A phone that shoots **slow-motion (120–240 fps)** — golf is fast; 30 fps
  blurs impact. 240 fps is ideal.
- Consistent camera angles: **down-the-line** (camera behind, on hand-path line)
  and **face-on**. A cheap tripod. Same setup each time = comparable data.
- (Optional, later) a second phone for synced two-angle 3D, or any sensor/launch
  monitor data you can export.

**Accounts / keys**
- An **Anthropic API key** for the Claude reasoning layer.

**Domain input (the most valuable thing)**
- Which **coaches/philosophies** you trust — so we seed the knowledge base with
  *your* sources, not generic advice.
- Your **feel vocabulary** — the words you actually use, so translations land.

---

## 7. Roadmap (phased)

- **Phase 0 — Scaffold + Feel translator (text only).** CLI skeleton, Claude
  integration, SQLite history, first slice of the physics/coaching KB. You can
  log feels and get reasoned mechanical hypotheses. *Useful on day one, no CV yet.*
- **Phase 1 — Video → motion.** MediaPipe pose extraction, frame/event detection
  (address, top, impact), overlay rendering so you can see the tracking.
- **Phase 2 — Club tracking + deeper features.** PRIORITY: track the club shaft
  (line detection anchored at the hands + temporal consistency). Real footage
  proved the hands peak ~0.1-0.3s BEFORE the club finishes loading, so a
  hands-only tracker cannot see the true top of the backswing — the club is the
  second pendulum and carries the energy peak. Then: kinematic sequence,
  X-factor, launch-monitor screen OCR.
- **Phase 3 — Cross-reference engine.** Tie measured features + feel + KB into
  suggestions that cite their source and flag feel/real contradictions.
- **Phase 4 — Quantitative physics.** Double-pendulum model, then inverse
  dynamics. Two-camera 3D if you want real depth.

---

## 8. Honest limitations to design around

- **2D single-camera pose is approximate** — depth, true 3D rotation, and club
  data are limited until multi-view/sensors are added. The tool should always
  state its confidence.
- **Club tracking is unsolved by pose models** — needs its own approach.
- **Feel translation is probabilistic** — the tool proposes hypotheses to test
  against your numbers, it doesn't declare absolute truth.
- **Garbage in, garbage out** — inconsistent camera setup wrecks comparability.

---

## 9. Open questions to resolve before building

1. Which **coaches/philosophies** seed the knowledge base?
2. Down-the-line only to start, or two angles from the beginning?
3. Is there **launch-monitor or sensor data** you can already export to ground
   the feel-vs-real comparison sooner?
4. Preference on local-only processing vs. cloud for the heavier CV later?
```
