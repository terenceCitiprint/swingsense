# SwingSense — session handoff

Read this first when picking up in a fresh session. It points you at the
validated method, the working code, and the one open problem.

## What SwingSense is
A golf-swing analysis tool. The owner describes a swing "feel"; the tool helps
understand the movement, suggests mechanics from coaches, and cross-references
with physics. The work has concentrated on **swing-event detection ("stitches")**:
producing a 5-pillar montage from an uploaded swing video.

The five pillars (in order):
**SET UP → PEAK BACKSWING → IMPACT → FOLLOW THROUGH → BALANCE**
with skeleton overlay, ball marker, and a club-shaft line on each frame.

## The method that works (validated by the owner on 6 swings)

**The ball-first rule (CRITICAL — do not deviate):**
1. Find **IMPACT from the ball FIRST.** The ball rests on a fixed spot and is
   present every backswing frame, absent every follow-through frame.
   `impact = ball-departure frame − 1`. (`vision/ball.py`)
2. **PEAK backswing = the club extreme BEFORE impact.** It must pass the
   "is the ball still present?" test.
3. **FOLLOW THROUGH = the club extreme AFTER impact.**
4. **Never label phases from body posture.** Impact posture mimics address;
   a held finish mimics a backswing top. The ball is the only unambiguous arbiter.

**Why this matters (the failure it prevents):** hands peak *before* the club
(second-pendulum lag), and the follow-through wrap takes the hands *higher* than
the real top. Posture- or hands-based detection therefore inverts the swing —
labeling follow-through as the backswing top. Every early wrong stitch made this
error. See `docs/GROUND_TRUTH.md` for the owner-verified frame corpus and the
full lesson log.

## Club-shaft tracing (owner's method)
"Look for the straight line with a club head at the end of the shaft. The shaft
is a straight line viewed at an angle." Trace it by contrast/motion — never guess
a line. Practically:
- **moving frames:** motion-trace (frame differencing) — the club is the fast
  object; background cancels.
- **near-stationary frames (top, finish):** dark silhouette against the sky.
- **setup / impact:** anchor on the ball.

## The one open problem: robust club/shaft tracking
This is the next build the owner explicitly chose ("Build a real tracker").
Every static-image heuristic failed (Hough, dark-line, morphology, radial ray):
in a silhouette the shaft connects to the dark body, the background has its own
dark straight lines (net poles, horizon), and motion-only tracking latches onto
legs/feet on slow top/finish frames.

**Recommended approach (not yet built): arc-fitting.**
The clubhead path is a smooth arc on the swing plane. Instead of locking each
frame greedily:
1. Collect candidate clubhead points across the whole window (motion + darkness +
   strong oriented-gradient — the shaft is a long straight edge even over the body).
2. Fit the global arc (ties into the pendulum plane model) and snap detections
   to it; reject outliers (legs, poles) as off-arc.
3. Use a fixed-length shaft prior (constant pixel length from a fixed camera).
4. Bidirectional temporal smoothing, not greedy frame-to-frame.
5. Emit "unobserved" with a confidence flag rather than a wrong line — a wrong
   club line is worse than none for angle analysis.

Relevant files (all WIP, read the docstrings):
- `vision/club_tracker.py` — motion-based clubhead tracker (setup/impact correct; legs confound top/finish).
- `vision/club_radial.py` — radial dark-ray shaft search (works silhouetted, fails over body).
- `vision/pillars.py` — pillar detection via arc-angle + ball-anchored impact + neighbor-verified extrema.

## Also pending
- **Stitch `VID20220714WA0009.mp4` (a.k.a. v0009.mp4).** Deferred — it's a
  practice swing with a body-occluded ball, so impact couldn't be cleanly pinned.
  Ball at ~(395,390); the real swing appears to end in a held finish ~508–548.
  Re-attempt once the robust club tracker exists.

## Repo / branch facts
- Develop on branch `claude/serene-johnson-Co96K`.
- Commit footers: `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>` plus
  the `Claude-Session:` line. Do NOT put the model identifier in commits/PRs/code.
- Don't open a PR unless explicitly asked.

## Cost notes (for reference)
Production tool ≈ 5¢/swing (CV runs locally, free). Interactive frame-reading by
Claude ≈ $0.30–$1.50/swing. Image tokens ≈ (w×h)/750.
