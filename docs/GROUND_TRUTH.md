# Ground-truth corpus (owner-verified swing events)

| Clip | File (session upload) | True events (frames) | Notes |
|---|---|---|---|
| Terence indoor sim | `f1772bb7-...mp4` 480x848@24 | address ~95-99, top ~119, impact ~126, follow-peak ~138, recoil 140+ | Fast swing. Follow-through wrap takes the HANDS higher than the real top — hands-only detection mislabels follow-through as top. Club tracking resolves it. |
| Terence outdoor range | `7f8b0252-...mp4` 352x640@30 | full swing; storyboard verified "perfectly captured" by owner (pose events addr 122 / top 150 / imp 165) | Club tracker suggests an earlier swing ~108-125 too — unverified; possibly a practice swing. |
| Terence range backswing-only ×2 | `swing3/4.mov` 1080x1920@30 | address ~28, top ~67; NO downswing/impact (clip ends at top) | Must classify `backswing_only`. |
| Trisha range | `swing5.mp4` 576x1024@30 | address ~385, mid-bsw ~411, top ~427, impact ~430; frames after ~440 are CELEBRATION, not swing | Pre-shot waggle V around 378-387 fools deepest-V selection. |
| Mika range | `mika.mp4` 576x768@30 | true top (club peak, reference image) ~148-150; hands peak ~142-144; impact ~157 | Club keeps loading ~0.2s after hands peak (second pendulum). Practice swing ~frames 47-110. Downswing blur breaks club detection density. |

## Two objective signals that match owner ground truth (the breakthrough)

After many wrong stitches (all biased LATE on the peak — catching the early
downswing because the hands/eyeball cues lag the clubhead), two objective
signals finally matched the owner's reference images and method:

1. **PEAK backswing = clubhead FARTHEST from the ball, where it reverses.**
   Not "highest" and not hand-based. For Mika this is frame 140 (club dist
   from ball peaks at 0.66 then reverses) — and frame 140 matches the owner's
   supplied peak photo. My earlier "144"/"148" were 4-8 frames late: the club
   lays off and looks high while already descending (second-pendulum lag).

2. **IMPACT = ball-departure - 1 (owner's method).** The ball is static until
   struck; a frame-difference in a small ROI at the ball is ~0 until it spikes
   as the ball vanishes. Mika: ROI flat through the downswing, spikes at frame
   150 -> impact = 149 (confidence 6.6x baseline). Implemented in vision/ball.py.

### Owner-verified frames (corrected)

| Clip | setup | PEAK back | impact | follow peak | source |
|---|---|---|---|---|---|
| Mika  | 130 | **140** | **149** | ~172 | peak=dist+photo, impact=ball-departure |
| Trisha| 381 | ~418-420 | TBD | ~447 | peak=photo; ball auto-locate still wrong |

Open: ball auto-locator picks the wrong blob on Trisha; her tee at (385,745)
reads ambiguous (occlusion / wrong position). Follow-peak via club-distance is
unreliable (club goes behind the head). PEAK detection needs denser club
tracking through the backswing-against-net.

## Key lessons encoded so far

1. **Hands ≠ club.** The hands peak before the club finishes loading, and the
   follow-through wrap can take the hands higher than the top. Both break
   hands-only event detection. The club head's trajectory passes the ball
   exactly once — impact — which splits backswing from follow-through.
2. **The pose layer reliably brackets WHEN the swing happens** even when it
   mislabels phases within it. Use it as the club tracker's search window
   (excludes practice swings before, camera junk after).
3. **Impact = deepest V** in the clubhead height trace (high before = top,
   high after = follow-through), in a densely-detected region.
4. Detection requires motion edges (frame differencing) — static-edge Hough
   latches onto screen/mat lines.

## Current status (club tracker, experimental)

- Indoor: ✅ top 119 / impact 127 / follow 145 (truth 119/126/138)
- Backswing-only: ✅ correctly rejected (falls back to pose, classified bsw-only)
- Outdoor: plausible but unverified vs the owner-confirmed pose storyboard
- Trisha: ❌ picks waggle V at 378 (truth ~430)
- Mika: ❌ no valid candidate in window (downswing blur) — pose fallback
