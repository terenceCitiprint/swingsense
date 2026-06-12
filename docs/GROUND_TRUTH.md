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

### Owner-verified frames (corrected AGAIN — ball-anchored, final for Mika)

The owner caught that the earlier 'peak 140' showed an already-gone ball.
Zoomed ball-region strips settled the truth: ball on tee through 129,
clubhead ON the ball at 131, ball gone at 132. The frame once labeled
'setup 130' was the downswing's last instant (impact posture looks like
address); 136-144 'backswing' was follow-through — frame 142, the old
detector's 'top', is actually the FOLLOW-THROUGH peak.

| Clip | setup | PEAK back | impact | follow peak | hold | source |
|---|---|---|---|---|---|---|
| Mika  | 90 | **117** (apex 114-118) | **131** (ball gone 132) | 142 | 158 | ball strips + owner photo |
| Trisha| 381 | ~418-420 | TBD | ~447 | — | peak=photo; ball position unresolved |
| Terence June 2023 (DTL, terence_june192023.mp4) | 172 | 204 (zone 202-206) | **211** (ball gone 212, 16x ROI spike; yellow spare ball as control) | 230 | 244-256 held | full ball-anchored method, clean first pass |
| Terence July 2023 (DTL, terence_july_1_2023.mp4) | 34 | 52 | **59** (ball present @59, gone @60) | 96 | 120 held | full swing — NOT a rehearsal (I first inverted it) |
| Terence March 2023 (DTL, terence_march_2023.mp4) | 60 | 85 | **98** (ball gone @99; clubhead contacts @98) | 115 | 140 held | ball-first method; finish 108-148 again mimicked a backswing top |
| Lloyd range (face-on, lloyd.mp4) 480x848@30 | 100 | 122 | **131** (ball gone @132; clubhead contacts @131, 18x ROI spike) | 156 | 172 held | new person; clean first pass, ball-first |

July correction: I first called the July clip a "backswing rehearsal, no
impact." WRONG. It is a full swing; I read the FINISH (club held over the
shoulder ~80-114) as a backswing top. The ball settled it: present through
59, gone at 60 => impact = 59. RULE that prevents inversion: find IMPACT
from the ball FIRST, then peak = the club extreme BEFORE it, follow = AFTER.
Never label phases from body posture (impact posture mimics address; a held
finish mimics a backswing top).

Terence June notes: down-the-line view, 1080x1920@30. Tempo ~4:1
(takeaway ~176 -> peak 204 -> impact 211). Finish held rock-still 244-256+.
Best candidate clip for the first in-between (plane/force) analysis.

LESSON: never trust a swing timeline until the BALL has been watched
through it. Impact posture mimics address; follow-through mimics backswing.
The ball is the only unambiguous arbiter — it is present for every
backswing frame and absent for every follow-through frame.

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
