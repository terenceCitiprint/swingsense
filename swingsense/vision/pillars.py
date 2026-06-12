"""Pillar detection (the owner's protocol), v2.

Pillars are detected from physical anchors, not fragile event labels. The
swing is found as the LAST occurrence of the full physical pattern:

    stillness  ->  club climbs  ->  club returns to the ball, fast

which is robust against practice swings (followed by re-address, so a later
valid pattern exists), waggles (club returns to the ball but never climbs),
and post-swing celebration (no setup stillness after it).

  1. SETUP   one still frame, ~4 frames before movement starts (end of the
             swing's stillness period). The clubhead's resting position during
             this stillness IS the ball position — free calibration.
  2. TOP     the club's height peak between setup and impact.
  3. IMPACT  the clubhead's return to the ball position (same club geometry as
             the setup still), moving fast.
  4. FOLLOW  the club's height peak after impact.
  5. BALANCE frames after the finish settles.

In-between frames are only analyzed relative to verified pillars.
"""

from __future__ import annotations

import numpy as np

from .club import ClubTrack, _interp_track
from .pose import L_WRIST, R_WRIST, PoseTrack


def _smooth(x: np.ndarray, k: int = 5) -> np.ndarray:
    if len(x) < k:
        return x
    pad = k // 2
    return np.convolve(np.pad(x, pad, mode="edge"), np.ones(k) / k, mode="valid")


def _stillness_periods(speed: np.ndarray, thresh: float, min_len: int):
    """Yield (start, end) of runs where speed stays below thresh."""
    out = []
    run_start = None
    for j, s in enumerate(speed):
        if s < thresh:
            if run_start is None:
                run_start = j
        else:
            if run_start is not None and j - run_start >= min_len:
                out.append((run_start, j - 1))
            run_start = None
    if run_start is not None and len(speed) - run_start >= min_len:
        out.append((run_start, len(speed) - 1))
    return out


def detect_pillars(pose: PoseTrack, club: ClubTrack) -> dict | None:
    """Locate the pillar frames. Returns None when no swing is found."""
    fps = pose.fps
    n = pose.n_frames
    wrist = pose.midpoint(L_WRIST, R_WRIST)
    wx, wy = _smooth(wrist[:, 0]), _smooth(wrist[:, 1])
    speed = np.hypot(np.gradient(wx), np.gradient(wy)) * fps
    # Frames with no pose detection carry the last pose forward -> fake zero
    # speed. Mark them un-still so frozen-pose regions can't masquerade as the
    # setup stillness.
    speed = np.where(pose.detected, speed, np.inf)

    # Absolute thresholds (coords are normalized, so units are
    # camera-independent): a resting golfer's hands drift < ~0.08 frame/s; a
    # downswing moves them > ~0.5 frame/s.
    still_thresh = 0.08
    fast_thresh = 0.5
    head = _interp_track(club)

    best = None  # the LAST stillness that launches a valid swing pattern
    for s_lo, s_hi in _stillness_periods(speed, still_thresh, max(int(fps * 0.4), 4)):
        # Ball: clubhead resting position over the tail of the stillness.
        win_lo = max(s_lo, s_hi - int(fps * 0.8))
        idx = [j for j in range(win_lo, s_hi + 1) if club.found[j]]
        if len(idx) < 3:
            continue
        ball = np.median(club.head_xy[idx], axis=0)

        # Look ahead for the climb-and-return pattern.
        look_hi = min(n, s_hi + int(fps * 3.0))
        seg_y = head[s_hi:look_hi, 1]
        if len(seg_y) < int(fps * 0.5):
            continue
        climb = float(ball[1] - np.min(seg_y))  # how high the club got (y up = smaller)
        if climb < 0.12:  # no real backswing climbed from this stillness
            continue
        top_rel = int(np.argmin(seg_y))
        # Return to the ball AFTER the climb, moving fast.
        r_lo, r_hi = s_hi + top_rel, min(n, s_hi + top_rel + int(fps * 0.8))
        cands = [
            j
            for j in range(r_lo, r_hi)
            if speed[j] >= fast_thresh * 0.6
            and float(np.linalg.norm(head[j] - ball)) < 0.12
        ]
        if not cands:
            continue
        impact = min(cands, key=lambda j: float(np.linalg.norm(head[j] - ball)))
        best = (s_lo, s_hi, ball, s_hi + top_rel, impact)

    if best is None:
        return None
    s_lo, s_hi, ball, top, impact = best
    setup = max(s_lo, s_hi - 4)

    # Follow-through peak: club height peak after impact (fallback: hands).
    f_hi = min(n, impact + int(fps * 1.5))
    if club.found[impact:f_hi].sum() >= 3:
        follow = impact + int(np.argmin(head[impact:f_hi, 1]))
    else:
        follow = impact + int(np.argmin(wy[impact:f_hi])) if f_hi > impact + 1 else impact

    # Balance: where motion settles after the follow-through.
    finish = min(n - 1, follow + int(fps * 0.4))
    for j in range(follow, n - 3):
        if np.all(speed[j : j + 3] < still_thresh * 2):
            finish = j
            break

    def t(i: int) -> float:
        return round(i / fps, 3)

    return {
        "setup": {"frame": int(setup), "t": t(setup)},
        "top": {"frame": int(top), "t": t(top)},
        "impact": {"frame": int(impact), "t": t(impact)},
        "follow_peak": {"frame": int(follow), "t": t(follow)},
        "finish": {"frame": int(finish), "t": t(finish)},
        "ball": [round(float(ball[0]), 3), round(float(ball[1]), 3)],
    }
