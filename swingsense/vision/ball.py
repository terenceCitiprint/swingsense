"""Ball-departure impact detection (owner's method).

The ball rests at a fixed spot until it is struck; the first frame in which it
leaves that spot is impact+1, so impact = (ball-departure frame) - 1. The ball
is far easier to track than a motion-blurred clubhead: its location is static,
so a temporal frame-difference in a small window around it is ~0 until the
strike, then spikes as the ball vanishes.

This is the most reliable impact signal we have on face-on range footage, and
it directly defines the "mid" pillar between backswing and follow-through.
"""

from __future__ import annotations

import numpy as np

from .pose import PoseTrack


def find_ball(video_path: str, setup_frame: int, pose: PoseTrack):
    """Locate the ball near setup: the static bright blob on the mat in front of
    the golfer that DISAPPEARS during the swing. Returns (x_px, y_px) or None."""
    import cv2

    W, H = pose.width, pose.height
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, setup_frame)
    ok, setup = cap.read()
    if not ok:
        cap.release()
        return None
    g0 = cv2.cvtColor(setup, cv2.COLOR_BGR2GRAY)

    # A late frame (likely after the strike) to confirm the ball has gone.
    later = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1, setup_frame + int(pose.fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, later)
    ok, lf = cap.read()
    cap.release()
    if not ok:
        return None
    g1 = cv2.cvtColor(lf, cv2.COLOR_BGR2GRAY)

    # Bright, small, roundish blobs present at setup; prefer ones that vanish.
    _, thr = cv2.threshold(g0, 175, 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best, best_score = None, -1.0
    for c in cnts:
        a = cv2.contourArea(c)
        if not (4 < a < 300):
            continue
        x, y, w, h = cv2.boundingRect(c)
        ar = w / max(h, 1)
        cx, cy = x + w // 2, y + h // 2
        if cy < H * 0.45 or not (0.4 < ar < 2.5):
            continue
        # disappearance: how much darker the spot got later
        gone = float(g0[cy, cx]) - float(g1[cy, cx])
        score = gone + 0.1 * a
        if score > best_score:
            best_score, best = score, (cx, cy)
    return best


def ball_departure(
    video_path: str, ball_xy: tuple[int, int], lo: int, hi: int, r: int = 11
) -> tuple[int, float] | None:
    """Return (impact_frame, confidence) from the ball-ROI frame-difference spike.

    The spike frame is impact+1 (the ball has moved), so impact = spike - 1.
    Confidence is the spike height over the local baseline.
    """
    import cv2

    bx, by = ball_xy
    cap = cv2.VideoCapture(video_path)
    prev = None
    diffs: list[tuple[int, float]] = []
    for i in range(hi):
        ok, fr = cap.read()
        if not ok:
            break
        if i < lo - 1:
            continue
        roi = cv2.cvtColor(
            fr[by - r : by + r, bx - r : bx + r], cv2.COLOR_BGR2GRAY
        ).astype(np.float32)
        if prev is not None and i >= lo:
            diffs.append((i, float(np.abs(roi - prev).mean())))
        prev = roi
    cap.release()
    if not diffs:
        return None
    vals = np.array([d for _, d in diffs])
    k = int(np.argmax(vals))
    spike = vals[k]
    baseline = float(np.median(vals))
    if spike < 3 * max(baseline, 0.3):  # not a clean departure
        return None
    return diffs[k][0] - 1, float(spike / max(baseline, 0.3))
