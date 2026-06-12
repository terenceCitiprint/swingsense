"""Radial shaft search (WIP) — toward accurate club tracking for plane analysis.

Why this exists: the Hough-fragment tracker in club.py grabs arbitrary edges and
produces a wrong "club line" most frames. Accurate club tracking is the
prerequisite for the relative-angle / swing-plane analysis the owner wants
(shaft angle at takeaway vs impact, plane consistency, lag, face proxy).

Method: the grip is known from pose. Cast rays from the grip at every angle and
score each by the length of its DARK run (the shaft is dark against a bright
sky/grass background). The best ray's far end is the clubhead.

STATUS — partial. Verified behaviour on the DTL July clip:
  works  : frames where the shaft is silhouetted against sky/grass (late
           backswing, address) — locks the real shaft cleanly.
  fails  : frames where the shaft crosses the DARK BODY (mid-backswing, top) —
           "dark-against-bright" has no contrast there; it grabs the body
           outline, a leg, or a net pole instead.

KNOWN NEXT STEPS (the path to robust tracking):
  1. Multi-cue score: darkness OR strong oriented-gradient (the shaft is a long
     straight edge even over the body), not darkness alone.
  2. Fixed-length prior: a given club's shaft is ~constant pixel length from a
     fixed camera; penalise runs far from the running median length.
  3. Bidirectional temporal smoothing (forward+backward) + outlier rejection,
     not greedy frame-to-frame continuity (one bad lock currently propagates).
  4. Optimise the WHOLE trajectory at once (the clubhead path is a smooth arc;
     fit an arc and snap detections to it) — ties directly to the pendulum
     plane model.
  5. Fall back to "unobserved" with a confidence flag rather than emitting a
     wrong line — a wrong club line is worse than none for angle analysis.
"""

from __future__ import annotations

import numpy as np

from .pose import L_WRIST, PoseTrack, R_WRIST


def _ray_length(gray, gx: int, gy: int, deg: float, max_len: int) -> tuple[int, float]:
    """March a ray from (gx,gy) at angle deg; return (dark-run length, dark fraction)."""
    H, W = gray.shape
    th = np.radians(deg)
    dx, dy = np.cos(th), np.sin(th)
    run = 0
    last = 0
    dark = 0
    tot = 0
    for r in range(10, max_len, 2):
        x, y = int(gx + dx * r), int(gy + dy * r)
        if not (0 <= x < W and 0 <= y < H):
            break
        tot += 1
        if int(gray[y, x]) < 95:
            run += 1
            last = r
            dark += 1
        else:
            run -= 2
            if run < -4:
                break
    return last, dark / max(tot, 1)


def detect_shaft(gray, grip: tuple[int, int], max_len: int, prev_deg: float | None):
    """Best shaft ray from the grip. Returns (head_xy, deg, length, dark_frac) or None."""
    gx, gy = grip
    best = None
    best_score = -1.0
    for deg in range(0, 360, 3):
        L, frac = _ray_length(gray, gx, gy, deg, max_len)
        if frac < 0.6 or L < max_len * 0.15:
            continue
        score = float(L)
        if prev_deg is not None:
            dd = abs((deg - prev_deg + 180) % 360 - 180)
            score -= 4.0 * dd  # temporal continuity (greedy — see module notes)
        if score > best_score:
            best_score = score
            th = np.radians(deg)
            best = ((int(gx + np.cos(th) * L), int(gy + np.sin(th) * L)), deg, L, frac)
    return best


def track_shaft(video_path: str, pose: PoseTrack):
    """Per-frame shaft estimates. WIP: see module docstring for accuracy caveats."""
    import cv2

    W, H = pose.width, pose.height
    wrist = pose.midpoint(L_WRIST, R_WRIST)
    cap = cv2.VideoCapture(video_path)
    prev_deg = None
    heads = np.full((pose.n_frames, 2), np.nan)
    angles = np.full(pose.n_frames, np.nan)
    for i in range(pose.n_frames):
        ok, fr = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        grip = (int(wrist[i, 0] * W), int(wrist[i, 1] * H))
        res = detect_shaft(gray, grip, int(0.5 * H), prev_deg)
        if res is not None:
            (hx, hy), deg, L, frac = res
            heads[i] = [hx / W, hy / H]
            angles[i] = deg
            prev_deg = deg
    cap.release()
    return heads, angles
