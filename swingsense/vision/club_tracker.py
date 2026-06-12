"""Real club tracker — motion-based clubhead trajectory.

Why this exists: every static-image heuristic (dark line, Hough, morphology)
fails because in a silhouette the shaft connects to the dark body, and the
background has its own dark straight lines (net poles, horizon). The one
property that separates the club from ALL of that is MOTION: across the swing
the club is the fast-moving object, the background is static, and the body
moves comparatively slowly.

Method:
  1. Per frame, motion = max frame-difference vs the immediate neighbours.
     Static background (net, horizon, trees) cancels to ~0.
  2. The clubhead is the moving pixel-cluster FARTHEST from the body centre
     (mid shoulders/hips) — the club is swung out on a long lever, so its head
     is the outermost moving point.
  3. Track it with temporal continuity (it moves smoothly along an arc); fill
     low-motion gaps (top, finish) by interpolation of the trajectory.
  4. The shaft at any frame = line from the grip (pose wrists) to the clubhead.

This isolates the club without depending on darkness or the pose grip for the
detection itself (the grip is only used to draw the final shaft line).
"""

from __future__ import annotations

import numpy as np

from .pose import L_HIP, L_SHOULDER, R_HIP, R_SHOULDER, PoseTrack


def track_clubhead(video_path: str, pose: PoseTrack, lo: int, hi: int):
    """Return (heads, found): per-frame clubhead xy (normalized) over [lo,hi]."""
    import cv2

    W, H = pose.width, pose.height
    n = pose.n_frames
    body = (pose.midpoint(L_SHOULDER, R_SHOULDER) + pose.midpoint(L_HIP, R_HIP)) / 2

    cap = cv2.VideoCapture(video_path)
    grays = {}

    def gray(i):
        if i in grays:
            return grays[i]
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, fr = cap.read()
        grays[i] = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).astype(np.float32) if ok else None
        return grays[i]

    heads = np.full((n, 2), np.nan)
    raw = []
    for i in range(lo, hi):
        g0, g1, g2 = gray(i - 1), gray(i), gray(i + 1)
        if g1 is None:
            continue
        mot = np.zeros_like(g1)
        if g0 is not None:
            mot = np.maximum(mot, np.abs(g1 - g0))
        if g2 is not None:
            mot = np.maximum(mot, np.abs(g1 - g2))
        m = (mot > 30).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        nlab, lab, stats, cent = cv2.connectedComponentsWithStats(m)
        bx, by = body[i, 0] * W, body[i, 1] * H
        best, bd = None, -1
        for k in range(1, nlab):
            if stats[k, cv2.CC_STAT_AREA] < 25:
                continue
            cx, cy = cent[k]
            d = np.hypot(cx - bx, cy - by)
            if d > bd:  # farthest moving cluster from the body = clubhead end
                bd, best = d, (cx, cy)
        if best is not None:
            raw.append((i, best[0], best[1]))
    cap.release()

    if len(raw) < 3:
        return heads, np.zeros(n, dtype=bool)

    # temporal continuity: reject points that jump too far from the running path
    raw = np.array(raw)
    idx = raw[:, 0].astype(int)
    xs, ys = raw[:, 1], raw[:, 2]
    # median-smooth then reject outliers > 18% of frame from the smooth path
    def smooth(a):
        out = a.copy()
        for j in range(len(a)):
            s = max(0, j - 2); e = min(len(a), j + 3)
            out[j] = np.median(a[s:e])
        return out
    sx, sy = smooth(xs), smooth(ys)
    keep = np.hypot(xs - sx, ys - sy) < 0.18 * H
    if keep.sum() >= 3:
        idx, xs, ys = idx[keep], xs[keep], ys[keep]
    # interpolate across all frames in window (fills top/finish low-motion gaps)
    allf = np.arange(lo, hi)
    ix = np.interp(allf, idx, xs)
    iy = np.interp(allf, idx, ys)
    found = np.zeros(n, dtype=bool)
    for j, f in enumerate(allf):
        heads[f] = [ix[j] / W, iy[j] / H]
        found[f] = True
    return heads, found
