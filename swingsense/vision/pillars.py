"""Pillar detection v3 — clubhead distance from the ball (owner's principle).

The ball's position is where the clubhead rests at setup. Relative to that
point, the clubhead's distance d(t) traces an M:

    ~0 (setup)  ->  MAX (peak backswing: the coil begins to reverse)
                ->  ~0 (impact: club returns to the ball)
                ->  MAX again (follow-through)

Peak backswing = the clubhead's FARTHEST point from the ball, NOT its highest
point — height fails when the follow-through wrap climbs above the real top.
Impact is the deep minimum BETWEEN the two maxima, regardless of body position.
The last valid M in the clip is the real swing (practice swings are followed by
re-address; waggles never get far from the ball; celebration has no setup).
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


def detect_pillars(pose: PoseTrack, club: ClubTrack, debug: bool = False) -> dict | None:
    """Locate the pillar frames from the clubhead's distance to the ball."""
    fps = pose.fps
    n = pose.n_frames
    wrist = pose.midpoint(L_WRIST, R_WRIST)
    wx, wy = _smooth(wrist[:, 0]), _smooth(wrist[:, 1])
    speed = np.hypot(np.gradient(wx), np.gradient(wy)) * fps
    speed = np.where(pose.detected, speed, np.inf)  # frozen pose is not "still"

    head = _interp_track(club)
    best = None
    log = []

    for s_lo, s_hi in _stillness_periods(speed, 0.08, max(int(fps * 0.35), 3)):
        # Ball = clubhead resting position over the stillness tail.
        win_lo = max(s_lo, s_hi - int(fps * 0.8))
        idx = [j for j in range(win_lo, s_hi + 1) if club.found[j]]
        if len(idx) < 2:
            log.append(f"still[{s_lo}-{s_hi}]: no club at rest")
            continue
        ball = np.median(club.head_xy[idx], axis=0)

        # Distance-from-ball trace after this stillness.
        look_hi = min(n, s_hi + int(fps * 3.5))
        d = np.linalg.norm(head[s_hi:look_hi] - ball, axis=1)
        if len(d) < int(fps * 0.6):
            log.append(f"still[{s_lo}-{s_hi}]: clip ends")
            continue

        # Impact FIRST: the deep minimum flanked by two highs (the M's middle).
        # The global max of d may be EITHER the top or the follow-through (the
        # wrap can travel farther from the ball than the top), so anchoring on
        # the valley between maxima is the only order that always works.
        m = len(d)
        pad = max(int(fps * 0.2), 3)
        imp_rel, best_v = None, 0.0
        for t in range(pad, m - pad):
            if d[t] > 0.16:  # the club must be essentially back AT the ball
                continue
            before = float(np.max(d[:t]))
            after = float(np.max(d[t:]))
            v = min(before, after)
            if before >= 0.22 and after >= 0.18 and v - d[t] > best_v:
                imp_rel, best_v = t, v - d[t]
        if imp_rel is None:
            log.append(
                f"still[{s_lo}-{s_hi}]: no ball-return flanked by two highs "
                f"(max dist {float(np.max(d)):.2f})"
            )
            continue
        d_imp = float(d[imp_rel])

        # Peak backswing: farthest from the ball BEFORE impact (coil reversal).
        top_rel = int(np.argmax(d[:imp_rel]))
        d_top = float(d[top_rel])

        # Follow-through peak: farthest from the ball after impact.
        f_hi = min(m, imp_rel + int(fps * 1.4))
        fol_rel = (
            imp_rel + int(np.argmax(d[imp_rel:f_hi])) if f_hi > imp_rel + 1 else imp_rel
        )

        best = {
            "setup": max(s_lo, s_hi - 4),
            "top": s_hi + top_rel,
            "impact": s_hi + imp_rel,
            "follow_peak": s_hi + fol_rel,
            "ball": ball,
            "d_top": d_top,
            "d_imp": d_imp,
        }
        log.append(
            f"still[{s_lo}-{s_hi}]: VALID top=+{top_rel} imp=+{imp_rel} "
            f"(d {d_top:.2f}->{d_imp:.2f})"
        )

    if debug:
        for line in log:
            print(line)
    if best is None:
        return None

    # Balance: where motion settles after the follow-through peak.
    follow = best["follow_peak"]
    finish = min(n - 1, follow + int(fps * 0.4))
    for j in range(follow, n - 3):
        if np.all(speed[j : j + 3] < 0.16):
            finish = j
            break

    def t(i: int) -> float:
        return round(i / fps, 3)

    return {
        "setup": {"frame": int(best["setup"]), "t": t(best["setup"])},
        "top": {"frame": int(best["top"]), "t": t(best["top"])},
        "impact": {"frame": int(best["impact"]), "t": t(best["impact"])},
        "follow_peak": {"frame": int(follow), "t": t(follow)},
        "finish": {"frame": int(finish), "t": t(finish)},
        "ball": [round(float(best["ball"][0]), 3), round(float(best["ball"][1]), 3)],
        "quality": {
            "d_top": round(best["d_top"], 2),
            "d_impact": round(best["d_imp"], 2),
        },
    }
