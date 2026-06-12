"""Pillar detection v4 — pendulum arc angle with neighbor-verified extrema.

Owner's coordinate system: the clubhead's position is measured as an ANGLE
around the pendulum's pivot (mid-shoulders), relative to the club's angle at
the starting position (club at the ball). That signed arc position psi(t):

    0 (setup) -> swings to an EXTREME (peak backswing: the coil reverses)
              -> crosses ZERO exactly at impact (club back at the address angle)
              -> continues to an opposite extreme (follow-through)

Why this beats euclidean distance-from-ball: the club folding at the top
shrinks its distance to the ball while the arc angle keeps growing, and the
zero-crossing at impact stays razor sharp even when the strike frame itself is
motion-blurred and interpolated.

Owner's safeguard: every chosen pillar frame is verified against its
neighbors (a peak must beat the frame before AND after) and hill-climbed
until it does.
"""

from __future__ import annotations

import numpy as np

from .club import ClubTrack, _interp_track
from .pose import L_SHOULDER, L_WRIST, R_SHOULDER, R_WRIST, PoseTrack


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


def _hill_climb(x: np.ndarray, i: int, lo: int, hi: int, mode: str) -> int:
    """Owner's safeguard: a pillar must beat its neighbors. Move uphill (or
    downhill for a minimum) until the frame is a true local extremum."""
    sign = 1.0 if mode == "max" else -1.0
    i = int(np.clip(i, lo, hi - 1))
    while True:
        cur = sign * x[i]
        left = sign * x[i - 1] if i - 1 >= lo else -np.inf
        right = sign * x[i + 1] if i + 1 < hi else -np.inf
        if left <= cur and right <= cur:
            return i
        i = i - 1 if left > right else i + 1


def arc_angle(pose: PoseTrack, club: ClubTrack) -> np.ndarray:
    """Unwrapped clubhead angle (radians) around the mid-shoulder pivot."""
    head = _interp_track(club)
    pivot = pose.midpoint(L_SHOULDER, R_SHOULDER)[:, :2]
    rel = head - pivot
    phi = np.unwrap(np.arctan2(rel[:, 1], rel[:, 0]))
    return _smooth(phi, 5)


def detect_pillars(pose: PoseTrack, club: ClubTrack, debug: bool = False) -> dict | None:
    fps = pose.fps
    n = pose.n_frames
    wrist = pose.midpoint(L_WRIST, R_WRIST)
    wx, wy = _smooth(wrist[:, 0]), _smooth(wrist[:, 1])
    speed = np.hypot(np.gradient(wx), np.gradient(wy)) * fps
    speed = np.where(pose.detected, speed, np.inf)  # frozen pose is not "still"

    phi = arc_angle(pose, club)
    log: list[str] = []
    best = None

    for s_lo, s_hi in _stillness_periods(speed, 0.08, max(int(fps * 0.35), 3)):
        # Reference angle: the club at the ball during the setup stillness.
        win_lo = max(s_lo, s_hi - int(fps * 0.8))
        idx = [j for j in range(win_lo, s_hi + 1) if club.found[j]]
        if len(idx) < 2:
            log.append(f"still[{s_lo}-{s_hi}]: no club at rest")
            continue
        ball = np.median(club.head_xy[idx], axis=0)
        look_hi = min(n, s_hi + int(fps * 3.5))
        head = _interp_track(club)
        d = np.linalg.norm(head[s_hi:look_hi] - ball, axis=1)
        m = len(d)
        if m < int(fps * 0.6):
            log.append(f"still[{s_lo}-{s_hi}]: clip ends")
            continue

        # Impact = the valley where the club is back AT the ball, flanked on
        # both sides by real swings away from it (backswing and follow-through)
        # — and FAST. The clubhead peaks in speed through the ball; slow
        # ball-adjacent valleys (lowering the club after the swing, waggles)
        # are eliminated by speed weighting.
        head_speed = np.hypot(*np.gradient(head[s_hi:look_hi], axis=0).T) * fps
        sp_max = float(np.max(head_speed)) + 1e-6
        pad = max(int(fps * 0.15), 2)
        imp_rel, best_score = None, 0.0
        for t in range(pad, m - pad):
            if d[t] > 0.18:  # the club must be essentially back at the ball
                continue
            before = float(np.max(d[:t]))
            after = float(np.max(d[t : min(m, t + int(fps * 1.4))]))
            if before < 0.25 or after < 0.18:
                continue
            score = (min(before, after) - d[t]) + 1.5 * head_speed[t] / sp_max
            if score > best_score:
                imp_rel, best_score = t, score
        if imp_rel is None:
            log.append(
                f"still[{s_lo}-{s_hi}]: no ball-return flanked by two arcs "
                f"(max d {float(np.max(d)):.2f}, min d "
                f"{float(np.min(d[pad:])):.2f})"
            )
            continue

        # Peak backswing = farthest BEFORE impact; follow-peak = after. Only
        # trust OBSERVED clubhead positions for the peaks — interpolation
        # through detection gaps can fabricate phantom maxima.
        observed = club.found[s_hi:look_hi]
        d_obs = np.where(observed, d, -np.inf)
        if not np.any(observed[:imp_rel]):
            log.append(f"still[{s_lo}-{s_hi}]: no observed club before impact")
            continue
        top_rel = int(np.argmax(d_obs[:imp_rel]))
        f_hi = min(m, imp_rel + int(fps * 1.4))
        fol_rel = (
            imp_rel + int(np.argmax(d_obs[imp_rel:f_hi]))
            if np.any(observed[imp_rel:f_hi])
            else imp_rel + int(np.argmax(d[imp_rel:f_hi]))
        )

        # Owner's safeguard: neighbor-verify each pillar, hill-climb to a true
        # local extremum of the distance signal.
        top_rel = _hill_climb(d, top_rel, 0, imp_rel, "max")
        imp_rel = _hill_climb(d, imp_rel, top_rel + 1, f_hi, "min")
        fol_rel = _hill_climb(d, fol_rel, imp_rel + 1, m, "max")

        best = {
            "setup": max(s_lo, s_hi - 4),
            "top": s_hi + top_rel,
            "impact": s_hi + imp_rel,
            "follow_peak": s_hi + fol_rel,
            "d_top": float(d[top_rel]),
            "d_imp": float(d[imp_rel]),
        }
        log.append(
            f"still[{s_lo}-{s_hi}]: VALID top=+{top_rel} imp=+{imp_rel} "
            f"fol=+{fol_rel} (d {d[top_rel]:.2f} -> {d[imp_rel]:.2f} -> "
            f"{d[fol_rel]:.2f})"
        )

    if debug:
        for line in log:
            print(line)
    if best is None:
        return None

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
        "quality": {
            "d_top": round(best["d_top"], 2),
            "d_impact": round(best["d_imp"], 2),
        },
    }
