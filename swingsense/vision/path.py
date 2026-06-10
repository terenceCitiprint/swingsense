"""Hand-path and swing-plane analysis.

The hands (wrist midpoint) trace the most informative path a single camera can
follow — the club itself is NOT tracked, so this is hand path, never club path.
The downswing portion is fit with a least-squares plane; how tilted that plane
is, how tightly the path hugs it, and how it differs from the backswing plane
are the classic "plane analysis" reads.

Honesty: the depth axis comes from MediaPipe's z estimate, its least accurate
output. Everything here is therefore *illustrative shape analysis*; for real
plane work, film down-the-line and treat these numbers as proxies.
"""

from __future__ import annotations

import numpy as np

from .pose import L_WRIST, R_WRIST, PoseTrack


def _render_xyz(track: PoseTrack) -> np.ndarray:
    """Wrist-midpoint path in render coordinates (x, depth, up)."""
    wr = (track.landmarks[:, L_WRIST, :] + track.landmarks[:, R_WRIST, :]) / 2.0
    return np.stack([wr[:, 0], wr[:, 2], 1.0 - wr[:, 1]], axis=1)


def _fit_plane(pts: np.ndarray):
    """Least-squares plane. Returns centroid, unit normal, in-plane basis."""
    c = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - c)
    n = vt[2] / (np.linalg.norm(vt[2]) + 1e-12)
    if n[2] < 0:
        n = -n
    return c, n, vt[0], vt[1]


def analyze_path(track: PoseTrack, events: dict) -> dict:
    """Split the hand path by swing phase and fit the downswing plane.

    Returns point arrays for rendering plus a `plane` metrics dict (or None
    when the downswing is too short to fit).
    """
    xyz = _render_xyz(track)
    a = events["address"]["frame"]
    top = events["top"]["frame"]
    imp = events["impact"]["frame"]
    fin = events.get("finish", {}).get("frame", len(xyz) - 1)

    out: dict = {
        "back_pts": xyz[a:top + 1],
        "down_pts": xyz[top:imp + 1],
        "follow_pts": xyz[imp:fin + 1],
        "plane": None,
        "notes": [
            "Hand path, not club path — the club is not tracked.",
            "Depth uses MediaPipe's z estimate (its weakest axis): treat plane "
            "numbers as illustrative shape proxies. Film down-the-line for "
            "real plane work.",
        ],
    }

    down = out["down_pts"]
    if len(down) < 4:
        return out

    c, n, e1, e2 = _fit_plane(down)
    # Tilt of the fitted plane from horizontal = angle between its normal and
    # vertical. Steep swings -> large tilt.
    tilt = float(np.degrees(np.arccos(np.clip(abs(n[2]), 0.0, 1.0))))
    d = (down - c) @ n
    rms = float(np.sqrt((d ** 2).mean()))

    plane = {
        "tilt_deg": round(tilt, 1),
        "rms_offplane_pct": round(rms * 100, 1),  # % of frame height
        "back_vs_down_deg": None,
    }

    back = out["back_pts"]
    if len(back) >= 4:
        _, nb, _, _ = _fit_plane(back)
        delta = float(np.degrees(np.arccos(np.clip(abs(nb @ n), 0.0, 1.0))))
        plane["back_vs_down_deg"] = round(delta, 1)

    # Quad for rendering the fitted plane, sized to the downswing extent.
    proj1 = (down - c) @ e1
    proj2 = (down - c) @ e2
    s1 = 1.25 * max(abs(proj1.max()), abs(proj1.min()), 0.05)
    s2 = 1.25 * max(abs(proj2.max()), abs(proj2.min()), 0.05)
    out["plane_quad"] = np.array([
        c - e1 * s1 - e2 * s2,
        c + e1 * s1 - e2 * s2,
        c + e1 * s1 + e2 * s2,
        c - e1 * s1 + e2 * s2,
    ])
    out["plane"] = plane
    return out
