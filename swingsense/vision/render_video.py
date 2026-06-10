"""Pro-grade annotated swing video: glow skeleton, comet hand path, event
freeze-frames with callout labels, slow-motion downswing, metric HUD.

This draws on the REAL footage (the visual language of commercial swing apps)
rather than an abstract widget. Output is a trimmed mp4 that starts shortly
before address — long static setups are cut — plus a 'trace card' still:
the impact frame with the full hand path and ghosted key positions.
"""

from __future__ import annotations

import numpy as np

from .features import _smooth
from .pose import L_WRIST, R_WRIST, PoseTrack

# BGR palette
_BONE = (215, 222, 228)        # warm white bones
_JOINT = (12, 89, 232)         # signal orange #E8590C
_JOINT_RING = (245, 245, 245)
_TRAIL_BACK = (69, 200, 255)   # gold
_TRAIL_DOWN = (34, 87, 255)    # hot orange-red
_TRAIL_FOLLOW = (160, 163, 154)
_HUD_TEXT = (235, 238, 235)
_ACCENT = (70, 107, 46)        # fairway green (BGR)

_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (24, 26), (26, 28),
]
_JOINTS = sorted({i for c in _CONNECTIONS for i in c})

_CALLOUTS = [  # (label, anchor joints)
    ("Head", [0]),
    ("Upper torso", [11, 12]),
    ("Pelvis", [23, 24]),
    ("Stance", [27, 28]),
]


def smooth_pixels(track: PoseTrack) -> np.ndarray:
    """(n, 33, 2) pixel coordinates, temporally smoothed per joint."""
    n = track.n_frames
    px = np.empty((n, 33, 2), dtype=np.float32)
    for j in range(33):
        px[:, j, 0] = _smooth(track.landmarks[:, j, 0], 5) * track.width
        px[:, j, 1] = _smooth(track.landmarks[:, j, 1], 5) * track.height
    return px


def draw_skeleton(frame, pts, scale: float = 1.0, joint_color=_JOINT):
    """Glow skeleton: soft halo layer + crisp core bones + ringed joints."""
    import cv2

    glow = np.zeros_like(frame)
    w_glow = max(6, int(10 * scale))
    w_core = max(2, int(4 * scale))
    segs = [(tuple(pts[a].astype(int)), tuple(pts[b].astype(int)))
            for a, b in _CONNECTIONS]
    for p, q in segs:
        cv2.line(glow, p, q, _BONE, w_glow, cv2.LINE_AA)
    glow = cv2.GaussianBlur(glow, (0, 0), 6 * scale)
    cv2.add(frame, (glow * 0.55).astype(frame.dtype), frame)
    for p, q in segs:
        cv2.line(frame, p, q, _BONE, w_core, cv2.LINE_AA)
    r = max(4, int(6 * scale))
    for j in _JOINTS + [0]:
        c = tuple(pts[j].astype(int))
        cv2.circle(frame, c, r + 2, (25, 32, 28), -1, cv2.LINE_AA)
        cv2.circle(frame, c, r, joint_color, -1, cv2.LINE_AA)
        cv2.circle(frame, c, r, _JOINT_RING, 1, cv2.LINE_AA)
    return frame


def draw_trail(frame, pts_xy, colors, scale: float = 1.0, max_pts: int = 9999):
    """Fading comet trail. pts_xy: (k,2) pixel points oldest->newest;
    colors: per-point BGR."""
    import cv2

    k = len(pts_xy)
    if k < 2:
        return frame
    start = max(0, k - max_pts)
    overlay = frame.copy()
    for i in range(start + 1, k):
        age = (i - start) / max(k - start, 1)          # 0 old -> 1 new
        w = max(2, int((2 + 5 * age) * scale))
        p = tuple(np.asarray(pts_xy[i - 1]).astype(int))
        q = tuple(np.asarray(pts_xy[i]).astype(int))
        cv2.line(overlay, p, q, colors[i], w, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    head = tuple(np.asarray(pts_xy[-1]).astype(int))
    cv2.circle(frame, head, max(4, int(6 * scale)), colors[-1], -1, cv2.LINE_AA)
    cv2.circle(frame, head, max(4, int(6 * scale)), (255, 255, 255), 1,
               cv2.LINE_AA)
    return frame


def _text(frame, s, org, scale, color=_HUD_TEXT, weight=1):
    import cv2
    cv2.putText(frame, s, (org[0] + 1, org[1] + 2), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (15, 20, 16), weight + 2, cv2.LINE_AA)
    cv2.putText(frame, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color,
                weight, cv2.LINE_AA)


def _hud(frame, left: str, right: str, scale: float):
    """Bottom translucent bar with left/right text."""
    import cv2
    h, w = frame.shape[:2]
    bar_h = int(44 * scale)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - bar_h), (w, h), (12, 16, 13), -1)
    cv2.rectangle(overlay, (0, h - bar_h), (w, h - bar_h + max(2, int(3 * scale))),
                  _ACCENT, -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    _text(frame, left, (int(12 * scale), h - int(14 * scale)), 0.55 * scale)
    (tw, _), _ = cv2.getTextSize(right, cv2.FONT_HERSHEY_SIMPLEX,
                                 0.55 * scale, 1)
    _text(frame, right, (w - tw - int(12 * scale), h - int(14 * scale)),
          0.55 * scale)
    return frame


def _callout_card(frame, pts, title: str, subtitle: str, scale: float):
    """Event freeze-frame: dim, big title, joint-group callout labels with
    leader lines (the labeled look of commercial swing apps)."""
    import cv2
    h, w = frame.shape[:2]
    dim = (frame * 0.78).astype(frame.dtype)
    frame[:] = dim
    _text(frame, title, (int(16 * scale), int(52 * scale)), 1.25 * scale,
          (255, 255, 255), 2)
    if subtitle:
        _text(frame, subtitle, (int(16 * scale), int(86 * scale)),
              0.62 * scale, (200, 230, 205))
    # Callouts down the right edge.
    x_text = w - int(150 * scale)
    y = int(140 * scale)
    for label, joints in _CALLOUTS:
        anchor = pts[joints].mean(axis=0).astype(int)
        cv2.circle(frame, tuple(anchor), max(4, int(5 * scale)),
                   (230, 230, 230), -1, cv2.LINE_AA)
        cv2.line(frame, tuple(anchor), (x_text - int(8 * scale), y - 5),
                 (210, 210, 210), 1, cv2.LINE_AA)
        _text(frame, label, (x_text, y), 0.52 * scale)
        y += int(34 * scale)
    return frame


def render_swing_video(
    track: PoseTrack,
    video_path: str,
    events: dict,
    out_path: str,
    pad_s: float = 1.2,
    slow_downswing: int = 3,
    hold_s: float = 0.9,
    tempo_ratio: float | None = None,
) -> str:
    """Write the annotated mp4. Returns out_path."""
    import cv2

    a = events["address"]["frame"]
    top = events["top"]["frame"]
    imp = events["impact"]["frame"]
    fin = events.get("finish", {}).get("frame", track.n_frames - 1)
    fps = track.fps

    lo = max(0, a - int(fps * pad_s))
    hi = min(track.n_frames - 1, fin + int(fps * 0.4))

    px = smooth_pixels(track)
    wrist_px = (px[:, L_WRIST] + px[:, R_WRIST]) / 2.0
    scale = track.height / 850.0
    hold_n = int(fps * hold_s)

    def phase_of(fi):
        if fi < a:
            return "setup"
        if fi <= top:
            return "backswing"
        if fi <= imp:
            return "downswing"
        return "follow-through"

    trail_pts, trail_cols = [], []

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, lo)
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (track.width, track.height))
    events_at = {top: ("TOP", "backswing complete"),
                 imp: ("IMPACT",
                       f"tempo {tempo_ratio}:1" if tempo_ratio else "")}

    for fi in range(lo, hi + 1):
        ok, frame = cap.read()
        if not ok:
            break
        if fi >= a:
            trail_pts.append(wrist_px[fi])
            if fi <= top:
                trail_cols.append(_TRAIL_BACK)
            elif fi <= imp:
                trail_cols.append(_TRAIL_DOWN)
            else:
                trail_cols.append(_TRAIL_FOLLOW)
        draw_trail(frame, trail_pts, trail_cols, scale)
        draw_skeleton(frame, px[fi], scale)
        phase = phase_of(fi)
        _hud(frame, f"SwingSense  |  {phase}",
             f"{(fi - a) / fps:+.2f}s", scale)

        repeats = slow_downswing if (top < fi <= imp) else 1
        if repeats > 1:
            _text(frame, f"{repeats}x slow", (int(12 * scale), int(30 * scale)),
                  0.55 * scale, (200, 200, 200))
        for _ in range(repeats):
            writer.write(frame)

        if fi in events_at:
            title, sub = events_at[fi]
            card = frame.copy()
            _callout_card(card, px[fi], title, sub, scale)
            for _ in range(hold_n):
                writer.write(card)

    cap.release()
    writer.release()
    return out_path


def render_trace_card(
    track: PoseTrack,
    video_path: str,
    events: dict,
    out_path: str,
) -> str:
    """Still: impact frame + full hand path + ghosted address/top skeletons."""
    import cv2

    a = events["address"]["frame"]
    top = events["top"]["frame"]
    imp = events["impact"]["frame"]
    px = smooth_pixels(track)
    wrist_px = (px[:, L_WRIST] + px[:, R_WRIST]) / 2.0
    scale = track.height / 850.0

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, imp)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("Could not read impact frame.")

    # Ghosted earlier positions: thin bones only (no joints) so the impact
    # skeleton stays the unambiguous subject.
    def ghost_bones(fi, alpha):
        layer = frame.copy()
        for a_, b_ in _CONNECTIONS:
            p = tuple(px[fi, a_].astype(int))
            q = tuple(px[fi, b_].astype(int))
            cv2.line(layer, p, q, (235, 240, 242), max(2, int(2 * scale)),
                     cv2.LINE_AA)
        cv2.addWeighted(layer, alpha, frame, 1 - alpha, 0, frame)

    ghost_bones(a, 0.35)
    ghost_bones(top, 0.45)
    pts = [wrist_px[i] for i in range(a, imp + 1)]
    cols = [(_TRAIL_BACK if i <= top else _TRAIL_DOWN)
            for i in range(a, imp + 1)]
    draw_trail(frame, pts, cols, scale)
    draw_skeleton(frame, px[imp], scale)
    _text(frame, "hand path", (int(14 * scale), int(40 * scale)),
          0.8 * scale, (255, 255, 255), 2)
    _text(frame, "gold backswing / orange downswing",
          (int(14 * scale), int(68 * scale)), 0.5 * scale, (220, 224, 220))
    _hud(frame, "SwingSense", "address / top / impact", scale)
    cv2.imwrite(out_path, frame)
    return out_path
