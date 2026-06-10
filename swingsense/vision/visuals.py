"""Visual artifacts rendered from video + pose: the panoramic swing strip.

The strip is the report's hero: N stills sampled across the swing, each with
the skeleton drawn and colored by joint speed (cool = slow, hot = fast), plus
the hand-path arc traced continuously across the whole panorama.
"""

from __future__ import annotations

import numpy as np

from .kinematics import Kinematics
from .pose import L_WRIST, R_WRIST, PoseTrack

# Same readable BlazePose subset as overlay.py.
_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (24, 26), (26, 28),
]
_JOINTS = sorted({i for c in _CONNECTIONS for i in c})


def _speed_to_bgr(v: float) -> tuple[int, int, int]:
    """Map normalized speed [0,1] to a cool->hot BGR color.

    0.0 = steel blue (calm), 0.5 = amber, 1.0 = signal red-orange (fast).
    """
    v = float(np.clip(v, 0.0, 1.0))
    if v < 0.5:
        a = v / 0.5  # blue -> amber
        r, g, b = int(70 + a * (255 - 70)), int(110 + a * (170 - 110)), int(180 - a * 160)
    else:
        a = (v - 0.5) / 0.5  # amber -> red-orange
        r, g, b = 255, int(170 - a * 110), 20
    return (b, g, r)


def _draw_colored_skeleton(frame, row, speeds_norm, width, height, thickness=3):
    """Glow-style skeleton with velocity-colored bones and ringed joints."""
    import cv2
    import numpy as np

    def px(i):
        return int(row[i, 0] * width), int(row[i, 1] * height)

    glow = np.zeros_like(frame)
    segs = []
    for a, b in _CONNECTIONS:
        if row[a, 3] > 0.3 and row[b, 3] > 0.3:
            v = (speeds_norm[a] + speeds_norm[b]) / 2.0
            segs.append((px(a), px(b), _speed_to_bgr(v)))
    for p, q, c in segs:
        cv2.line(glow, p, q, c, thickness * 3, cv2.LINE_AA)
    glow = cv2.GaussianBlur(glow, (0, 0), 5)
    cv2.add(frame, (glow * 0.5).astype(frame.dtype), frame)
    for p, q, c in segs:
        cv2.line(frame, p, q, c, thickness, cv2.LINE_AA)
    for i in _JOINTS:
        if row[i, 3] > 0.3:
            cv2.circle(frame, px(i), 5, (25, 32, 28), -1, cv2.LINE_AA)
            cv2.circle(frame, px(i), 4, _speed_to_bgr(speeds_norm[i]), -1, cv2.LINE_AA)
            cv2.circle(frame, px(i), 4, (245, 245, 245), 1, cv2.LINE_AA)
    return frame


def _sample_frames(events: dict, n_panels: int, n_frames: int) -> list[int]:
    """Pick panel frames: dense through the downswing, sparser elsewhere."""
    a = events["address"]["frame"]
    top = events["top"]["frame"]
    imp = events["impact"]["frame"]
    fin = events.get("finish", {}).get("frame", min(n_frames - 1, imp + 1))

    back = np.linspace(a, top, max(2, n_panels // 3), endpoint=False)
    down = np.linspace(top, imp, max(3, n_panels // 2), endpoint=True)
    after = np.linspace(imp, fin, 2, endpoint=True)[1:]
    idx = sorted({int(round(i)) for i in (*back, *down, *after)})
    return [min(max(i, 0), n_frames - 1) for i in idx]


def panorama_strip(
    track: PoseTrack,
    video_path: str,
    events: dict,
    kin: Kinematics,
    out_path: str,
    n_panels: int = 8,
    panel_height: int = 480,
) -> str:
    """Write the panoramic strip PNG; returns out_path."""
    import cv2

    frames_idx = _sample_frames(events, n_panels, track.n_frames)

    # Normalize joint speeds over the swing window for stable coloring.
    a = events["address"]["frame"]
    imp = events["impact"]["frame"]
    window = kin.joint_speed[a : imp + 1]
    ref = float(np.percentile(window, 95)) if window.size else 1.0
    ref = max(ref, 1e-6)

    cap = cv2.VideoCapture(video_path)
    panels, panel_w = [], None
    labels = _event_labels(frames_idx, events)
    for k, fi in enumerate(frames_idx):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok:
            continue
        speeds_norm = np.clip(kin.joint_speed[fi] / ref, 0, 1)
        _draw_colored_skeleton(
            frame, track.landmarks[fi], speeds_norm, track.width, track.height
        )
        scale = panel_height / frame.shape[0]
        panel = cv2.resize(frame, (int(frame.shape[1] * scale), panel_height))
        if panel_w is None:
            panel_w = panel.shape[1]
        panel = cv2.resize(panel, (panel_w, panel_height))
        # Caption bar: time + event label if this panel sits on an event.
        bar = np.full((34, panel_w, 3), 18, dtype=np.uint8)
        txt = f"{fi / track.fps:.2f}s"
        if labels.get(fi):
            txt += f"  |  {labels[fi].upper()}"
        cv2.putText(bar, txt, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (235, 235, 235), 1, cv2.LINE_AA)
        panels.append(cv2.vconcat([panel, bar]))
    cap.release()

    if not panels:
        raise RuntimeError("Could not read any frames for the panorama.")

    strip = cv2.hconcat(panels)
    strip = _draw_hand_arc(strip, track, frames_idx, panel_w, panel_height)
    cv2.imwrite(out_path, strip)
    return out_path


def _event_labels(frames_idx: list[int], events: dict) -> dict:
    """Map sampled frame -> nearest event name (within 2 frames)."""
    out: dict[int, str] = {}
    for name in ("address", "top", "impact", "finish"):
        ev = events.get(name)
        if not isinstance(ev, dict):
            continue
        nearest = min(frames_idx, key=lambda fi: abs(fi - ev["frame"]))
        if abs(nearest - ev["frame"]) <= 2 and nearest not in out:
            out[nearest] = name
    return out


def _draw_hand_arc(strip, track: PoseTrack, frames_idx, panel_w, panel_h):
    """Trace the hand path continuously across the panorama panels."""
    import cv2

    wrist = track.midpoint(L_WRIST, R_WRIST)
    pts = []
    for k, fi in enumerate(frames_idx):
        x = int(k * panel_w + wrist[fi, 0] * panel_w)
        y = int(wrist[fi, 1] * panel_h)
        pts.append((x, y))
    for p, q in zip(pts, pts[1:]):
        cv2.line(strip, p, q, (40, 200, 255), 2, cv2.LINE_AA)
    for p in pts:
        cv2.circle(strip, p, 5, (40, 200, 255), 2, cv2.LINE_AA)
    return strip
