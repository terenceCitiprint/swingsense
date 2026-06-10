"""Derive swing events and biomechanics features from a PoseTrack.

Everything here is 2D image-plane geometry from a single camera, so the numbers
are *proxies*, not lab-grade measurements. Each result carries notes and an
overall confidence the reasoning engine is told to respect.

Coordinate note: MediaPipe y increases DOWNWARD, so a smaller y means the joint
is higher in the frame.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .pose import (
    L_HIP,
    L_SHOULDER,
    L_WRIST,
    NOSE,
    R_HIP,
    R_SHOULDER,
    R_WRIST,
    PoseTrack,
)


@dataclass
class SwingFeatures:
    events: dict  # frame index + time for address/top/impact/finish
    metrics: dict  # named numeric features (tempo, angles, sway...)
    notes: list[str] = field(default_factory=list)
    confidence: str = "low"

    def to_dict(self) -> dict:
        return {
            "events": self.events,
            "metrics": self.metrics,
            "notes": self.notes,
            "confidence": self.confidence,
        }


def _line_angle_deg(track: PoseTrack, frame: int, a: int, b: int) -> float:
    """Angle of the segment a→b vs horizontal, in degrees (image plane)."""
    pa, pb = track.landmarks[frame, a], track.landmarks[frame, b]
    dx, dy = pb[0] - pa[0], pb[1] - pa[1]
    return math.degrees(math.atan2(dy, dx))


def _wrap180(deg: float) -> float:
    """Fold an angle difference into (-180, 180] so atan2 wraparound near ±180
    can't produce nonsense like a -352° 'separation'."""
    return (deg + 180.0) % 360.0 - 180.0


def _window_visibility(track: PoseTrack, lo: int, hi: int, joints: list[int]) -> float:
    """Mean MediaPipe visibility of the given joints over [lo, hi]."""
    seg = track.landmarks[lo : hi + 1][:, joints, 3]
    return float(seg.mean()) if seg.size else 0.0


def _smooth(x: np.ndarray, k: int = 5) -> np.ndarray:
    if len(x) < k:
        return x
    kernel = np.ones(k) / k
    return np.convolve(x, kernel, mode="same")


def detect_events(track: PoseTrack) -> dict:
    """Locate address, top of backswing, and impact from the hand-path trace.

    Strategy: the wrists trace a big vertical arc. The top of the backswing is
    the highest hand position (min y). Address is the lowest stable hand
    position before that; impact is the lowest hand position just after it
    (before the hands rise again into the finish).
    """
    wrist = track.midpoint(L_WRIST, R_WRIST)
    y = _smooth(wrist[:, 1])  # vertical hand path
    n = len(y)

    # Top of backswing = highest hands overall (smallest y).
    top = int(np.argmin(y))

    # Takeaway/address: lowest hands (largest y) shortly before the top. Keep the
    # window tight (~1.2s) so a long static setup or waggle isn't counted as
    # backswing time and doesn't inflate the tempo ratio.
    addr_lo = max(0, top - int(track.fps * 1.2))
    address = addr_lo + int(np.argmax(y[addr_lo:top])) if top > addr_lo else 0

    # Impact: physically, the hands return to roughly their address height as the
    # club comes back to the ball. Find the first frame after the top where the
    # hand path descends back through the address level. Fall back to peak hand
    # speed if that crossing never happens (e.g. a clipped follow-through).
    imp_hi = min(n, top + int(track.fps * 0.8))
    address_y = y[address]
    impact = None
    for j in range(top + 1, imp_hi):
        if y[j] >= address_y:
            impact = j
            break
    if impact is None:
        wx = _smooth(wrist[:, 0])
        if imp_hi > top + 1:
            speed = np.hypot(np.diff(wx[top:imp_hi]), np.diff(y[top:imp_hi]))
            impact = top + 1 + int(np.argmax(speed))
        else:
            impact = min(top + 1, n - 1)

    finish = min(n - 1, impact + int(track.fps * 1.0))

    def t(i: int) -> float:
        return round(i / track.fps, 3)

    return {
        "address": {"frame": address, "t": t(address)},
        "top": {"frame": top, "t": t(top)},
        "impact": {"frame": impact, "t": t(impact)},
        "finish": {"frame": finish, "t": t(finish)},
    }


def compute_features(track: PoseTrack) -> SwingFeatures:
    events = detect_events(track)
    a = events["address"]["frame"]
    top = events["top"]["frame"]
    imp = events["impact"]["frame"]

    notes: list[str] = []
    metrics: dict = {}

    # --- Tempo: backswing vs downswing duration (classic ~3:1). ---
    backswing_f = max(top - a, 1)
    downswing_f = max(imp - top, 1)
    ratio = backswing_f / downswing_f
    # A real downswing needs >=5 frames to localize and a sane swing sits
    # roughly between 1.5:1 and 5:1; outside that the timing is not trustworthy.
    tempo_reliable = downswing_f >= 5 and 1.0 <= ratio <= 5.0
    metrics["tempo"] = {
        "backswing_s": round(backswing_f / track.fps, 3),
        "downswing_s": round(downswing_f / track.fps, 3),
        "ratio_back_to_down": round(ratio, 2),
        "downswing_frames": downswing_f,
        "reliable": tempo_reliable,
    }
    if not tempo_reliable:
        notes.append(
            f"Downswing spans only {downswing_f} frames at {track.fps:.0f}fps "
            f"(ratio {ratio:.1f}:1) — impact/tempo are below this camera's time "
            "resolution. Treat tempo as approximate; 120-240fps would fix this."
        )

    # --- Rotation proxies: shoulder & hip line angles at each event. ---
    rot = {}
    for label, fr in (("address", a), ("top", top), ("impact", imp)):
        sh = _line_angle_deg(track, fr, R_SHOULDER, L_SHOULDER)
        hip = _line_angle_deg(track, fr, R_HIP, L_HIP)
        rot[label] = {
            "shoulder_line_deg": round(sh, 1),
            "hip_line_deg": round(hip, 1),
            "separation_deg": round(_wrap180(sh - hip), 1),
        }
    metrics["rotation"] = rot
    notes.append(
        "Shoulder/hip angles are 2D image-plane tilts, not true 3D turn; "
        "separation is an X-factor *proxy* only."
    )

    # --- Head stability: how much the nose drifts during the swing. ---
    nose = track.joint(NOSE)[a : imp + 1]
    if len(nose):
        sway = float(np.ptp(nose[:, 0]))  # horizontal range, normalized
        lift = float(np.ptp(nose[:, 1]))  # vertical range, normalized
        metrics["head_movement"] = {
            "lateral_range_pct": round(sway * 100, 1),
            "vertical_range_pct": round(lift * 100, 1),
        }

    # --- Hand-path vertical excursion (rough swing "size"). ---
    wrist = track.midpoint(L_WRIST, R_WRIST)
    metrics["hand_path"] = {
        "vertical_excursion_pct": round(float(np.ptp(wrist[a : imp + 1, 1])) * 100, 1)
    }

    # --- Detection quality over the swing window. ---
    window = track.detected[a : imp + 1]
    det_rate = float(window.mean()) if len(window) else 0.0
    # A body being *found* is not enough — the swing read lives or dies on how
    # well the HANDS specifically are tracked. Measure key-landmark visibility.
    hand_vis = _window_visibility(track, a, imp, [L_WRIST, R_WRIST])
    core_vis = _window_visibility(
        track, a, imp, [L_SHOULDER, R_SHOULDER, L_HIP, R_HIP]
    )
    metrics["detection_rate"] = round(det_rate, 2)
    metrics["hand_visibility"] = round(hand_vis, 2)
    metrics["core_visibility"] = round(core_vis, 2)

    # A real backswing takes time; if address and top collapse together the
    # event detector failed (usually because the hands vanished during the
    # upswing), no matter how it scored on raw visibility.
    backswing_implausible = backswing_f < 4

    # --- Overall confidence: gated on hand tracking AND event plausibility. ---
    if hand_vis < 0.3 or backswing_implausible:
        confidence = "low"
        events["reliable"] = False
        if backswing_implausible:
            notes.append(
                f"Backswing resolved to only {backswing_f} frame(s) — the "
                "top-of-swing could not be located (hands likely lost during "
                "the upswing). Tempo and event timing are UNRELIABLE here."
            )
        if hand_vis < 0.3:
            notes.append(
                f"Hands are poorly tracked (wrist visibility {hand_vis:.2f}). "
                "Reframe so the body fills more of the frame, against a plain "
                "background, ideally at 120-240fps."
            )
    elif hand_vis < 0.6 or core_vis < 0.6 or downswing_f < 5:
        confidence = "medium"
        events["reliable"] = True
    else:
        confidence = "high"
        events["reliable"] = True

    return SwingFeatures(
        events=events, metrics=metrics, notes=notes, confidence=confidence
    )
