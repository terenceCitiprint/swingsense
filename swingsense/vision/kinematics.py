"""Kinematics derived from a PoseTrack: joint speeds and segment angular velocities.

This is the data layer behind the visual report. Everything is computed in the
normalized 2D image plane, so speeds are in "frame-heights per second" — useful
for *shape and timing* comparisons (when does each segment peak, in what order),
not absolute m/s. The honest framing everywhere: timing is trustworthy,
magnitudes are proxies.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .features import _smooth, _wrap180
from .pose import (
    L_HIP,
    L_SHOULDER,
    L_WRIST,
    R_HIP,
    R_SHOULDER,
    R_WRIST,
    PoseTrack,
)


@dataclass
class Kinematics:
    """Per-frame derived motion signals (all numpy arrays of length n_frames)."""

    t: np.ndarray                 # time in seconds per frame
    joint_speed: np.ndarray       # (n_frames, 33) speed of every landmark
    hand_speed: np.ndarray        # midpoint-of-wrists speed
    hip_ang_vel: np.ndarray       # hip-line angular velocity, deg/s (abs)
    shoulder_ang_vel: np.ndarray  # shoulder-line angular velocity, deg/s (abs)
    arm_ang_vel: np.ndarray       # lead-arm (shoulder->wrist) angular velocity, deg/s (abs)

    def segment_peaks(self, lo: int, hi: int) -> dict:
        """Frame index of peak angular speed for each segment within [lo, hi]."""
        out = {}
        for name, sig in (
            ("pelvis", self.hip_ang_vel),
            ("thorax", self.shoulder_ang_vel),
            ("arm", self.arm_ang_vel),
        ):
            window = sig[lo : hi + 1]
            out[name] = lo + int(np.argmax(window)) if len(window) else lo
        out["hands"] = lo + int(np.argmax(self.hand_speed[lo : hi + 1])) if hi > lo else lo
        return out


def _series_angle_deg(track: PoseTrack, a: int, b: int) -> np.ndarray:
    """Per-frame angle (deg) of segment a->b vs horizontal, unwrapped."""
    pa = track.landmarks[:, a, :2]
    pb = track.landmarks[:, b, :2]
    ang = np.degrees(np.arctan2(pb[:, 1] - pa[:, 1], pb[:, 0] - pa[:, 0]))
    # Unwrap so frame-to-frame differences are not polluted by ±180 jumps.
    return np.degrees(np.unwrap(np.radians(ang)))


def _ang_vel(angles_deg: np.ndarray, fps: float) -> np.ndarray:
    v = np.abs(np.gradient(_smooth(angles_deg, 5))) * fps
    return _smooth(v, 5)


def compute_kinematics(track: PoseTrack) -> Kinematics:
    fps = track.fps
    n = track.n_frames
    t = np.arange(n) / fps

    # Per-landmark speed (normalized units / second), smoothed per joint.
    xy = track.landmarks[:, :, :2]
    d = np.zeros_like(xy)
    d[1:] = np.diff(xy, axis=0)
    speed = np.hypot(d[..., 0], d[..., 1]) * fps  # (n, 33)
    for j in range(speed.shape[1]):
        speed[:, j] = _smooth(speed[:, j], 5)

    wrist = track.midpoint(L_WRIST, R_WRIST)[:, :2]
    dw = np.zeros_like(wrist)
    dw[1:] = np.diff(wrist, axis=0)
    hand_speed = _smooth(np.hypot(dw[:, 0], dw[:, 1]) * fps, 5)

    hip_av = _ang_vel(_series_angle_deg(track, R_HIP, L_HIP), fps)
    sh_av = _ang_vel(_series_angle_deg(track, R_SHOULDER, L_SHOULDER), fps)
    # Lead arm proxy: left shoulder -> left wrist (right-handed golfer's lead
    # side). For a lefty this is the trail arm; timing shape is still useful.
    arm_av = _ang_vel(_series_angle_deg(track, L_SHOULDER, L_WRIST), fps)

    return Kinematics(
        t=t,
        joint_speed=speed,
        hand_speed=hand_speed,
        hip_ang_vel=hip_av,
        shoulder_ang_vel=sh_av,
        arm_ang_vel=arm_av,
    )


def separation_series(track: PoseTrack) -> np.ndarray:
    """Hip-shoulder separation proxy (deg) for every frame, wrapped to ±180."""
    sh = _series_angle_deg(track, R_SHOULDER, L_SHOULDER)
    hip = _series_angle_deg(track, R_HIP, L_HIP)
    raw = sh - hip
    return np.array([_wrap180(v) for v in raw])


def sequence_read(kin: Kinematics, top: int, impact: int) -> dict:
    """Order in which segments peak through the downswing + a verdict.

    Ideal proximal-to-distal order: pelvis -> thorax -> arm -> hands.
    """
    peaks = kin.segment_peaks(top, impact)
    order = sorted(peaks, key=peaks.get)
    ideal = ["pelvis", "thorax", "arm", "hands"]
    verdict = "in order" if order == ideal else "out of order"
    return {"peaks": peaks, "order": order, "ideal": ideal, "verdict": verdict}
