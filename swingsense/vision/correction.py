"""Generate a 'corrected' ghost skeleton by re-timing the player's own motion.

Principle: never invent positions. The ghost is built from the player's actual
recorded joints, with segment *timing* shifted toward the ideal kinematic
sequence (pelvis -> thorax -> arm -> hands). If the arms peak N frames before
the pelvis, the ghost replays the same arm trajectory delayed by N frames while
the lower body stays as recorded. What you see is your swing, re-sequenced —
an illustration of the timing fix, not a fabricated ideal body.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .kinematics import Kinematics
from .pose import PoseTrack

# Joint groups that get re-timed together (BlazePose indices).
_ARM_JOINTS = [13, 14, 15, 16]            # elbows + wrists
_THORAX_JOINTS = [11, 12]                 # shoulders
_MIN_GAP_S = 0.03                          # ideal min lead of each segment


@dataclass
class Ghost:
    landmarks: np.ndarray   # same shape as track.landmarks, re-timed copy
    changed: bool           # False if the sequence needed no correction
    description: str        # human-readable note of what was shifted
    shifts: dict            # segment -> frames delayed


def _delay_joints(dst: np.ndarray, src: np.ndarray, joints: list[int],
                  delay: int, lo: int, hi: int) -> None:
    """Within [lo, hi], replace dst's joints with src's same joints from
    `delay` frames earlier (clamped at lo so the ghost holds its position
    rather than borrowing pre-window motion)."""
    for f in range(lo, hi + 1):
        src_f = max(lo, f - delay)
        dst[f, joints, :] = src[src_f, joints, :]


def build_ghost(track: PoseTrack, kin: Kinematics, events: dict) -> Ghost:
    top = events["top"]["frame"]
    imp = events["impact"]["frame"]
    fps = track.fps
    min_gap = max(1, int(round(_MIN_GAP_S * fps)))

    peaks = kin.segment_peaks(top, imp)
    ghost = track.landmarks.copy()
    shifts: dict[str, int] = {}

    # Thorax should peak at least `min_gap` after the pelvis.
    thorax_delay = max(0, peaks["pelvis"] + min_gap - peaks["thorax"])
    # Arms should peak at least `min_gap` after the (corrected) thorax.
    corrected_thorax_peak = peaks["thorax"] + thorax_delay
    arm_delay = max(0, corrected_thorax_peak + min_gap - peaks["arm"])

    # Re-time within a window that starts a little before the top so the shift
    # blends in, and ends a little after impact.
    lo = max(0, top - int(fps * 0.2))
    hi = min(track.n_frames - 1, imp + int(fps * 0.2))

    if thorax_delay:
        _delay_joints(ghost, track.landmarks, _THORAX_JOINTS, thorax_delay, lo, hi)
        # Arms ride on the shoulders; they inherit at least the thorax delay.
        shifts["thorax"] = thorax_delay
    total_arm_delay = thorax_delay + arm_delay
    if total_arm_delay:
        _delay_joints(ghost, track.landmarks, _ARM_JOINTS, total_arm_delay, lo, hi)
        shifts["arm"] = total_arm_delay

    if not shifts:
        return Ghost(landmarks=ghost, changed=False,
                     description="Sequence already fires in order — no timing "
                                 "correction needed.", shifts={})

    parts = []
    if "thorax" in shifts:
        parts.append(f"thorax held {shifts['thorax']} frame(s) "
                     f"({shifts['thorax'] / fps * 1000:.0f} ms)")
    if "arm" in shifts:
        parts.append(f"arms held {shifts['arm']} frame(s) "
                     f"({shifts['arm'] / fps * 1000:.0f} ms)")
    desc = ("Ghost = your own motion with " + " and ".join(parts) +
            " so the pelvis leads. Same positions, fixed timing.")
    return Ghost(landmarks=ghost, changed=True, description=desc, shifts=shifts)
