"""Draw the detected pose skeleton onto frames — for visual sanity-checking.

MediaPipe's Tasks build here ships without the legacy drawing utilities, so we
draw the BlazePose connections ourselves with OpenCV.
"""

from __future__ import annotations

import numpy as np

from .pose import PoseTrack

# A readable subset of the 33-point BlazePose skeleton (arms, torso, legs).
_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),  # shoulders + arms
    (11, 23), (12, 24), (23, 24),  # torso
    (23, 25), (25, 27), (24, 26), (26, 28),  # legs
]


def draw_skeleton(frame, row, width: int, height: int):
    """Draw one frame's landmarks (a (33,4) row) onto a BGR frame in place."""
    import cv2

    def px(i):
        return int(row[i, 0] * width), int(row[i, 1] * height)

    for a, b in _CONNECTIONS:
        if row[a, 3] > 0.3 and row[b, 3] > 0.3:
            cv2.line(frame, px(a), px(b), (0, 255, 0), 2)
    for i in range(33):
        if row[i, 3] > 0.3:
            cv2.circle(frame, px(i), 3, (0, 200, 255), -1)
    return frame


def event_montage(track: PoseTrack, video_path: str, events: dict, out_path: str):
    """Write a 6-panel storyboard of the swing.

    Frame selection is anchored on the three most reliably-detected positions —
    peak backswing, impact, and finish — and the in-between panels are derived
    from them so the whole swing is represented. This deliberately avoids the
    failure mode where the panels bunch up in the through-swing (top and
    transition one frame apart, impact/follow/finish clustered) and the
    backswing is never actually shown.
    """
    import cv2

    a = events.get("address", {}).get("frame", 0)
    top = events.get("top", {}).get("frame", 0)
    imp = events.get("impact", {}).get("frame", top)
    fin = events.get("finish", {}).get("frame", imp)

    if events.get("captured") == "backswing_only":
        # Only the backswing exists — spread the panels across it so the climb
        # to the top is fully shown, rather than padding with bogus through-swing.
        pts = np.linspace(a, top, 6).astype(int)
        plan = list(
            zip(
                pts,
                ["address", "takeaway", "early-bsw", "mid-bsw", "late-bsw", "top (peak)"],
            )
        )
    else:
        # Anchors: top (peak backswing), impact, finish. Derived: a mid-backswing
        # frame (so the climb is visible) and a mid-follow-through frame.
        mid_bsw = events.get("mid_backswing", {}).get("frame", (a + top) // 2)
        mid_fol = (imp + fin) // 2
        plan = [
            (a, "address"),
            (mid_bsw, "mid-backswing"),
            (top, "top (peak)"),
            (imp, "impact"),
            (mid_fol, "follow-through"),
            (fin, "finish"),
        ]

    panels = []
    cap = cv2.VideoCapture(video_path)
    for fr_idx, label in plan:
        fr_idx = int(max(0, min(track.n_frames - 1, fr_idx)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, fr_idx)
        ok, frame = cap.read()
        if not ok:
            continue
        draw_skeleton(frame, track.landmarks[fr_idx], track.width, track.height)
        cv2.putText(
            frame, f"{label} ({fr_idx})", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
        )
        panels.append(cv2.resize(frame, (240, 426)))
    cap.release()
    if panels:
        cv2.imwrite(out_path, cv2.hconcat(panels))
    return out_path


def pillar_montage(track: PoseTrack, video_path: str, events: dict, out_path: str):
    """The owner's pillar protocol, as one stitch:

      1. three frames just before movement starts (club head at rest)
      2. top of the backswing
      3. impact
      4. peak of the follow-through
      5. two frames after, to check the player's balance

    These pillars are the foundation; in-between frames are only analyzed once
    the pillars are verified. 8 panels total.
    """
    import cv2

    fps = track.fps
    a = events.get("address", {}).get("frame", 0)
    setup = events.get("setup", {}).get("frame", a - 4)  # one still, ~4 before movement
    top = events.get("top", {}).get("frame", 0)
    imp = events.get("impact", {}).get("frame", top)
    fol = events.get("follow_through", events.get("follow_peak", {})).get("frame", imp)
    fin = events.get("finish", {}).get("frame", fol)

    bal_gap = max(int(fps * 0.35), 3)  # spacing of the balance-check frames
    plan = [
        (setup, "SETUP (club at ball)"),
        (top, "PEAK backswing (farthest from ball)"),
        (imp, "IMPACT (club back at ball)"),
        (fol, "FOLLOW-THROUGH"),
        (fin + bal_gap, "balance +1"),
        (fin + 2 * bal_gap, "balance +2"),
    ]

    panels = []
    cap = cv2.VideoCapture(video_path)
    for fr_idx, label in plan:
        fr_idx = int(max(0, min(track.n_frames - 1, fr_idx)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, fr_idx)
        ok, frame = cap.read()
        if not ok:
            continue
        draw_skeleton(frame, track.landmarks[fr_idx], track.width, track.height)
        cv2.putText(
            frame, f"{label} ({fr_idx})", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
        )
        panels.append(cv2.resize(frame, (220, 391)))
    if panels:
        cv2.imwrite(out_path, cv2.hconcat(panels))
    cap.release()
    return out_path


def overlay_video(track: PoseTrack, video_path: str, out_path: str):
    """Write a full copy of the video with the skeleton drawn on every frame."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        out_path, fourcc, track.fps, (track.width, track.height)
    )
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i < track.n_frames:
            draw_skeleton(frame, track.landmarks[i], track.width, track.height)
        writer.write(frame)
        i += 1
    cap.release()
    writer.release()
    return out_path
