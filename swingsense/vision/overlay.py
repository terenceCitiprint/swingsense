"""Draw the detected pose skeleton onto frames — for visual sanity-checking.

MediaPipe's Tasks build here ships without the legacy drawing utilities, so we
draw the BlazePose connections ourselves with OpenCV.
"""

from __future__ import annotations

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
    """Write a side-by-side image of the skeleton at address/top/impact/finish."""
    import cv2

    panels = []
    cap = cv2.VideoCapture(video_path)
    for name in (
        "address",
        "top",
        "transition",
        "impact",
        "follow_through",
        "finish",
    ):
        ev = events.get(name)
        if not isinstance(ev, dict):
            continue
        fr_idx = ev["frame"]
        cap.set(cv2.CAP_PROP_POS_FRAMES, fr_idx)
        ok, frame = cap.read()
        if not ok:
            continue
        draw_skeleton(frame, track.landmarks[fr_idx], track.width, track.height)
        cv2.putText(
            frame, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2
        )
        panels.append(cv2.resize(frame, (240, 426)))
    cap.release()
    if panels:
        cv2.imwrite(out_path, cv2.hconcat(panels))
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
