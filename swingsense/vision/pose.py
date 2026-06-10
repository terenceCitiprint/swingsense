"""Run MediaPipe pose over a video and return per-frame landmarks."""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass

import numpy as np

warnings.filterwarnings("ignore")
os.environ.setdefault("GLOG_minloglevel", "3")

# MediaPipe pose landmark indices we care about (33-point BlazePose topology).
NOSE = 0
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24


@dataclass
class PoseTrack:
    """Per-frame pose data extracted from a video."""

    landmarks: np.ndarray  # shape (n_frames, 33, 4): x, y, z, visibility (normalized)
    fps: float
    width: int
    height: int
    detected: np.ndarray  # bool per frame: was a pose found

    @property
    def n_frames(self) -> int:
        return self.landmarks.shape[0]

    def joint(self, idx: int) -> np.ndarray:
        """(n_frames, 4) track for one landmark."""
        return self.landmarks[:, idx, :]

    def midpoint(self, a: int, b: int) -> np.ndarray:
        """(n_frames, 4) midpoint track between two landmarks."""
        return (self.landmarks[:, a, :] + self.landmarks[:, b, :]) / 2.0


def extract_pose(video_path: str, model_variant: str = "full") -> PoseTrack:
    """Detect pose in every frame. Missing detections carry forward the last
    known pose so downstream feature code sees a continuous signal.

    Prefers the MediaPipe Tasks landmarker (downloaded once); if the model
    cannot be fetched (offline / blocked network), falls back to the legacy
    solutions API whose models ship inside the pip package.
    """
    try:
        return _extract_pose_tasks(video_path, model_variant)
    except Exception:
        return _extract_pose_legacy(video_path, model_variant)


def _extract_pose_legacy(video_path: str, model_variant: str) -> PoseTrack:
    """Offline path: mp.solutions.pose with bundled models."""
    import cv2
    import mediapipe as mp

    complexity = {"lite": 0, "full": 1, "heavy": 2}.get(model_variant, 1)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    landmarks = np.zeros((n, 33, 4), dtype=np.float32)
    detected = np.zeros(n, dtype=bool)
    last = np.zeros((33, 4), dtype=np.float32)

    with mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=complexity,
        smooth_landmarks=True,
    ) as pose:
        for i in range(n):
            ok, frame = cap.read()
            if not ok:
                landmarks[i] = last
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb)
            if result.pose_landmarks:
                arr = np.array(
                    [[p.x, p.y, p.z, p.visibility]
                     for p in result.pose_landmarks.landmark],
                    dtype=np.float32,
                )
                landmarks[i] = arr
                last = arr
                detected[i] = True
            else:
                landmarks[i] = last
    cap.release()
    return PoseTrack(landmarks=landmarks, fps=fps, width=width, height=height,
                     detected=detected)


def _extract_pose_tasks(video_path: str, model_variant: str) -> PoseTrack:
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    from .model import ensure_model

    model_path = str(ensure_model(model_variant))
    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO,
    )

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    landmarks = np.zeros((n, 33, 4), dtype=np.float32)
    detected = np.zeros(n, dtype=bool)
    last = np.zeros((33, 4), dtype=np.float32)

    # Fresh landmarker per call so timestamps stay monotonic across videos.
    with vision.PoseLandmarker.create_from_options(options) as lm:
        for i in range(n):
            ok, frame = cap.read()
            if not ok:
                landmarks[i] = last
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts_ms = int(i / fps * 1000)
            result = lm.detect_for_video(mp_img, ts_ms)
            if result.pose_landmarks:
                pts = result.pose_landmarks[0]
                arr = np.array(
                    [[p.x, p.y, p.z, p.visibility] for p in pts], dtype=np.float32
                )
                landmarks[i] = arr
                last = arr
                detected[i] = True
            else:
                landmarks[i] = last
    cap.release()

    return PoseTrack(
        landmarks=landmarks, fps=fps, width=width, height=height, detected=detected
    )
