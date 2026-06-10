"""Smoke tests: synthetic swings with known ground truth through the pipeline.

Two parametric swings are generated — one with the hips leading (in-order
kinematic sequence) and one arm-led (out of order). The tests assert that
event detection, sequence reading, the correction ghost, the report, and the
comparison all behave correctly on them. No real video or API key needed;
vision deps (numpy/opencv/plotly) are required, mediapipe is not.

Run: pytest tests/ -q
"""

from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
pytest.importorskip("plotly")

from swingsense.vision.features import compute_features
from swingsense.vision.kinematics import compute_kinematics, sequence_read
from swingsense.vision.correction import build_ghost
from swingsense.vision.pose import PoseTrack

FPS, W, H = 30.0, 540, 960
HOLD, BACK, DOWN, FOLLOW = 12, 33, 11, 30
N = HOLD + BACK + DOWN + FOLLOW
TRUE_TOP = HOLD + BACK
TRUE_IMPACT = HOLD + BACK + DOWN


def _lerp(a, b, t):
    return a + (b - a) * t


def _ease(t):
    return 0.5 - 0.5 * np.cos(np.pi * np.clip(t, 0, 1))


def make_track(arm_led: bool) -> PoseTrack:
    def phase_at(i, lag=0):
        j = i - lag
        if j < HOLD:
            return 0.0
        if j < HOLD + BACK:
            return _ease((j - HOLD) / BACK)
        if j < HOLD + BACK + DOWN:
            return _lerp(1.0, -0.05, _ease((j - HOLD - BACK) / DOWN))
        return _lerp(-0.05, -0.6, _ease((j - HOLD - BACK - DOWN) / FOLLOW))

    lms = np.zeros((N, 33, 4), dtype=np.float32)
    lms[:, :, 3] = 0.95
    cx, hip_y, sh_y = 0.5, 0.62, 0.42
    hip_lag = 4 if arm_led else 0
    hip_lead = 0 if arm_led else 2

    for i in range(N):
        p_body = phase_at(i)
        if i >= HOLD + BACK:
            p_hip = phase_at(i, lag=hip_lag) if arm_led else phase_at(i + hip_lead)
        else:
            p_hip = p_body

        def pair(y, half_w, rot_deg, l_i, r_i):
            r = np.radians(rot_deg)
            dx, dy = half_w * np.cos(r), half_w * np.sin(r)
            lms[i, l_i, :2] = (cx + dx, y + dy)
            lms[i, r_i, :2] = (cx - dx, y - dy)

        pair(hip_y, 0.07, 12 * p_hip, 23, 24)
        pair(sh_y, 0.10, 28 * p_body, 11, 12)

        ang = np.radians(_lerp(95, -75, (p_body + 0.6) / 1.6))
        arm = 0.26
        hx, hy = cx + arm * np.cos(ang) * 0.9, sh_y + arm * np.sin(ang)
        for wi in (15, 16):
            lms[i, wi, :2] = (hx, hy)
        for sh_i, el_i, wr_i in ((11, 13, 15), (12, 14, 16)):
            lms[i, el_i, :2] = (lms[i, sh_i, :2] + lms[i, wr_i, :2]) / 2 + (0, 0.015)
        for hip_i, kn_i, an_i in ((23, 25, 27), (24, 26, 28)):
            lms[i, kn_i, :2] = lms[i, hip_i, :2] + (0, 0.14)
            lms[i, an_i, :2] = lms[i, hip_i, :2] + (0, 0.27)
        lms[i, 0, :2] = (cx, sh_y - 0.10)
        lms[i, (15, 16), 2] = -0.18 * np.sin(
            np.radians(_lerp(0, 160, (p_body + 0.6) / 1.6)))
    return PoseTrack(landmarks=lms, fps=FPS, width=W, height=H,
                     detected=np.ones(N, dtype=bool))


def write_video(track: PoseTrack, path: str) -> str:
    wr = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    conns = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23),
             (12, 24), (23, 24), (23, 25), (25, 27), (24, 26), (26, 28)]
    for i in range(track.n_frames):
        f = np.full((H, W, 3), (44, 60, 44), dtype=np.uint8)
        lm = track.landmarks[i]
        pts = [(int(lm[j, 0] * W), int(lm[j, 1] * H)) for j in range(33)]
        for a, b in conns:
            cv2.line(f, pts[a], pts[b], (90, 100, 95), 10)
        wr.write(f)
    wr.release()
    return path


@pytest.fixture(scope="module")
def good():
    t = make_track(arm_led=False)
    return t, compute_features(t).to_dict()


@pytest.fixture(scope="module")
def bad():
    t = make_track(arm_led=True)
    return t, compute_features(t).to_dict()


def test_event_detection_accuracy(good):
    track, feats = good
    ev = feats["events"]
    assert abs(ev["top"]["frame"] - TRUE_TOP) <= 2
    assert abs(ev["impact"]["frame"] - TRUE_IMPACT) <= 2
    assert ev["reliable"] is True
    assert feats["confidence"] in ("medium", "high")


def test_tempo_in_sane_range(good):
    _, feats = good
    ratio = feats["metrics"]["tempo"]["ratio_back_to_down"]
    assert 2.0 <= ratio <= 4.0  # generator targets ~3:1


def test_sequence_verdicts(good, bad):
    for (track, feats), expected in ((good, "in order"), (bad, "out of order")):
        kin = compute_kinematics(track)
        ev = feats["events"]
        seq = sequence_read(kin, ev["top"]["frame"], ev["impact"]["frame"])
        assert seq["verdict"] == expected, seq


def test_ghost_only_on_faulty_swing(good, bad):
    for (track, feats), expect_changed in ((good, False), (bad, True)):
        kin = compute_kinematics(track)
        g = build_ghost(track, kin, feats["events"])
        assert g.changed is expect_changed
        assert g.landmarks.shape == track.landmarks.shape
    # The faulty-swing ghost must actually delay the upper body.
    kin = compute_kinematics(bad[0])
    g = build_ghost(bad[0], kin, bad[1]["events"])
    assert g.shifts and all(v > 0 for v in g.shifts.values())


def test_report_builds(tmp_path, bad):
    from swingsense.report import build_report

    track, feats = bad
    video = write_video(track, str(tmp_path / "swing.mp4"))
    out = build_report(track, feats, None, video, str(tmp_path / "r.html"))
    doc = (tmp_path / "r.html").read_text()
    for marker in ("The read", "For nerds", "Kinematic sequence",
                   "ghost (re-timed)", "data:image/png;base64"):
        assert marker in doc, f"missing: {marker}"


def test_comparison_builds(tmp_path, good, bad):
    from swingsense.compare import build_comparison

    (tg, fg), (tb, fb) = good, bad
    va = write_video(tb, str(tmp_path / "a.mp4"))
    vb = write_video(tg, str(tmp_path / "b.mp4"))
    out = build_comparison(tb, fb, va, tg, fg, vb,
                           str(tmp_path / "c.html"), "before", "after")
    doc = (tmp_path / "c.html").read_text()
    assert "What changed" in doc and "Time-aligned" in doc


def test_trends(tmp_path, monkeypatch, good, bad):
    monkeypatch.setenv("SWINGSENSE_HOME", str(tmp_path))
    from swingsense import db
    from swingsense.trends import build_trends, collect

    for label, (_, feats) in (("good", good), ("bad", bad)):
        db.add_swing(feel=f"test {label}", analysis={"_features": feats},
                     club="7i")
    rows = collect()
    assert len(rows) == 2
    out = build_trends(str(tmp_path / "t.html"))
    assert out and "Tempo" in (tmp_path / "t.html").read_text()
