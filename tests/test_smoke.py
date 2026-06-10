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


def test_pro_reference_roundtrip(tmp_path, monkeypatch, good, bad):
    monkeypatch.setenv("SWINGSENSE_HOME", str(tmp_path))
    from swingsense.pro import (add_reference, load_reference,
                                build_pro_comparison, norms_assessment)

    (tg, fg), (tb, fb) = good, bad
    add_reference(tg, fg, "tour_model", source="test")
    ref = load_reference("tour_model")
    assert ref and len(ref["signals"]["pelvis"]) == 200
    assert ref["firing_order"][0] == "pelvis"

    kin = compute_kinematics(tb)
    lines = norms_assessment(fb, kin, fb["events"])
    assert any("reversed" in l or "simultaneously" in l for l in lines)

    video = write_video(tb, str(tmp_path / "s.mp4"))
    out = build_pro_comparison(tb, fb, video, ref,
                               str(tmp_path / "vs.html"), label="you")
    doc = (tmp_path / "vs.html").read_text()
    assert "published tour norms" in doc and "tour_model" in doc


def test_restyled_3d_loop_markers(tmp_path, bad):
    from swingsense.report import build_report

    track, feats = bad
    video = write_video(track, str(tmp_path / "swing.mp4"))
    build_report(track, feats, None, video, str(tmp_path / "r.html"))
    doc = (tmp_path / "r.html").read_text()
    for label in ("Head", "Upper torso", "Pelvis", "Stance", "hand path"):
        assert label in doc, f"missing joint-group label: {label}"


def test_path_analysis(bad):
    from swingsense.vision.path import analyze_path

    track, feats = bad
    pa = analyze_path(track, feats["events"])
    plane = pa["plane"]
    assert plane is not None
    assert 0 <= plane["tilt_deg"] <= 90
    assert plane["rms_offplane_pct"] >= 0
    assert len(pa["down_pts"]) >= 4
    assert any("not club path" in n for n in pa["notes"])


def test_body_mesh_valid(good):
    from swingsense.loop3d import _to_xyz
    from swingsense.vision.bodymesh import body_mesh_xyz

    track, _ = good
    verts, tris = body_mesh_xyz(_to_xyz(track.landmarks[20]))
    assert verts.ndim == 2 and verts.shape[1] == 3
    assert tris.ndim == 2 and tris.shape[1] == 3
    assert tris.max() < len(verts) and tris.min() >= 0
    assert np.isfinite(verts).all()


def test_full_swing_loop_in_report(tmp_path, bad):
    from swingsense.report import build_report

    track, feats = bad
    video = write_video(track, str(tmp_path / "swing.mp4"))
    build_report(track, feats, None, video, str(tmp_path / "r.html"))
    doc = (tmp_path / "r.html").read_text()
    for m in ("mesh3d", "downswing plane (fit)", "Path analysis",
              "full swing", "hand path"):
        assert m.lower() in doc.lower(), f"missing: {m}"


def test_annotated_video_render(tmp_path, good):
    from swingsense.vision.render_video import (render_swing_video,
                                                render_trace_card)

    track, feats = good
    video = write_video(track, str(tmp_path / "in.mp4"))
    out = render_swing_video(track, video, feats["events"],
                             str(tmp_path / "out.mp4"), tempo_ratio=3.0)
    assert (tmp_path / "out.mp4").stat().st_size > 10_000
    card = render_trace_card(track, video, feats["events"],
                             str(tmp_path / "card.png"))
    assert (tmp_path / "card.png").stat().st_size > 5_000
    # The annotated video should be LONGER than the swing segment it covers
    # (slow-mo downswing + event freeze-frames add frames).
    n_out = int(cv2.VideoCapture(out).get(cv2.CAP_PROP_FRAME_COUNT))
    a = feats["events"]["address"]["frame"]
    fin = feats["events"]["finish"]["frame"]
    assert n_out > (fin - a)
