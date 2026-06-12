"""Club shaft/head tracking.

The club is the second pendulum and carries the swing's energy peak, but pose
models don't see it. We detect the shaft as a line segment anchored near the
hands (Canny edges + probabilistic Hough), score candidates by length and
temporal angle-continuity, and take the far endpoint as the club head.

Why this matters (learned from real footage): on a full swing the follow-through
wrap often takes the HANDS higher than the actual top of the backswing, so any
hands-only "highest point" logic mislabels the follow-through as the top. The
club head's trajectory passes through the ball exactly once — that crossing is
impact, and it splits backswing from follow-through unambiguously.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .pose import L_HIP, L_SHOULDER, L_WRIST, R_HIP, R_SHOULDER, R_WRIST, PoseTrack


@dataclass
class ClubTrack:
    """Per-frame club observations (NaN where the shaft wasn't found)."""

    head_xy: np.ndarray  # (n, 2) clubhead position, normalized image coords
    angle: np.ndarray  # (n,) shaft angle in degrees (atan2 of head-grip)
    length: np.ndarray  # (n,) shaft length in normalized units
    found: np.ndarray  # (n,) bool

    def coverage(self) -> float:
        return float(self.found.mean()) if len(self.found) else 0.0


def _detect_shaft_in_frame(
    gray,
    prev_gray,
    grip_xy: tuple[float, float],
    body_scale: float,
    prev_angle: float | None,
):
    """Find the most shaft-like line segment near the grip in one frame.

    Works on MOTION edges (frame difference) rather than raw edges: the shaft
    moves while background lines (screen edges, mats, net poles) stay still, so
    differencing suppresses static clutter and highlights the swinging shaft —
    including its motion-blur streak on fast frames.

    Returns (head_xy, angle_deg, length) in pixel coords, or None.
    """
    import cv2

    h, w = gray.shape[:2]
    gx, gy = grip_xy
    if not (0 <= gx < w and 0 <= gy < h):
        return None

    # Search a generous window around the hands — the shaft can extend ~2.5x
    # the torso scale in any direction.
    r = int(2.8 * body_scale)
    x0, y0 = max(0, int(gx - r)), max(0, int(gy - r))
    x1, y1 = min(w, int(gx + r)), min(h, int(gy + r))
    roi = gray[y0:y1, x0:x1]
    if roi.size == 0:
        return None

    if prev_gray is not None:
        diff = cv2.absdiff(roi, prev_gray[y0:y1, x0:x1])
        diff = cv2.GaussianBlur(diff, (3, 3), 0)
        edges = cv2.Canny(diff, 15, 60)
    else:
        edges = cv2.Canny(roi, 40, 120)
    # Keep the length gate permissive: Hough fragments the shaft (especially
    # when motion-blurred), so demanding torso-scale segments returns nothing.
    # Short fragments are fine — the near-grip + continuity scoring sorts them.
    min_len = max(int(0.3 * body_scale), 20)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=20,
        minLineLength=min_len,
        maxLineGap=8,
    )
    if lines is None:
        return None

    grip_local = np.array([gx - x0, gy - y0])
    # Generous: pose wrists wander when the hands are overhead/occluded.
    near_r = 0.9 * body_scale  # one endpoint must be near the hands
    best = None
    best_score = -1.0
    for x1l, y1l, x2l, y2l in lines[:, 0]:
        p1 = np.array([x1l, y1l], dtype=float)
        p2 = np.array([x2l, y2l], dtype=float)
        d1, d2 = np.linalg.norm(p1 - grip_local), np.linalg.norm(p2 - grip_local)
        near, far = (p1, p2) if d1 < d2 else (p2, p1)
        d_near = min(d1, d2)
        if d_near > near_r:
            continue
        seg_len = float(np.linalg.norm(p2 - p1))
        ang = float(np.degrees(np.arctan2(far[1] - near[1], far[0] - near[0])))
        # Score: long segments anchored close to the hands, consistent with the
        # previous frame's shaft direction when we have one.
        score = seg_len - 0.8 * d_near
        if prev_angle is not None:
            dang = abs((ang - prev_angle + 180) % 360 - 180)
            score -= 0.6 * dang
        if score > best_score:
            best_score = score
            best = (far + np.array([x0, y0]), ang, seg_len)
    return best


def track_club(video_path: str, pose: PoseTrack) -> ClubTrack:
    """Run shaft detection over the whole video, anchored at the tracked hands."""
    import cv2

    n = pose.n_frames
    head = np.full((n, 2), np.nan, dtype=float)
    angle = np.full(n, np.nan, dtype=float)
    length = np.full(n, np.nan, dtype=float)
    found = np.zeros(n, dtype=bool)

    wrist = pose.midpoint(L_WRIST, R_WRIST)
    shoulder = pose.midpoint(L_SHOULDER, R_SHOULDER)
    hip = pose.midpoint(L_HIP, R_HIP)

    cap = cv2.VideoCapture(video_path)
    W, H = pose.width, pose.height
    prev_angle: float | None = None
    prev_gray = None
    last_head = None
    miss_streak = 0

    for i in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # body scale: torso height in pixels (shoulder-to-hip)
        torso = np.linalg.norm((shoulder[i, :2] - hip[i, :2]) * [W, H])
        body_scale = max(float(torso), 30.0)
        grip = (float(wrist[i, 0] * W), float(wrist[i, 1] * H))

        res = _detect_shaft_in_frame(gray, prev_gray, grip, body_scale, prev_angle)
        prev_gray = gray
        if res is not None:
            head_px, ang, seg_len = res
            cand = np.array([head_px[0] / W, head_px[1] / H])
            # Trajectory gate: a real clubhead moves continuously; junk lines
            # (mat edges, legs, screen frames) teleport. Accept only detections
            # within a speed-plausible distance of the last accepted position,
            # with the gate widening the longer we've been blind.
            if last_head is not None:
                gap = min(miss_streak + 1, 6)
                gate = 0.12 * gap  # ~12% of the frame per frame of gap
                if float(np.linalg.norm(cand - last_head)) > gate:
                    res = None  # reject the teleport
        if res is not None:
            head[i] = cand
            angle[i] = ang
            length[i] = seg_len / max(W, H)
            found[i] = True
            prev_angle = ang
            last_head = cand
            miss_streak = 0
        else:
            miss_streak += 1
            if miss_streak > 5:
                prev_angle = None  # don't drag stale continuity through long gaps
                last_head = None  # ...nor a stale position gate

    cap.release()
    return ClubTrack(head_xy=head, angle=angle, length=length, found=found)


def _interp_track(club: ClubTrack) -> np.ndarray:
    """Clubhead trace with gaps linearly interpolated between detections."""
    n = len(club.found)
    out = club.head_xy.copy()
    idx = np.where(club.found)[0]
    if len(idx) < 2:
        return out
    t = np.arange(n)
    for k in (0, 1):
        out[:, k] = np.interp(t, idx, club.head_xy[idx, k])
    return out


def impact_shaft_lean(
    club: ClubTrack, pose: PoseTrack, impact: int
) -> dict | None:
    """Measure shaft lean at the impact pillar (face-on view).

    Efficient impact: hands AHEAD of the clubhead toward the target (forward
    shaft lean) — the release isn't spent, the head is still accelerating, and
    the contact force aligns with the ball's trajectory (see KB:
    shaft_lean_impact). Positive lean = hands lead; negative = clubhead has
    passed the hands (flip/early release).

    Target direction is inferred from the clubhead's horizontal motion just
    after impact. Uses the nearest frame to `impact` with a club detection.
    """
    from .pose import L_WRIST, R_WRIST

    n = len(club.found)
    # Read lean strictly BEFORE the strike: at/after contact the clubhead
    # passes the hands naturally, so a post-impact frame always reads "flip"
    # regardless of how good the impact actually was. Use the last detection
    # in the few frames leading into impact.
    cands = [j for j in range(max(0, impact - 4), max(1, impact)) if club.found[j]]
    if not cands:
        return None
    j = max(cands)

    head = _interp_track(club)
    lo, hi = max(0, impact - 2), min(n - 1, impact + 3)
    dx = head[hi, 0] - head[lo, 0]
    if abs(dx) < 1e-4:
        return None
    target_sign = 1.0 if dx > 0 else -1.0

    grip_x = float(pose.midpoint(L_WRIST, R_WRIST)[j, 0])
    lean_raw = (grip_x - float(head[j, 0])) * target_sign
    # normalize by shaft horizontal extent so the number is camera-independent
    shaft_dx = abs(grip_x - float(head[j, 0])) + 1e-6
    return {
        "frame_used": int(j),
        "hands_lead": bool(lean_raw > 0),
        "lean_norm": round(float(lean_raw / shaft_dx), 2),  # +1 lead .. -1 flip
        "lean_raw": round(float(lean_raw), 3),
    }


def club_events(
    club: ClubTrack, fps: float, window: tuple[int, int] | None = None
) -> dict | None:
    """Derive the swing sequence from the clubhead trajectory.

    The clubhead passes through the ball exactly once, moving fast — that is
    impact, and it cleanly splits backswing from follow-through (the hands
    cannot: a full follow-through wrap often takes them HIGHER than the top).

      impact       lowest clubhead point where the clubhead is moving fast
      top          highest clubhead point before impact (the club's true peak)
      follow_peak  highest clubhead point after impact
      address      last slow-and-low clubhead frame before the climb to the top

    Returns None when coverage is too thin to trust.
    """
    if club.coverage() < 0.2 or club.found.sum() < 10:
        return None

    n = len(club.found)
    head = _interp_track(club)
    y = head[:, 1]
    speed = np.hypot(*np.gradient(head, axis=0).T) * fps

    # Impact candidates: clubhead LOW (y in the bottom third of its range) and
    # FAST (top quartile of speed). Among them take the fastest.
    y_lo, y_hi = float(np.min(y)), float(np.max(y))
    low_cut = y_lo + 0.6 * (y_hi - y_lo)
    fast_cut = float(np.percentile(speed[club.found], 75))
    # Constrain the search to the activity window when given (from the pose
    # layer, which reliably brackets WHEN the swing happens even when it
    # mislabels the phases within it). This excludes practice swings before
    # and camera-pan junk after the real strike.
    j_lo, j_hi = int(fps * 0.3), n
    if window is not None:
        j_lo = max(j_lo, window[0])
        j_hi = min(n, window[1])

    cands = [
        j for j in range(j_lo, j_hi) if y[j] > low_cut and speed[j] >= fast_cut
    ]
    if not cands:
        return None

    # The real impact sits at the bottom of a V: the clubhead was HIGH before it
    # (the top of the backswing) AND goes HIGH after it (the follow-through).
    # Junk motion has neither side; a backswing-only clip lacks the after-side.
    # Score each candidate by the weaker of its two sides and take the best.
    def v_score(j: int) -> float:
        before = y[j] - float(np.min(y[max(0, j - int(fps * 1.8)) : j + 1]))
        after = y[j] - float(np.min(y[j : min(n, j + int(fps * 1.5))]))
        return min(before, after)

    def local_density(j: int) -> float:
        lo, hi = max(0, j - int(fps * 1.2)), min(n, j + int(fps * 1.2))
        return float(club.found[lo:hi].mean())

    # Valid impacts: a real V (backswing AND follow-through) in a DENSELY
    # detected region (sparse regions are interpolation artifacts, not club).
    valid = [
        j
        for j in cands
        if v_score(j) >= 0.25 * (y_hi - y_lo) and local_density(j) >= 0.5
    ]
    if not valid:
        return None
    # The real strike is the DEEPEST V — practice waggles and post-swing
    # club-waving produce shallower ones. (Practice swings before / camera junk
    # after are already excluded by the activity window when provided.)
    impact = max(valid, key=v_score)

    # Top: the club's true height peak in the window before impact.
    t_lo = max(0, impact - int(fps * 1.8))
    top = t_lo + int(np.argmin(y[t_lo:impact])) if impact > t_lo else impact
    if impact - top < 3:
        return None

    # Follow-through peak: the club's height peak after impact.
    f_hi = min(n, impact + int(fps * 1.5))
    follow_peak = (
        impact + int(np.argmin(y[impact:f_hi])) if f_hi > impact + 1 else impact
    )

    # Address: last slow-and-low frame before the climb.
    a_lo = max(0, top - int(fps * 1.6))
    address = a_lo
    slow_cut = float(np.percentile(speed[club.found], 35))
    for j in range(top - 1, a_lo, -1):
        if y[j] > low_cut and speed[j] < slow_cut:
            address = j
            break

    def t(i: int) -> float:
        return round(i / fps, 3)

    mid_bsw = address + int(round((top - address) * 0.66))
    return {
        "address": {"frame": int(address), "t": t(int(address))},
        "mid_backswing": {"frame": int(mid_bsw), "t": t(int(mid_bsw))},
        "top": {"frame": int(top), "t": t(int(top))},
        "transition": {"frame": int(top) + 1, "t": t(int(top) + 1)},
        "impact": {"frame": int(impact), "t": t(int(impact))},
        "follow_through": {"frame": int(follow_peak), "t": t(int(follow_peak))},
        "finish": {"frame": int(min(n - 1, follow_peak + int(fps * 0.5))),
                   "t": t(int(min(n - 1, follow_peak + int(fps * 0.5))))},
        "source": "club",
    }
