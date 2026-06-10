"""Full-swing 3D loop: rendered body silhouette, labeled joints, hand path,
and the fitted downswing plane.

Replaces the old transition-only stick-figure loop: the window now runs from
just before address through the finish, the body is a low-poly silhouette
volume (so joints sit in a visible body, in plane), the hand path is traced
through the whole swing colored by phase, and the least-squares downswing
plane is rendered as a translucent surface.
"""

from __future__ import annotations

import numpy as np

from .vision.bodymesh import body_mesh_xyz
from .vision.pose import PoseTrack

_JOINT_GROUPS = [
    ("Head", [0], "#FF8A3D", 10),
    ("Upper torso", [11, 12, 13, 14, 15, 16], "#E8590C", 6),
    ("Pelvis", [23, 24], "#D9500B", 7),
    ("Stance", [25, 26, 27, 28], "#C2470A", 6),
]
_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (24, 26), (26, 28),
]
_BODY = "#39423D"
_GHOST = "#9AA3A0"
_MAX_FRAMES = 44


def _to_xyz(lm: np.ndarray) -> np.ndarray:
    """(33,4) landmarks -> (33,3) render coords: x, depth, up."""
    return np.stack([lm[:, 0], lm[:, 2], 1.0 - lm[:, 1]], axis=1)


def build_loop_fig(track: PoseTrack, events: dict, ghost=None,
                   path: dict | None = None):
    import plotly.graph_objects as go

    fps = track.fps
    a = events["address"]["frame"]
    fin = events.get("finish", {}).get("frame", track.n_frames - 1)
    lo = max(0, a - int(0.25 * fps))
    hi = min(track.n_frames - 1, fin)
    step = max(1, (hi - lo) // _MAX_FRAMES)
    idxs = list(range(lo, hi + 1, step))

    show_ghost = ghost is not None and getattr(ghost, "changed", False)

    def frame_data(fi: int):
        xyz = _to_xyz(track.landmarks[fi])
        verts, tris = body_mesh_xyz(xyz)
        data = [go.Mesh3d(
            x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
            i=tris[:, 0], j=tris[:, 1], k=tris[:, 2],
            color=_BODY, opacity=0.96, flatshading=True,
            lighting=dict(ambient=0.45, diffuse=0.7, specular=0.12,
                          roughness=0.9),
            lightposition=dict(x=0.4, y=-2.0, z=1.6),
            hoverinfo="skip", showlegend=False, name="body",
        )]
        for label, joints, color, size in _JOINT_GROUPS:
            pts = xyz[joints]
            data.append(go.Scatter3d(
                x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode="markers",
                name=label,
                marker=dict(size=size, color=color,
                            line=dict(color="#F2F2F2", width=2)),
            ))
        if show_ghost:
            gx = _to_xyz(ghost.landmarks[fi])
            xs, ys, zs = [], [], []
            for p, q in _CONNECTIONS:
                xs += [gx[p, 0], gx[q, 0], None]
                ys += [gx[p, 1], gx[q, 1], None]
                zs += [gx[p, 2], gx[q, 2], None]
            data.append(go.Scatter3d(
                x=xs, y=ys, z=zs, mode="lines",
                line=dict(color=_GHOST, width=5, dash="dot"),
                name="ghost (re-timed)",
            ))
        return data

    frames = [go.Frame(data=frame_data(fi), name=f"{fi / fps:.2f}s")
              for fi in idxs]

    # Static path + plane traces, appended AFTER the animated ones so frame
    # updates (which replace traces by index) never touch them.
    statics = []
    if path:
        for key, color, name in (
            ("back_pts", "#FFC845", "hand path — backswing"),
            ("down_pts", "#E8590C", "hand path — downswing"),
            ("follow_pts", "#9AA3A0", "hand path — follow"),
        ):
            pts = path.get(key)
            if pts is not None and len(pts) > 1:
                statics.append(go.Scatter3d(
                    x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode="lines",
                    line=dict(color=color, width=5), name=name, opacity=0.85,
                ))
        quad = path.get("plane_quad")
        if quad is not None:
            statics.append(go.Mesh3d(
                x=quad[:, 0], y=quad[:, 1], z=quad[:, 2],
                i=[0, 0], j=[1, 2], k=[2, 3],
                color="#4178B4", opacity=0.16,
                name="downswing plane (fit)", showlegend=True,
                hoverinfo="skip",
            ))

    title = ("3D loop — full swing: silhouette render, hand path by phase, "
             "fitted downswing plane (depth illustrative)")
    if show_ghost:
        title += " · grey ghost = your motion re-timed to fire in order"

    fig = go.Figure(data=frame_data(idxs[0]) + statics, frames=frames)
    fig.update_layout(
        title=dict(text=title, font=dict(color="#E8E8E4", size=14)),
        paper_bgcolor="#10150F",
        legend=dict(font=dict(color="#D8DCD4", size=11), orientation="h",
                    y=-0.04, bgcolor="rgba(0,0,0,0)"),
        scene=dict(
            xaxis=dict(visible=False), yaxis=dict(visible=False),
            zaxis=dict(visible=False), aspectmode="data",
            bgcolor="#10150F",
            camera=dict(eye=dict(x=0.0, y=-2.2, z=0.15)),
        ),
        height=640, margin=dict(t=64, b=10),
        updatemenus=[dict(
            type="buttons", showactive=False, y=0.02, x=0,
            font=dict(color="#14211A"), bgcolor="#E8E8E4",
            buttons=[
                dict(label="▶ play", method="animate",
                     args=[None, dict(frame=dict(duration=70, redraw=True),
                                      fromcurrent=True)]),
                dict(label="⏸ pause", method="animate",
                     args=[[None], dict(mode="immediate",
                                        frame=dict(duration=0, redraw=False))]),
            ],
        )],
        sliders=[dict(
            font=dict(color="#D8DCD4", size=10),
            steps=[dict(method="animate", label=f.name,
                        args=[[f.name], dict(mode="immediate",
                                             frame=dict(duration=0,
                                                        redraw=True))])
                   for f in frames],
            y=-0.10, len=0.92,
        )],
    )
    return fig
