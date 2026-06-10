"""Low-poly body volume from pose landmarks — the silhouette for the 3D loop.

Each bone becomes a 6-sided tube, the torso gets thicker tubes, the head a
small UV sphere; everything is merged into one Mesh3d-ready vertex/triangle
set per frame so the render stays light enough to animate.
"""

from __future__ import annotations

import numpy as np

# (a, b, radius) in normalized units. Limbs thinner, torso thicker.
_LIMBS = [
    (11, 13, 0.024), (13, 15, 0.019),   # left arm
    (12, 14, 0.024), (14, 16, 0.019),   # right arm
    (23, 25, 0.032), (25, 27, 0.026),   # left leg
    (24, 26, 0.032), (26, 28, 0.026),   # right leg
]
_TORSO = [
    (11, 12, 0.045), (23, 24, 0.048),   # shoulder line, hip line
    (11, 23, 0.052), (12, 24, 0.052),   # flanks
]
_HEAD_R = 0.052
_NECK_R = 0.022


def _basis(axis: np.ndarray):
    axis = axis / (np.linalg.norm(axis) + 1e-9)
    tmp = (np.array([0.0, 0.0, 1.0]) if abs(axis[2]) < 0.9
           else np.array([1.0, 0.0, 0.0]))
    u = np.cross(axis, tmp)
    u /= np.linalg.norm(u) + 1e-9
    v = np.cross(axis, u)
    return u, v


def _tube(p0: np.ndarray, p1: np.ndarray, r: float, n: int = 6):
    u, v = _basis(p1 - p0)
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = (np.outer(np.cos(ang), u) + np.outer(np.sin(ang), v)) * r
    verts = np.vstack([p0 + ring, p1 + ring])
    tri = []
    for k in range(n):
        a, b = k, (k + 1) % n
        tri += [(a, b, n + a), (b, n + b, n + a)]
    return verts, np.array(tri, dtype=int)


def _sphere(c: np.ndarray, r: float, n_u: int = 8, n_v: int = 5):
    us = np.linspace(0, 2 * np.pi, n_u, endpoint=False)
    vs = np.linspace(0.15 * np.pi, 0.85 * np.pi, n_v)
    verts = np.array([
        c + r * np.array([np.cos(u) * np.sin(v),
                          np.sin(u) * np.sin(v),
                          np.cos(v)])
        for v in vs for u in us
    ])
    tri = []
    for i in range(n_v - 1):
        for j in range(n_u):
            a = i * n_u + j
            b = i * n_u + (j + 1) % n_u
            c2 = (i + 1) * n_u + j
            d = (i + 1) * n_u + (j + 1) % n_u
            tri += [(a, b, c2), (b, d, c2)]
    return verts, np.array(tri, dtype=int)


def body_mesh_xyz(xyz: np.ndarray):
    """Build the merged body mesh for one frame.

    xyz: (33, 3) landmark positions already in render coordinates.
    Returns (verts (N,3), tris (M,3)).
    """
    V, T, off = [], [], 0

    def add(verts, tris):
        nonlocal off
        V.append(verts)
        T.append(tris + off)
        off += len(verts)

    for a, b, r in _LIMBS + _TORSO:
        add(*_tube(xyz[a], xyz[b], r))
    neck_base = (xyz[11] + xyz[12]) / 2.0
    add(*_tube(neck_base, xyz[0], _NECK_R))
    add(*_sphere(xyz[0] + np.array([0.0, 0.0, 0.012]), _HEAD_R))
    return np.vstack(V), np.vstack(T)
