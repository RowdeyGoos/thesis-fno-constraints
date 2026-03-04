"""
Finite-difference helpers for Dirichlet BC PDE data generation.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve


def rbf_source(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    sigma: float = 1.0 / 64.0,
    spacing: float = 2.0 / 64.0,
    center: tuple[float, float] = (0.5, 0.5),
    remove_mean: bool = False,
) -> np.ndarray:
    """Create a smooth source from a weighted grid of Gaussian RBFs."""
    ng = len(weights)
    num = int(np.sqrt(ng))
    if num * num != ng:
        raise ValueError("ng must be a perfect square for grid RBF placement")

    l = (num - 1) * spacing
    centers_x = np.arange(center[0] - l / 2.0, center[0] + l / 2.0 + spacing, spacing)
    centers_y = np.arange(center[1] - l / 2.0, center[1] + l / 2.0 + spacing, spacing)
    cx, cy = np.meshgrid(centers_x, centers_y, indexing='ij')

    source = np.zeros_like(x, dtype=np.float64)
    for idx, (px, py) in enumerate(zip(cx.reshape(-1), cy.reshape(-1))):
        r = (x - px) ** 2 + (y - py) ** 2
        source += float(weights[idx]) * np.exp(-r / (2.0 * sigma ** 2))

    # Normalize per sample so amplitude statistics stay stable across random draws.
    max_abs = float(np.max(np.abs(source)))
    if max_abs > 0:
        source /= max_abs

    if remove_mean:
        source -= np.mean(source)

    return source.astype(np.float32)


def sample_rbf_weights(
    rng: np.random.Generator,
    ng: int,
    vf: float,
    sparse: bool = True,
    min_act: float = 1.0e-3,
    max_act: float = 1.0,
) -> np.ndarray:
    """Sample activation weights for RBF sources."""
    if not sparse:
        return rng.random((ng,), dtype=np.float32)

    p = np.zeros((ng,), dtype=np.float32)
    for idx in range(ng):
        if rng.random() > vf:
            p[idx] = min_act + (max_act - min_act) * rng.random()
    if not np.any(p):
        ridx = int(rng.integers(0, ng))
        p[ridx] = min_act + (max_act - min_act) * rng.random()
    return p


def sample_diffusion_tensor(rng: np.random.Generator, e1: float = 1.0, e2: float = 5.0) -> np.ndarray:
    """Sample a random SPD 2x2 tensor via random rotation + eigenvalue draw."""
    a1 = 1.0
    a2 = float(e1 + rng.random() * (e2 - e1))

    theta = float(rng.random() * 2.0 * np.pi)
    c = np.cos(theta)
    s = np.sin(theta)
    rot = np.array([[c, -s], [s, c]], dtype=np.float64)
    rot_t = rot.T
    A = np.array([[a1, 0.0], [0.0, a2]], dtype=np.float64)
    K = rot_t @ (A @ rot)
    return K.astype(np.float32)


def sample_velocity_vector(rng: np.random.Generator, magnitude: float = 1.0) -> np.ndarray:
    """Sample a 2D velocity vector on a circle with fixed magnitude."""
    theta = float(rng.random() * 2.0 * np.pi)
    return np.array([magnitude * np.cos(theta), magnitude * np.sin(theta)], dtype=np.float32)


def _interior_idx(i: int, j: int, ny: int) -> int:
    return (i - 1) * (ny - 2) + (j - 1)


def solve_dirichlet_fd(
    source: np.ndarray,
    g: np.ndarray,
    lx: float,
    ly: float,
    k11: float,
    k12: float,
    k22: float,
    vx: float = 0.0,
    vy: float = 0.0,
    omega: float = 0.0,
) -> np.ndarray:
    """
    Solve
      k11*u_xx + 2*k12*u_xy + k22*u_yy - vx*u_x - vy*u_y + omega*u + source = 0
    on a rectangular grid with Dirichlet boundary values from g.

    Sign convention is chosen to match existing generator equations:
      diffusion - advection + helmholtz + source = 0
    """
    nx, ny = source.shape
    if g.shape != (nx, ny):
        raise ValueError(f"g must have shape {(nx, ny)}, got {g.shape}")
    if nx < 3 or ny < 3:
        raise ValueError("Grid must be at least 3x3")

    dx = lx / float(nx - 1)
    dy = ly / float(ny - 1)

    # 9-point stencil coefficients for:
    # k11*u_xx + 2*k12*u_xy + k22*u_yy - vx*u_x - vy*u_y + omega*u
    c_center = (-2.0 * k11 / (dx * dx)) + (-2.0 * k22 / (dy * dy)) + omega
    c_e = (k11 / (dx * dx)) - (vx / (2.0 * dx))
    c_w = (k11 / (dx * dx)) + (vx / (2.0 * dx))
    c_n = (k22 / (dy * dy)) - (vy / (2.0 * dy))
    c_s = (k22 / (dy * dy)) + (vy / (2.0 * dy))
    c_ne = k12 / (2.0 * dx * dy)
    c_nw = -k12 / (2.0 * dx * dy)
    c_se = -k12 / (2.0 * dx * dy)
    c_sw = k12 / (2.0 * dx * dy)

    n_unknowns = (nx - 2) * (ny - 2)
    A = lil_matrix((n_unknowns, n_unknowns), dtype=np.float64)
    b = np.zeros((n_unknowns,), dtype=np.float64)

    neighbors = [
        (0, 0, c_center),
        (1, 0, c_e),
        (-1, 0, c_w),
        (0, 1, c_n),
        (0, -1, c_s),
        (1, 1, c_ne),
        (-1, 1, c_nw),
        (1, -1, c_se),
        (-1, -1, c_sw),
    ]

    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            row = _interior_idx(i, j, ny)
            rhs = -float(source[i, j])
            for di, dj, coef in neighbors:
                ni = i + di
                nj = j + dj
                if di == 0 and dj == 0:
                    A[row, row] += coef
                    continue

                if 1 <= ni <= nx - 2 and 1 <= nj <= ny - 2:
                    col = _interior_idx(ni, nj, ny)
                    A[row, col] += coef
                else:
                    # Move known boundary terms to the right-hand side.
                    rhs -= coef * float(g[ni, nj])
            b[row] = rhs

    u_int = spsolve(A.tocsr(), b)
    if np.any(~np.isfinite(u_int)):
        raise RuntimeError("Finite-difference solve produced non-finite values")

    # Enforce boundary exactly by construction, solve only for interior nodes.
    u = np.array(g, dtype=np.float32, copy=True)
    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            u[i, j] = float(u_int[_interior_idx(i, j, ny)])
    return u
