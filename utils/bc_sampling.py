"""
Boundary-condition sampling helpers for Dirichlet datasets.
"""
from __future__ import annotations

import numpy as np


def boundary_mask(nx: int, ny: int, width: int = 1, dtype=np.float32) -> np.ndarray:
    """Create a one-cell (or wider) boundary ring mask."""
    if nx < 2 or ny < 2:
        raise ValueError("nx and ny must both be >= 2")
    if width < 1:
        raise ValueError("width must be >= 1")
    if width * 2 >= nx or width * 2 >= ny:
        raise ValueError("width too large for grid dimensions")

    m = np.zeros((nx, ny), dtype=dtype)
    # We treat axis-0 as x-index and axis-1 as y-index throughout this codebase.
    m[:width, :] = 1
    m[-width:, :] = 1
    m[:, :width] = 1
    m[:, -width:] = 1
    return m


def _smooth_trace(n: int, rng: np.random.Generator, n_modes: int, amplitude: float) -> np.ndarray:
    """Sample a smooth 1D boundary trace using random Fourier coefficients."""
    t = np.linspace(0.0, 1.0, n, endpoint=False, dtype=np.float64)
    trace = np.zeros((n,), dtype=np.float64)
    for mode in range(1, n_modes + 1):
        # 1/mode scaling biases samples toward low frequencies, producing smooth traces.
        a = rng.normal(0.0, amplitude / mode)
        b = rng.normal(0.0, amplitude / mode)
        trace += a * np.sin(2.0 * np.pi * mode * t) + b * np.cos(2.0 * np.pi * mode * t)
    trace += rng.normal(0.0, 0.25 * amplitude)
    return trace.astype(np.float32)


def sample_boundary_value_map(
    nx: int,
    ny: int,
    rng: np.random.Generator,
    n_modes: int = 5,
    amplitude: float = 1.0,
) -> np.ndarray:
    """
    Build a boundary-only Dirichlet map g(x,y).

    Non-boundary values are zero. Boundary values are sampled as smooth random traces.
    """
    g = np.zeros((nx, ny), dtype=np.float32)

    # Independent traces per side allow diverse BC realizations.
    top = _smooth_trace(ny, rng=rng, n_modes=n_modes, amplitude=amplitude)
    bottom = _smooth_trace(ny, rng=rng, n_modes=n_modes, amplitude=amplitude)
    left = _smooth_trace(nx, rng=rng, n_modes=n_modes, amplitude=amplitude)
    right = _smooth_trace(nx, rng=rng, n_modes=n_modes, amplitude=amplitude)

    g[0, :] = top
    g[-1, :] = bottom
    g[:, 0] = left
    g[:, -1] = right

    # Corners are shared by two edges; blend to avoid edge discontinuity.
    g[0, 0] = 0.5 * (top[0] + left[0])
    g[0, -1] = 0.5 * (top[-1] + right[0])
    g[-1, 0] = 0.5 * (bottom[0] + left[-1])
    g[-1, -1] = 0.5 * (bottom[-1] + right[-1])

    return g


def sample_bc_pair(
    nx: int,
    ny: int,
    rng: np.random.Generator,
    width: int = 1,
    n_modes: int = 5,
    amplitude: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample `(g, m)` where:
      - g is boundary value map (non-zero only on boundary)
      - m is binary boundary mask
    """
    m = boundary_mask(nx=nx, ny=ny, width=width, dtype=np.float32)
    g = sample_boundary_value_map(nx=nx, ny=ny, rng=rng, n_modes=n_modes, amplitude=amplitude)
    # Explicitly zero out the interior, so g is strictly boundary-only.
    g = g * m
    return g.astype(np.float32), m.astype(np.float32)


def sample_bc_batch(
    n_samples: int,
    nx: int,
    ny: int,
    seed: int = 0,
    width: int = 1,
    n_modes: int = 5,
    amplitude: float = 1.0,
) -> np.ndarray:
    """Create BC tensor with shape (N, 2, nx, ny): channel0=g, channel1=m."""
    rng = np.random.default_rng(seed)
    bc = np.zeros((n_samples, 2, nx, ny), dtype=np.float32)
    for idx in range(n_samples):
        g, m = sample_bc_pair(
            nx=nx,
            ny=ny,
            rng=rng,
            width=width,
            n_modes=n_modes,
            amplitude=amplitude,
        )
        # Channel convention used by loaders/loss code:
        #   bc[:, 0] -> Dirichlet values g
        #   bc[:, 1] -> boundary mask m
        bc[idx, 0] = g
        bc[idx, 1] = m
    return bc
