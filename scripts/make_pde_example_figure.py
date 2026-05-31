#!/usr/bin/env python3
"""Create thesis-ready example figures for the Poisson, advection-diffusion,
and Helmholtz datasets.

The script expects each HDF5 dataset to use the repository's standard layout:

    fields[:, 0] = source / forcing field f(x)
    fields[:, 1] = solution field u(x)

If the compact example datasets are not present, they are generated with the
existing scripts/data/gen_data_*.py generators.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class PDESpec:
    key: str
    name: str
    equation: str
    generator: Path
    generated_dirname: str
    generated_filename: str
    individual_prefix: str
    generator_args: tuple[str, ...]

    def generated_path(self, data_root: Path) -> Path:
        return data_root / self.generated_dirname / self.generated_filename


PDE_SPECS: tuple[PDESpec, ...] = (
    PDESpec(
        key="poisson",
        name="Poisson",
        equation=r"$-\nabla\!\cdot(K\nabla u)=f$",
        generator=REPO_ROOT / "scripts" / "data" / "gen_data_poisson.py",
        generated_dirname="poisson",
        generated_filename="_train_k1_5_32k.h5",
        individual_prefix="poisson",
        generator_args=(),
    ),
    PDESpec(
        key="advdiff",
        name="Advection–Diffusion",
        equation=r"$-\nabla\!\cdot(K\nabla u)+v\!\cdot\nabla u=f$",
        generator=REPO_ROOT / "scripts" / "data" / "gen_data_advdiff.py",
        generated_dirname="advdiff",
        generated_filename="_train_adr0.2_1_32k.h5",
        individual_prefix="advdiff",
        generator_args=(),
    ),
    PDESpec(
        key="helmholtz",
        name="Helmholtz",
        equation=r"$-\Delta u-\omega u=f$",
        generator=REPO_ROOT / "scripts" / "data" / "gen_data_helmholtz.py",
        generated_dirname="helmholtz",
        generated_filename="_train_o1_10_32k.h5",
        individual_prefix="helmholtz",
        generator_args=(),
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Figure 3.1 PDE source/solution examples."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=REPO_ROOT / "data" / "pde_examples_matched",
        help="Directory for compact generated HDF5 datasets with matched source fields.",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=REPO_ROOT / "figures",
        help="Directory where figures are saved.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=64,
        help="Number of training samples to generate per PDE if data is missing.",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=128,
        help="Spatial grid size for generated example datasets.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for PNG output.",
    )
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="Regenerate compact example datasets even if they already exist.",
    )
    parser.add_argument(
        "--no-generate",
        action="store_true",
        help="Do not generate missing HDF5 datasets; fail if any are absent.",
    )
    parser.add_argument("--poisson-path", type=Path, default=None)
    parser.add_argument("--advdiff-path", type=Path, default=None)
    parser.add_argument("--helmholtz-path", type=Path, default=None)
    return parser.parse_args()


def path_for_spec(spec: PDESpec, args: argparse.Namespace) -> Path:
    explicit_paths = {
        "poisson": args.poisson_path,
        "advdiff": args.advdiff_path,
        "helmholtz": args.helmholtz_path,
    }
    return explicit_paths[spec.key] or spec.generated_path(args.data_root)


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def fft_coef(n: int) -> np.ndarray:
    positive = 1j * np.arange(0, n // 2 + 1, 1)
    negative = 1j * np.arange(-n // 2 + 1, 0, 1)
    return np.concatenate((positive, negative))


def make_random_source(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    ng: int,
    vf: float,
    sigma: float,
    spacing: float,
) -> np.ndarray:
    num = int(np.sqrt(ng))
    if num * num != ng:
        raise ValueError(f"--ng must be a perfect square, got {ng}")

    p = np.zeros(ng)
    active = np.random.random(ng) > vf
    p[active] = 1.0e-3 + (1.0 - 1.0e-3) * np.random.random(np.count_nonzero(active))
    if not np.any(p):
        p[np.random.randint(0, ng)] = 1.0

    length = (num - 1) * spacing
    centers_x = np.arange(0.5 - length / 2, 0.5 + length / 2 + spacing, spacing)
    centers_y = np.arange(0.5 - length / 2, 0.5 + length / 2 + spacing, spacing)
    centers_x, centers_y = np.meshgrid(centers_x, centers_y)

    source = np.zeros_like(x_grid, dtype=np.float64)
    for weight, cx, cy in zip(p, centers_x.ravel(), centers_y.ravel()):
        radius = (x_grid - cx) ** 2 + (y_grid - cy) ** 2
        source += weight * np.exp(-radius / (2.0 * sigma**2))

    source /= max(float(np.max(source)), 1.0e-12)
    source -= np.mean(source)
    return source


def random_diffusion_tensor(e1: float = 1.0, e2: float = 5.0) -> np.ndarray:
    a4 = e1 + np.random.random() * (e2 - e1)
    theta = np.random.random() * 2.0 * np.pi
    rot = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )
    rot_inv = rot.T
    return rot_inv @ (np.array([[1.0, 0.0], [0.0, a4]]) @ rot)


def random_velocity() -> np.ndarray:
    theta = np.random.random() * 2.0 * np.pi
    return np.array([np.cos(theta), np.sin(theta)])


def spectral_factors(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ikx = np.repeat(fft_coef(n).reshape(1, n), n, axis=0)
    iky = np.repeat(fft_coef(n).reshape(n, 1), n, axis=1)
    return ikx, iky, ikx**2, iky**2


def solve_poisson(source: np.ndarray, ikx: np.ndarray, iky: np.ndarray, ikx2: np.ndarray, iky2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    k_mat = random_diffusion_tensor()
    diff_factor = ikx2 * k_mat[0, 0] + iky2 * k_mat[1, 1] + 2.0 * ikx * iky * k_mat[0, 1]
    diff_factor *= 4.0 * np.pi**2 * 0.01
    inv_factor = np.zeros_like(diff_factor, dtype=np.complex128)
    np.divide(-1.0, diff_factor, out=inv_factor, where=diff_factor != 0)
    solution = np.real(np.fft.ifft2(inv_factor * np.fft.fft2(source)))
    solution -= np.mean(solution)
    tensor = np.array([k_mat[0, 0], k_mat[1, 1], k_mat[0, 1]]) * 0.01
    return solution, tensor


def solve_advdiff(source: np.ndarray, ikx: np.ndarray, iky: np.ndarray, ikx2: np.ndarray, iky2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    k_mat = random_diffusion_tensor()
    velocity = random_velocity()
    lams = np.load(REPO_ROOT / "utils" / "lambda.npy")
    ads = np.load(REPO_ROOT / "utils" / "ads.npy")
    # Bias the compact visualization pool toward transport-dominated cases so
    # the selected example is visually distinct from the Poisson solution.
    target_ratio = 0.55 + np.random.random() * (1.0 - 0.55)
    lam = float(lams[np.argmin(np.abs(np.mean(ads, axis=1) - target_ratio))])

    diff_factor = ikx2 * k_mat[0, 0] + iky2 * k_mat[1, 1] + 2.0 * ikx * iky * k_mat[0, 1]
    diff_factor *= 4.0 * np.pi**2
    adv_factor = (velocity[0] * ikx + velocity[1] * iky) * (2.0 * np.pi)
    factor = (1.0 - lam) * 0.01 * diff_factor - lam * adv_factor
    inv_factor = np.zeros_like(factor, dtype=np.complex128)
    np.divide(-1.0, factor, out=inv_factor, where=factor != 0)
    solution = np.real(np.fft.ifft2(inv_factor * np.fft.fft2(source)))
    solution -= np.mean(solution)
    tensor = np.zeros(5)
    tensor[:3] = np.array([k_mat[0, 0], k_mat[1, 1], k_mat[0, 1]]) * 0.01 * (1.0 - lam)
    tensor[3:5] = velocity * lam
    return solution, tensor


def solve_helmholtz(source: np.ndarray, ikx2: np.ndarray, iky2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    omega = np.random.randint(1, 11)
    factor = omega + (ikx2 + iky2) * 4.0 * np.pi**2 * 0.01
    inv_factor = np.zeros_like(factor, dtype=np.complex128)
    np.divide(-1.0, factor, out=inv_factor, where=factor != 0)
    solution = np.real(np.fft.ifft2(inv_factor * np.fft.fft2(source)))
    tensor = np.array([0.01, omega])
    return solution, tensor


def write_hdf5(path: Path, fields: np.ndarray, tensor: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("fields", data=fields.astype("<f4"))
        handle.create_dataset("tensor", data=tensor.astype("<f4"))


def generate_matched_datasets(args: argparse.Namespace) -> None:
    np.random.seed(0)
    n = args.grid_size
    x = np.arange(0.0, 1.0, 1.0 / n)
    y = np.arange(0.0, 1.0, 1.0 / n)
    x_grid, y_grid = np.meshgrid(x, y)
    ikx, iky, ikx2, iky2 = spectral_factors(n)
    std = 1.0 / 32.0
    spacing = 2.0 * std
    vfs = (0.2, 0.4, 0.6, 0.8)

    fields = {
        "poisson": np.zeros((args.n_samples, 2, n, n), dtype=np.float32),
        "advdiff": np.zeros((args.n_samples, 2, n, n), dtype=np.float32),
        "helmholtz": np.zeros((args.n_samples, 2, n, n), dtype=np.float32),
    }
    tensors = {
        "poisson": np.zeros((args.n_samples, 3), dtype=np.float32),
        "advdiff": np.zeros((args.n_samples, 5), dtype=np.float32),
        "helmholtz": np.zeros((args.n_samples, 2), dtype=np.float32),
    }

    for idx in range(args.n_samples):
        source = make_random_source(x_grid, y_grid, 144, vfs[idx % len(vfs)], std, spacing)

        poisson_solution, poisson_tensor = solve_poisson(source, ikx, iky, ikx2, iky2)
        advdiff_solution, advdiff_tensor = solve_advdiff(source, ikx, iky, ikx2, iky2)
        helmholtz_solution, helmholtz_tensor = solve_helmholtz(source, ikx2, iky2)

        for key, solution, tensor in (
            ("poisson", poisson_solution, poisson_tensor),
            ("advdiff", advdiff_solution, advdiff_tensor),
            ("helmholtz", helmholtz_solution, helmholtz_tensor),
        ):
            fields[key][idx, 0] = source
            fields[key][idx, 1] = solution
            tensors[key][idx] = tensor

    for spec in PDE_SPECS:
        write_hdf5(spec.generated_path(args.data_root), fields[spec.key], tensors[spec.key])
    print(f"Generated matched example datasets in {display_path(args.data_root)}")


def ensure_datasets(args: argparse.Namespace) -> dict[str, Path]:
    dataset_paths: dict[str, Path] = {}
    did_generate = False
    explicit_paths = any(
        path is not None
        for path in (args.poisson_path, args.advdiff_path, args.helmholtz_path)
    )
    generated_paths = [spec.generated_path(args.data_root) for spec in PDE_SPECS]
    if not explicit_paths and not args.no_generate:
        if args.regenerate or any(not path.exists() for path in generated_paths):
            generate_matched_datasets(args)
            did_generate = True

    for spec in PDE_SPECS:
        path = path_for_spec(spec, args)
        if path.exists() and (not args.regenerate or did_generate):
            dataset_paths[spec.key] = path
            continue
        if args.no_generate:
            raise FileNotFoundError(f"Missing {spec.name} dataset: {path}")
        if path != spec.generated_path(args.data_root):
            raise FileNotFoundError(
                f"Explicit {spec.name} dataset path does not exist: {path}"
            )
        generate_matched_datasets(args)
        did_generate = True
        dataset_paths[spec.key] = path
    return dataset_paths


def _as_float_array(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D field, got shape {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError("Field contains NaN or Inf values")
    return array


def load_dataset(h5_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    with h5py.File(h5_path, "r") as handle:
        if "fields" not in handle:
            raise KeyError(f"{h5_path} does not contain a 'fields' dataset")
        fields = handle["fields"]
        if fields.ndim != 4 or fields.shape[1] < 2:
            raise ValueError(
                f"{h5_path} fields must have shape (N, 2, nx, ny), got {fields.shape}"
            )

        solutions = np.asarray(fields[:, 1], dtype=np.float64)
        sources = np.asarray(fields[:, 0], dtype=np.float64)
        tensor = (
            np.asarray(handle["tensor"], dtype=np.float64)
            if "tensor" in handle
            else None
        )
    return sources, solutions, tensor


def candidate_indices(sources: np.ndarray, solutions: np.ndarray, limit: int = 24) -> np.ndarray:
    sol_variance = np.var(solutions, axis=(1, 2))
    src_variance = np.var(sources, axis=(1, 2))
    finite = np.isfinite(sol_variance) & np.isfinite(src_variance)
    informative = finite & (sol_variance > 1.0e-12) & (src_variance > 1.0e-12)
    candidates = np.flatnonzero(informative if np.any(informative) else finite)
    if candidates.size == 0:
        raise ValueError("No finite samples found")

    median_var = np.median(sol_variance[candidates])
    relative_distance = np.abs(sol_variance[candidates] - median_var) / (
        abs(median_var) + 1.0e-12
    )
    ordered = candidates[np.argsort(relative_distance)]
    return ordered[: min(limit, ordered.size)]


def normalized_source(field: np.ndarray) -> np.ndarray:
    centered = field - np.mean(field)
    scale = np.std(centered)
    if scale <= 1.0e-12:
        return centered
    return centered / scale


def source_distance(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        raise ValueError(
            f"Cannot compare source fields with shapes {a.shape} and {b.shape}"
        )
    return float(np.mean((normalized_source(a) - normalized_source(b)) ** 2))


def score01(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    values = np.where(np.isfinite(values), values, np.nan)
    if np.all(np.isnan(values)):
        return np.zeros_like(values)
    lo = np.nanpercentile(values, 5.0)
    hi = np.nanpercentile(values, 95.0)
    if not np.isfinite(lo) or not np.isfinite(hi) or np.isclose(lo, hi):
        lo = np.nanmin(values)
        hi = np.nanmax(values)
    if np.isclose(lo, hi):
        return np.zeros_like(values)
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0)


def robust_amplitude_batch(fields: np.ndarray) -> np.ndarray:
    high = np.percentile(fields, 98.0, axis=(1, 2))
    low = np.percentile(fields, 2.0, axis=(1, 2))
    return high - low


def centroid_distance(field: np.ndarray) -> float:
    weights = np.abs(field)
    total = float(np.sum(weights))
    if total <= 1.0e-12:
        return 0.0
    yy, xx = np.indices(field.shape)
    cx = float(np.sum(xx * weights) / total) / max(field.shape[1] - 1, 1)
    cy = float(np.sum(yy * weights) / total) / max(field.shape[0] - 1, 1)
    return float(np.hypot(cx - 0.5, cy - 0.5) / np.sqrt(0.5))


def anisotropy(field: np.ndarray) -> float:
    weights = np.abs(field)
    total = float(np.sum(weights))
    if total <= 1.0e-12:
        return 0.0
    yy, xx = np.indices(field.shape)
    x = xx / max(field.shape[1] - 1, 1)
    y = yy / max(field.shape[0] - 1, 1)
    mx = float(np.sum(x * weights) / total)
    my = float(np.sum(y * weights) / total)
    dx = x - mx
    dy = y - my
    cov_xx = float(np.sum(weights * dx * dx) / total)
    cov_yy = float(np.sum(weights * dy * dy) / total)
    cov_xy = float(np.sum(weights * dx * dy) / total)
    eigvals = np.linalg.eigvalsh(np.array([[cov_xx, cov_xy], [cov_xy, cov_yy]]))
    if eigvals[1] <= 1.0e-12:
        return 0.0
    return float(1.0 - eigvals[0] / eigvals[1])


def skewness_abs(field: np.ndarray) -> float:
    centered = field - np.mean(field)
    std = float(np.std(centered))
    if std <= 1.0e-12:
        return 0.0
    return float(abs(np.mean((centered / std) ** 3)))


def zero_crossing_fraction(field: np.ndarray) -> float:
    horizontal = field[:, 1:] * field[:, :-1] < 0
    vertical = field[1:, :] * field[:-1, :] < 0
    return float((np.mean(horizontal) + np.mean(vertical)) * 0.5)


def gradient_energy(field: np.ndarray) -> float:
    gy, gx = np.gradient(field)
    denom = float(np.mean(field**2)) + 1.0e-12
    return float((np.mean(gx**2) + np.mean(gy**2)) / denom)


def choose_common_matched_index(
    loaded: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray | None]],
) -> int | None:
    source_arrays = {key: value[0] for key, value in loaded.items()}
    poisson_sources = source_arrays["poisson"]
    matched_sources = all(
        sources.shape == poisson_sources.shape
        and np.allclose(sources, poisson_sources, rtol=0.0, atol=1.0e-6)
        for sources in source_arrays.values()
    )
    if not matched_sources:
        return None

    poisson_solutions = loaded["poisson"][1]
    advdiff_solutions = loaded["advdiff"][1]
    helmholtz_solutions = loaded["helmholtz"][1]
    advdiff_tensor = loaded["advdiff"][2]
    helmholtz_tensor = loaded["helmholtz"][2]

    finite = np.ones(poisson_sources.shape[0], dtype=bool)
    for sources, solutions, _ in loaded.values():
        finite &= np.isfinite(sources).all(axis=(1, 2))
        finite &= np.isfinite(solutions).all(axis=(1, 2))

    source_score = score01(robust_amplitude_batch(poisson_sources))
    poisson_score = score01(robust_amplitude_batch(poisson_solutions))

    adv_shape = np.array(
        [
            0.45 * centroid_distance(field)
            + 0.35 * anisotropy(field)
            + 0.20 * min(skewness_abs(field), 3.0) / 3.0
            for field in advdiff_solutions
        ]
    )
    adv_score = score01(adv_shape)
    if advdiff_tensor is not None and advdiff_tensor.shape[1] >= 5:
        adv_mag = np.linalg.norm(advdiff_tensor[:, 3:5], axis=1)
        diff_mag = np.linalg.norm(advdiff_tensor[:, :3], axis=1)
        adv_ratio = np.log1p(adv_mag / (diff_mag + 1.0e-12))
        adv_score = 0.55 * adv_score + 0.45 * score01(adv_ratio)

    helm_crossings = np.array(
        [zero_crossing_fraction(field) for field in helmholtz_solutions]
    )
    helm_gradients = np.array(
        [gradient_energy(field) for field in helmholtz_solutions]
    )
    helm_oscillation = helm_crossings + 0.15 * score01(helm_gradients)
    helm_score = score01(helm_oscillation)
    if helmholtz_tensor is not None and helmholtz_tensor.shape[1] >= 2:
        helm_score = 0.7 * helm_score + 0.3 * score01(helmholtz_tensor[:, 1])

    combined = (
        0.18 * source_score
        + 0.16 * poisson_score
        + 0.40 * adv_score
        + 0.26 * helm_score
    )
    candidates = np.flatnonzero(finite)
    if candidates.size == 0:
        return None
    return int(candidates[np.argmax(combined[candidates])])


def choose_representative_samples(
    dataset_paths: dict[str, Path],
) -> dict[str, tuple[int, np.ndarray, np.ndarray]]:
    loaded = {
        key: load_dataset(path)
        for key, path in dataset_paths.items()
    }
    common_idx = choose_common_matched_index(loaded)
    if common_idx is not None:
        return {
            key: (
                common_idx,
                _as_float_array(sources[common_idx]),
                _as_float_array(solutions[common_idx]),
            )
            for key, (sources, solutions, _) in loaded.items()
        }

    candidates = {
        key: candidate_indices(sources, solutions)
        for key, (sources, solutions, _) in loaded.items()
    }

    best_score = float("inf")
    best_indices: dict[str, int] | None = None
    poisson_sources, _, _ = loaded["poisson"]
    advdiff_sources, _, _ = loaded["advdiff"]
    helmholtz_sources, _, _ = loaded["helmholtz"]

    for p_idx in candidates["poisson"]:
        p_source = poisson_sources[p_idx]

        adv_scores = [
            source_distance(p_source, advdiff_sources[a_idx])
            for a_idx in candidates["advdiff"]
        ]
        helm_scores = [
            source_distance(p_source, helmholtz_sources[h_idx])
            for h_idx in candidates["helmholtz"]
        ]
        adv_pos = int(np.argmin(adv_scores))
        helm_pos = int(np.argmin(helm_scores))
        score = adv_scores[adv_pos] + helm_scores[helm_pos]

        if score < best_score:
            best_score = score
            best_indices = {
                "poisson": int(p_idx),
                "advdiff": int(candidates["advdiff"][adv_pos]),
                "helmholtz": int(candidates["helmholtz"][helm_pos]),
            }

    if best_indices is None:
        raise ValueError("Could not select representative PDE samples")

    samples: dict[str, tuple[int, np.ndarray, np.ndarray]] = {}
    for key, idx in best_indices.items():
        sources, solutions, _ = loaded[key]
        samples[key] = (
            idx,
            _as_float_array(sources[idx]),
            _as_float_array(solutions[idx]),
        )
    return samples


def robust_limits(field: np.ndarray) -> tuple[float, float]:
    lo, hi = np.percentile(field, [1.0, 99.0])
    if not np.isfinite(lo) or not np.isfinite(hi) or np.isclose(lo, hi):
        lo = float(np.min(field))
        hi = float(np.max(field))
    if np.isclose(lo, hi):
        pad = max(abs(float(lo)) * 0.05, 1.0e-6)
        lo -= pad
        hi += pad
    return float(lo), float(hi)


def norm_for_field(field: np.ndarray) -> Normalize:
    lo, hi = robust_limits(field)
    if lo < 0 < hi:
        bound = max(abs(lo), abs(hi))
        return TwoSlopeNorm(vmin=-bound, vcenter=0.0, vmax=bound)
    return Normalize(vmin=lo, vmax=hi)


def shared_symmetric_norm(fields: list[np.ndarray]) -> TwoSlopeNorm:
    bound = 0.0
    for field in fields:
        lo, hi = robust_limits(field)
        bound = max(bound, abs(lo), abs(hi))
    bound = max(bound, 1.0e-6)
    return TwoSlopeNorm(vmin=-bound, vcenter=0.0, vmax=bound)


def style_axis(ax: plt.Axes) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.45)
        spine.set_color("0.72")


def add_row_header(fig: plt.Figure, y: float, label: str) -> None:
    fig.text(
        0.5,
        y,
        label,
        ha="center",
        va="center",
        fontsize=8.3,
        color="0.2",
    )
    for x0, x1 in ((0.07, 0.365), (0.635, 0.93)):
        line = plt.Line2D(
            [x0, x1],
            [y, y],
            transform=fig.transFigure,
            color="0.82",
            linewidth=0.45,
            solid_capstyle="butt",
        )
        fig.add_artist(line)


def plot_combined_figure(
    samples: dict[str, tuple[int, np.ndarray, np.ndarray]],
    figure_dir: Path,
    dpi: int,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.titlesize": 9,
            "figure.titlesize": 10,
            "mathtext.fontset": "dejavuserif",
        }
    )

    fig = plt.figure(figsize=(6.35, 3.55), constrained_layout=False)
    grid = fig.add_gridspec(
        2,
        3,
        left=0.035,
        right=0.995,
        bottom=0.045,
        top=0.755,
        hspace=0.18,
        wspace=0.075,
    )
    axes = np.array(
        [
            [fig.add_subplot(grid[0, col]) for col in range(3)],
            [fig.add_subplot(grid[1, col]) for col in range(3)],
        ],
        dtype=object,
    )
    cmap = "RdBu_r"
    source_norm = shared_symmetric_norm(
        [samples[spec.key][1] for spec in PDE_SPECS]
    )
    # Solution panels use independent symmetric normalization. The figure is
    # for operator intuition, so visibility matters more than amplitude comparison.
    solution_norms = {
        spec.key: norm_for_field(samples[spec.key][2])
        for spec in PDE_SPECS
    }

    top_row = axes[0, 1].get_position()
    bottom_row = axes[1, 1].get_position()
    add_row_header(fig, top_row.y1 + 0.027, r"Forcing field $f(x)$")
    add_row_header(fig, 0.5 * (top_row.y0 + bottom_row.y1), r"Solution field $u(x)$")

    for col, spec in enumerate(PDE_SPECS):
        _, source, solution = samples[spec.key]
        col_box = axes[0, col].get_position()
        col_center = 0.5 * (col_box.x0 + col_box.x1)
        fig.text(col_center, 0.94, spec.name, ha="center", va="top", fontsize=10.5)
        fig.text(col_center, 0.872, spec.equation, ha="center", va="top", fontsize=10)
        for row, (field, norm) in enumerate(
            ((source, source_norm), (solution, solution_norms[spec.key]))
        ):
            ax = axes[row, col]
            ax.imshow(
                field,
                cmap=cmap,
                norm=norm,
                origin="lower",
                interpolation="bilinear",
            )
            style_axis(ax)

    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_dir / "pde_task_examples.pdf", bbox_inches="tight")
    fig.savefig(figure_dir / "pde_task_examples.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_individual_field(
    field: np.ndarray,
    title: str,
    output_path: Path,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(2.2, 2.2), constrained_layout=True)
    image = ax.imshow(
        field,
        cmap="RdBu_r",
        norm=norm_for_field(field),
        origin="lower",
        interpolation="bilinear",
    )
    style_axis(ax)
    ax.set_title(title, pad=5)
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.025)
    cbar.ax.tick_params(labelsize=7, length=2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    png_path = output_path.with_suffix(".png")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_individual_figures(
    samples: dict[str, tuple[int, np.ndarray, np.ndarray]],
    figure_dir: Path,
    dpi: int,
) -> None:
    for spec in PDE_SPECS:
        _, source, solution = samples[spec.key]
        plot_individual_field(
            source,
            rf"{spec.name}: source $f(x)$",
            figure_dir / f"{spec.individual_prefix}_source.pdf",
            dpi,
        )
        plot_individual_field(
            solution,
            rf"{spec.name}: solution $u(x)$",
            figure_dir / f"{spec.individual_prefix}_solution.pdf",
            dpi,
        )


def main() -> None:
    args = parse_args()
    dataset_paths = ensure_datasets(args)

    samples = choose_representative_samples(dataset_paths)
    for spec in PDE_SPECS:
        idx, _, _ = samples[spec.key]
        print(
            f"{spec.name}: selected sample {idx} from "
            f"{display_path(dataset_paths[spec.key])}"
        )

    plot_combined_figure(samples, args.figure_dir, args.dpi)
    plot_individual_figures(samples, args.figure_dir, args.dpi)
    print(f"Saved figures to {display_path(args.figure_dir)}")


if __name__ == "__main__":
    main()
