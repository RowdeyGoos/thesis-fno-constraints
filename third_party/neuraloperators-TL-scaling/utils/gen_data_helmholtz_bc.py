import os
import sys
import time
import argparse
import h5py
import numpy as np
from types import SimpleNamespace

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../')
from utils.bc_sampling import sample_bc_pair
from utils.fd_bc_utils import (
    rbf_source,
    sample_rbf_weights,
    solve_dirichlet_fd,
)


def _num_ex(total, idx, num_bins):
    if idx != num_bins - 1:
        return total // num_bins
    return total - (num_bins - 1) * (total // num_bins)


def _chunk_shape(n_samples, chunk_samples, *tail_shape):
    if n_samples <= 0:
        return None
    return (max(1, min(int(chunk_samples), int(n_samples))), *tail_shape)


def _report_progress(split_name, done, total, start_time):
    elapsed = max(time.time() - start_time, 1.0e-9)
    rate = done / elapsed
    remaining = max(total - done, 0)
    eta = (remaining / rate) if rate > 0 else float("inf")
    pct = (100.0 * done / max(total, 1))
    print(
        f"[{split_name}] {done}/{total} ({pct:5.1f}%) | "
        f"{rate:7.2f} samples/s | elapsed {elapsed:8.1f}s | eta {eta:8.1f}s",
        flush=True,
    )


def _interior_max_abs(u):
    if u.shape[0] <= 2 or u.shape[1] <= 2:
        return float(np.max(np.abs(u)))
    return float(np.max(np.abs(u[1:-1, 1:-1])))


def _create_hdf5_datasets(path, n_samples, nx, ny, tensor_dim, chunk_samples):
    f = h5py.File(path, "w")
    field_chunks = _chunk_shape(n_samples, chunk_samples, 2, nx, ny)
    tensor_chunks = _chunk_shape(n_samples, chunk_samples, tensor_dim)
    bc_chunks = _chunk_shape(n_samples, chunk_samples, 2, nx, ny)

    # Stored tensor layout: [diffusion_scale, omega]
    fields_ds = f.create_dataset(
        "fields",
        shape=(n_samples, 2, nx, ny),
        dtype="<f4",
        chunks=field_chunks,
    )
    tensor_ds = f.create_dataset(
        "tensor",
        shape=(n_samples, tensor_dim),
        dtype="<f4",
        chunks=tensor_chunks,
    )
    bc_ds = f.create_dataset(
        "bc",
        shape=(n_samples, 2, nx, ny),
        dtype="<f4",
        chunks=bc_chunks,
    )
    return f, fields_ds, tensor_ds, bc_ds


def _sample_helmholtz(xg, yg, vf, params, rng):
    omega = float(rng.integers(int(params.o1), int(params.o2) + 1))

    weights = sample_rbf_weights(
        rng=rng,
        ng=params.ng,
        vf=vf,
        sparse=params.sparse,
    )
    source = rbf_source(
        x=xg,
        y=yg,
        weights=weights,
        sigma=params.source_sigma,
        spacing=params.source_spacing,
        remove_mean=False,
    )

    g, m = sample_bc_pair(
        nx=xg.shape[0],
        ny=xg.shape[1],
        rng=rng,
        width=params.bc_width,
        n_modes=params.bc_modes,
        amplitude=params.bc_amplitude,
    )

    # Helmholtz track uses isotropic diffusion and sampled omega.
    k11 = params.diff_coef_scale
    k12 = 0.0
    k22 = params.diff_coef_scale
    u = solve_dirichlet_fd(
        source=source,
        g=g,
        lx=params.Lx,
        ly=params.Ly,
        k11=k11,
        k12=k12,
        k22=k22,
        omega=omega,
    )

    tensor = np.array([params.diff_coef_scale, omega], dtype=np.float32)
    bc = np.stack([g, m], axis=0).astype(np.float32)
    return u.astype(np.float32), source.astype(np.float32), tensor, bc


def _generate_split_to_hdf5(path, n_samples, xg, yg, params, rng, chunk_samples, split_name, progress_every):
    nx, ny = xg.shape
    h5_file, fields_ds, tensor_ds, bc_ds = _create_hdf5_datasets(
        path=path,
        n_samples=n_samples,
        nx=nx,
        ny=ny,
        tensor_dim=2,
        chunk_samples=chunk_samples,
    )

    try:
        split_start = time.time()
        last_report = 0
        if progress_every > 0:
            print(f"[{split_name}] starting generation ({n_samples} samples)", flush=True)
        vfs = [0.2, 0.4, 0.6, 0.8]
        num_vfs = len(vfs)
        sim = 0
        for idx, vf in enumerate(vfs):
            n_local = _num_ex(n_samples, idx, num_vfs)
            for _ in range(n_local):
                attempts = 0
                last_interior_max = float("nan")
                last_boundary_max = float("nan")
                while True:
                    attempts += 1
                    u, source, ten, bc_pair = _sample_helmholtz(xg=xg, yg=yg, vf=vf, params=params, rng=rng)
                    finite = np.all(np.isfinite(u)) and np.all(np.isfinite(source))
                    # Enforce amplitude bound on interior only; Dirichlet boundaries
                    # can naturally exceed this threshold due sampled boundary traces.
                    last_interior_max = _interior_max_abs(u)
                    last_boundary_max = float(np.max(np.abs(bc_pair[0])))
                    bounded = last_interior_max <= params.max_abs_solution
                    if finite and bounded:
                        break
                    if attempts >= params.max_sample_attempts:
                        raise RuntimeError(
                            "Failed to produce valid Helmholtz BC sample after "
                            f"{params.max_sample_attempts} attempts "
                            f"(last interior max={last_interior_max:.3f}, "
                            f"last boundary max={last_boundary_max:.3f}, "
                            f"max_abs_solution={params.max_abs_solution:.3f})"
                        )

                fields_ds[sim, 0] = source
                fields_ds[sim, 1] = u
                tensor_ds[sim] = ten
                bc_ds[sim] = bc_pair
                sim += 1

                if chunk_samples > 0 and (sim % chunk_samples == 0):
                    h5_file.flush()
                if progress_every > 0 and ((sim - last_report) >= progress_every or sim == n_samples):
                    _report_progress(split_name=split_name, done=sim, total=n_samples, start_time=split_start)
                    last_report = sim
        if progress_every > 0 and last_report != n_samples:
            _report_progress(split_name=split_name, done=n_samples, total=n_samples, start_time=split_start)
    finally:
        h5_file.close()


def _fmt(x):
    return str(x).replace('.', 'p')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", default=100, type=int)
    parser.add_argument("--nval", default=25, type=int)
    parser.add_argument("--ntest", default=25, type=int)
    parser.add_argument("--n", default=128, type=int, help="grid size nxn")
    parser.add_argument("--ng", default=144, type=int, help="number of RBF centers")
    parser.add_argument("--sparse", action='store_true', help="sample sparse RBF activations")
    parser.add_argument("--datapath", default="./", type=str, help="output directory")
    parser.add_argument("--o1", default=1, type=int, help="minimum omega")
    parser.add_argument("--o2", default=10, type=int, help="maximum omega")
    parser.add_argument("--diff_coef_scale", default=0.01, type=float)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--bc_modes", default=5, type=int)
    parser.add_argument("--bc_amplitude", default=1.0, type=float)
    parser.add_argument("--bc_width", default=1, type=int)
    parser.add_argument("--max_abs_solution", default=2.0, type=float)
    parser.add_argument("--max_sample_attempts", default=100, type=int, help="max retries for one accepted sample")
    parser.add_argument("--h5_chunk_samples", default=64, type=int, help="samples per HDF5 chunk/write flush")
    parser.add_argument("--progress_every", default=1000, type=int, help="print progress every N samples (<=0 disables)")
    args = parser.parse_args()

    if args.h5_chunk_samples <= 0:
        raise ValueError("--h5_chunk_samples must be >= 1")
    if args.max_sample_attempts <= 0:
        raise ValueError("--max_sample_attempts must be >= 1")

    rng = np.random.default_rng(args.seed)

    lx = ly = 1.0
    x = np.linspace(0.0, lx, args.n, dtype=np.float32)
    y = np.linspace(0.0, ly, args.n, dtype=np.float32)
    xg, yg = np.meshgrid(x, y, indexing='ij')

    params = {
        "Lx": lx,
        "Ly": ly,
        "ng": args.ng,
        "sparse": args.sparse,
        "o1": args.o1,
        "o2": args.o2,
        "diff_coef_scale": args.diff_coef_scale,
        "source_sigma": 1.0 / 32.0,
        "source_spacing": 2.0 / 32.0,
        "bc_modes": args.bc_modes,
        "bc_amplitude": args.bc_amplitude,
        "bc_width": args.bc_width,
        "max_abs_solution": args.max_abs_solution,
        "max_sample_attempts": args.max_sample_attempts,
    }
    params = SimpleNamespace(**params)

    t0 = time.time()
    os.makedirs(args.datapath, exist_ok=True)
    train_path = os.path.join(args.datapath, f"_train_o{_fmt(args.o1)}_{_fmt(args.o2)}_32k_bc.h5")
    val_path = os.path.join(args.datapath, f"_val_o{_fmt(args.o1)}_{_fmt(args.o2)}_4k_bc.h5")
    test_path = os.path.join(args.datapath, f"_test_o{_fmt(args.o1)}_{_fmt(args.o2)}_4k_bc.h5")

    _generate_split_to_hdf5(
        path=train_path,
        n_samples=args.ntrain,
        xg=xg,
        yg=yg,
        params=params,
        rng=rng,
        chunk_samples=args.h5_chunk_samples,
        split_name="train",
        progress_every=args.progress_every,
    )
    _generate_split_to_hdf5(
        path=val_path,
        n_samples=args.nval,
        xg=xg,
        yg=yg,
        params=params,
        rng=rng,
        chunk_samples=args.h5_chunk_samples,
        split_name="val",
        progress_every=args.progress_every,
    )
    _generate_split_to_hdf5(
        path=test_path,
        n_samples=args.ntest,
        xg=xg,
        yg=yg,
        params=params,
        rng=rng,
        chunk_samples=args.h5_chunk_samples,
        split_name="test",
        progress_every=args.progress_every,
    )
    print(f"Generated BC Helmholtz datasets in {args.datapath} (elapsed {time.time() - t0:.2f}s)")
