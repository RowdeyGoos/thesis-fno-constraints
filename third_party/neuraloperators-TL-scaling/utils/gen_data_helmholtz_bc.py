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


def _write_hdf5(path, fields, tensor, bc):
    with h5py.File(path, "a") as f:
        for key in ['fields', 'tensor', 'bc']:
            if key in f:
                del f[key]
        # Stored tensor layout: [diffusion_scale, omega]
        f.create_dataset('fields', fields.shape, dtype='<f4', data=fields)
        f.create_dataset('tensor', tensor.shape, dtype='<f4', data=tensor)
        f.create_dataset('bc', bc.shape, dtype='<f4', data=bc)


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


def _generate_split(n_samples, xg, yg, params, rng):
    fields = np.zeros((n_samples, 2, xg.shape[0], yg.shape[1]), dtype=np.float32)
    tensor = np.zeros((n_samples, 2), dtype=np.float32)
    bc = np.zeros((n_samples, 2, xg.shape[0], yg.shape[1]), dtype=np.float32)

    vfs = [0.2, 0.4, 0.6, 0.8]
    num_vfs = len(vfs)
    sim = 0
    for idx, vf in enumerate(vfs):
        n_local = _num_ex(n_samples, idx, num_vfs)
        for _ in range(n_local):
            attempts = 0
            while True:
                attempts += 1
                u, source, ten, bc_pair = _sample_helmholtz(xg=xg, yg=yg, vf=vf, params=params, rng=rng)
                finite = np.all(np.isfinite(u)) and np.all(np.isfinite(source))
                # Guard against rare unstable draws with very large amplitudes.
                bounded = np.max(np.abs(u)) <= params.max_abs_solution
                if finite and bounded:
                    break
                if attempts >= 10:
                    raise RuntimeError("Failed to produce valid Helmholtz BC sample after 10 attempts")

            fields[sim, 0] = source
            fields[sim, 1] = u
            tensor[sim] = ten
            bc[sim] = bc_pair
            sim += 1
    return fields, tensor, bc


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
    args = parser.parse_args()

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
    }
    params = SimpleNamespace(**params)

    t0 = time.time()
    train_fields, train_tensor, train_bc = _generate_split(args.ntrain, xg, yg, params, rng)
    val_fields, val_tensor, val_bc = _generate_split(args.nval, xg, yg, params, rng)
    test_fields, test_tensor, test_bc = _generate_split(args.ntest, xg, yg, params, rng)

    os.makedirs(args.datapath, exist_ok=True)
    _write_hdf5(
        os.path.join(args.datapath, f"_train_o{_fmt(args.o1)}_{_fmt(args.o2)}_32k_bc.h5"),
        train_fields,
        train_tensor,
        train_bc,
    )
    _write_hdf5(
        os.path.join(args.datapath, f"_val_o{_fmt(args.o1)}_{_fmt(args.o2)}_4k_bc.h5"),
        val_fields,
        val_tensor,
        val_bc,
    )
    _write_hdf5(
        os.path.join(args.datapath, f"_test_o{_fmt(args.o1)}_{_fmt(args.o2)}_4k_bc.h5"),
        test_fields,
        test_tensor,
        test_bc,
    )
    print(f"Generated BC Helmholtz datasets in {args.datapath} (elapsed {time.time() - t0:.2f}s)")
