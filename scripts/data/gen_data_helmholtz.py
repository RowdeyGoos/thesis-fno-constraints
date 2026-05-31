from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import os, sys, time
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import Normalize
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../')
from utils.misc_utils import show, fft_coef, grad, div, laplacian
from types import SimpleNamespace
import random
import h5py
import time

def rbf(x, y, p, sigma=1/64, spacing=2/64, ng=256, center=(0.5,0.5)):
    """ create an rbf grid basis functions """
    num = np.sqrt(ng) # num of centers in each direction
    l = (num - 1) * spacing # length of grid in each direction
    
    centers_x = np.arange(center[0]-l/2, center[0]+l/2+spacing, spacing)
    centers_y = np.arange(center[1]-l/2, center[1]+l/2+spacing, spacing)
    centers_x, centers_y = np.meshgrid(centers_x, centers_y)
    ratio = []
    c = []
    
    
    for cx, cy in zip(centers_x.flatten(), centers_y.flatten()):
        r = (x - cx)**2 + (y - cy)**2
        R = 2 * sigma**2
        ratio.append(r/R)
        c.append(1./(2*np.pi*sigma**2)) # normalizing factor

    source = 0*ratio[0]
    
    idx = 0
    for r, ci in zip(ratio, c):
        source += p[idx]*1*np.exp(-r)
        idx += 1

    source = source/np.max(source)
        
    return source

def helm_op(u, omega,  nx, ny, params):
    u = u.reshape((nx, ny))
    lap = params.diff_coef_scale * laplacian(u, params)
    lap += omega * u
    return lap


def helmholtz(x, y, std, space, vf, params):
    x_g, y_g = np.meshgrid(x, y)

    sol_max = 100
    while (sol_max > 2): # ignore solutions with large values
        ng = params.ng
        p = np.zeros(ng)
        min_act = 1E-3 # avoid zeros
        max_act = 1

        omega = np.random.randint(params.o1, params.o2+1)

        if not params.sparse:
            p = np.random.rand(ng,1)
        for i in range(ng):
            alpha = np.random.rand()
            if alpha > vf:
                p[i] = (max_act - min_act) * np.random.rand() + min_act

        all_zeros = not np.any(p)
        if all_zeros:
            randidx = np.random.randint(0, len(p))
            p[randidx] = (max_act - min_act) * np.random.rand() + min_act

        source = rbf(x_g, y_g, p, ng=ng, sigma=std, spacing=space)

        nx = x.shape[0]
        ny = y.shape[0]
        ikx = fft_coef(nx).reshape(1,nx)
        ikx = np.repeat(ikx, ny, axis=0)
        iky = fft_coef(ny).reshape(ny,1)
        iky = np.repeat(iky, nx, axis=1)
        ikx2 = ikx**2
        iky2 = iky**2

        f_hat = np.fft.fft2(source)

        ik_factor =  ikx2 + iky2
        ik_factor *= (4.0 * np.pi**2) / (params.Lx * params.Ly) * params.diff_coef_scale
        factor = (omega + ik_factor)
        condn = (factor == 0)

        f_hat = np.where(condn, 0, f_hat)
        source = source if np.all(~condn[:]) else np.real(np.fft.ifft2(f_hat))

        factor = np.where(condn, 0, -1/factor) # set to zero in undefined places in freq space
        u_hat = factor * f_hat

        u = np.real(np.fft.ifft2(u_hat))

        # check the result by seeing norm(LHS) == norm(RHS)
        lhs = helm_op(u, omega, nx, ny, params)
        lhs_norm = np.linalg.norm(lhs)
        rhs_norm = np.linalg.norm(source)
        if (np.abs(lhs_norm - rhs_norm) > 1E-5):
            print("INACCURATE SOLUTION!")

        sol_max = np.max(np.abs(u[:]))


    return u, source, omega


def create_hdf5(path, dat, ten):
    with h5py.File(path, "a") as f:
        try:
            f.create_dataset('fields', dat.shape, dtype='<f4', data=dat)
            f.create_dataset('tensor', ten.shape, dtype='<f4', data=ten)
        except:
            del f['fields']
            del f['tensor']
            f.create_dataset('fields', dat.shape, dtype='<f4', data=dat)
            f.create_dataset('tensor', ten.shape, dtype='<f4', data=ten)


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


def create_hdf5_datasets(path, n_samples, nx, ny, tensor_dim, chunk_samples):
    f = h5py.File(path, "w")
    field_chunks = _chunk_shape(n_samples, chunk_samples, 2, nx, ny)
    tensor_chunks = _chunk_shape(n_samples, chunk_samples, tensor_dim)

    fields_ds = f.create_dataset(
        'fields',
        shape=(n_samples, 2, nx, ny),
        dtype='<f4',
        chunks=field_chunks,
    )
    tensor_ds = f.create_dataset(
        'tensor',
        shape=(n_samples, tensor_dim),
        dtype='<f4',
        chunks=tensor_chunks,
    )
    return f, fields_ds, tensor_ds

def num_ex(ntrain, idx, num_vfs):
    nex = (ntrain//num_vfs if idx is not num_vfs-1 else ntrain - (num_vfs-1)*(ntrain//num_vfs))
    return nex

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", default=100, type=int, help="number of training examples")
    parser.add_argument("--nval", default=25, type=int, help="number of validation examples")
    parser.add_argument("--ntest", default=25, type=int, help="number of testing examples")
    parser.add_argument("--n", default=128, type=int, help="grid size nxn")
    parser.add_argument("--ng", default=144, type=int, help="number of gaussians in grid")
    parser.add_argument("--sparse", action='store_true', help="creates sparse gaussians")
    parser.add_argument("--datapath", default="./", type=str, help="path to root dir to store data")
    parser.add_argument("--o1", default=1, type=int, help="sample wavenumbers starting from o1")
    parser.add_argument("--o2", default=10, type=int, help="sample wavenumbers  ending at o2")
    parser.add_argument("--h5_chunk_samples", default=256, type=int, help="samples per HDF5 chunk/write flush")
    parser.add_argument("--progress_every", default=1000, type=int, help="print progress every N samples (<=0 disables)")

    args = parser.parse_args()
    print(args)
    if args.h5_chunk_samples <= 0:
        raise ValueError("--h5_chunk_samples must be >= 1")
    seed = 0

    random.seed(seed)
    np.random.seed(seed)

    ntrain = args.ntrain
    nval = args.nval
    ntest = args.ntest
    nx = ny = args.n
    lx = ly = 1
    dx = lx/nx
    dy = ly/ny
    x = np.arange(0, lx, dx)
    y = np.arange(0, ly, dy)
    params = {}
    params["diff_coef_freq"] = 0
    params["dc"] = 0
    params["Lx"] = lx
    params["Ly"] = ly

    params["diff_coef_scale"] = 0.01
    params["o1"] = args.o1
    params["o2"] = args.o2

    params["ng"] = args.ng
    params['sparse'] = args.sparse
    params = SimpleNamespace(**params)
    std = 1/32
    space = 2*std
    x_g, y_g = np.meshgrid(x, y)

    os.makedirs(args.datapath, exist_ok=True)
    train_path = os.path.join(args.datapath, "_train_o{}_{}_32k.h5".format(params.o1, params.o2))
    val_path = os.path.join(args.datapath, "_val_o{}_{}_4k.h5".format(params.o1, params.o2))
    test_path = os.path.join(args.datapath, "_test_o{}_{}_4k.h5".format(params.o1, params.o2))

    train_file, train_fields_ds, train_tensor_ds = create_hdf5_datasets(
        train_path, ntrain, x.shape[0], y.shape[0], 2, args.h5_chunk_samples
    )
    val_file, val_fields_ds, val_tensor_ds = create_hdf5_datasets(
        val_path, nval, x.shape[0], y.shape[0], 2, args.h5_chunk_samples
    )
    test_file, test_fields_ds, test_tensor_ds = create_hdf5_datasets(
        test_path, ntest, x.shape[0], y.shape[0], 2, args.h5_chunk_samples
    )

    vfs = [0.2, 0.4, 0.6, 0.8]
    num_vfs = len(vfs)
    in_distr = True
    t0 = time.time()
    train_start = val_start = test_start = t0
    train_last_report = val_last_report = test_last_report = 0
    sim_train = sim_val = sim_test = 0
    try:
        if args.progress_every > 0:
            print(f"[train] starting generation ({ntrain} samples)", flush=True)
            print(f"[val] starting generation ({nval} samples)", flush=True)
            print(f"[test] starting generation ({ntest} samples)", flush=True)

        for idx, vf in enumerate(vfs):
            nex = num_ex(ntrain, idx, num_vfs)
            print("For vf {}, there are {} train examples".format(vf, nex))
            for _ in range(nex):
                u, source, om = helmholtz(x, y, std, space, vf, params)
                train_fields_ds[sim_train, 0] = source
                train_fields_ds[sim_train, 1] = u
                train_tensor_ds[sim_train, 0] = params.diff_coef_scale
                train_tensor_ds[sim_train, 1] = om
                sim_train += 1
                if sim_train % args.h5_chunk_samples == 0:
                    train_file.flush()
                if args.progress_every > 0 and ((sim_train - train_last_report) >= args.progress_every or sim_train == ntrain):
                    _report_progress("train", sim_train, ntrain, train_start)
                    train_last_report = sim_train

            nex = num_ex(nval, idx, num_vfs)
            print("For vf {}, there are {} val examples".format(vf, nex))
            for _ in range(nex):
                u, source, om = helmholtz(x, y, std, space, vf, params)
                val_fields_ds[sim_val, 0] = source
                val_fields_ds[sim_val, 1] = u
                val_tensor_ds[sim_val, 0] = params.diff_coef_scale
                val_tensor_ds[sim_val, 1] = om
                sim_val += 1
                if sim_val % args.h5_chunk_samples == 0:
                    val_file.flush()
                if args.progress_every > 0 and ((sim_val - val_last_report) >= args.progress_every or sim_val == nval):
                    _report_progress("val", sim_val, nval, val_start)
                    val_last_report = sim_val

            nex = num_ex(ntest, idx, num_vfs)
            print("For vf {}, there are {} test examples".format(vf, nex))
            for _ in range(nex):
                u, source, om = helmholtz(x, y, std, space, vf, params)
                test_fields_ds[sim_test, 0] = source
                test_fields_ds[sim_test, 1] = u
                test_tensor_ds[sim_test, 0] = params.diff_coef_scale
                test_tensor_ds[sim_test, 1] = om
                sim_test += 1
                if sim_test % args.h5_chunk_samples == 0:
                    test_file.flush()
                if args.progress_every > 0 and ((sim_test - test_last_report) >= args.progress_every or sim_test == ntest):
                    _report_progress("test", sim_test, ntest, test_start)
                    test_last_report = sim_test

        train_file.flush()
        val_file.flush()
        test_file.flush()

        if args.progress_every > 0:
            if train_last_report != ntrain:
                _report_progress("train", ntrain, ntrain, train_start)
            if val_last_report != nval:
                _report_progress("val", nval, nval, val_start)
            if test_last_report != ntest:
                _report_progress("test", ntest, ntest, test_start)
    finally:
        train_file.close()
        val_file.close()
        test_file.close()

    print("time = {}".format(time.time() - t0))
    print("saved files to {}".format(args.datapath))
