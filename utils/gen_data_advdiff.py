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
from utils.misc_utils import show, fft_coef, grad, div
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
    dc = np.mean(source.flatten())
    source = source - dc # remove integral of source

    return source

def get_random_diffusion_tensor():
    ''' create a random diff tensor by controlling eigenvalues '''
    e1 = 1
    e2 = 5
    a1 = 1
    a4 = e1 + np.random.rand() * (e2 - e1) # random btw eigvals e1 and e2
    
    theta = np.random.rand() * 2 * np.pi # random rotation
    rot = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]]) 
    theta_neg = -1 * theta
    rot_neg = np.array([[np.cos(theta_neg),-np.sin(theta_neg)],[np.sin(theta_neg),np.cos(theta_neg)]])
    A = np.array([[a1, 0], [0, a4]])
    A = rot_neg @ (A @ rot) # similarity transf to preserve eigs
    
    return A

def get_random_velocity_vector():
    ''' create a random velocity vector by sampling on a circle '''
    theta = np.random.rand() * 2 * np.pi
    r = 1
    return [r*np.cos(theta), r*np.sin(theta)]

def advection_op(u, v, nx, ny, params):
    ''' does v.grad(u) '''
    u = u.reshape((nx, ny))    
    gradux, graduy = grad(u, params)
    return (v['v1']*gradux + v['v2']*graduy)

def diffusion_op(u, k, K, nx, ny, params):
    ''' does div(Kgradu) '''
    u = u.reshape((nx, ny))    
    gradux, graduy = grad(u, params)
    # diff tensor
    Kux = K['k11']*gradux + K['k12']*graduy
    Kuy = K['k22']*graduy + K['k12']*gradux
    # heterogeneous
    Kux *= k
    Kuy *= k
    return div(Kux, Kuy, params)

def advdiff(x, y, std, space, vf, params):
    """ -v.\gradu + \lapu = -f """
    k_mat = get_random_diffusion_tensor()       
    K = {'k11': k_mat[0,0], 'k22': k_mat[1,1], 'k12': k_mat[0,1]}
    vel = get_random_velocity_vector()
    v = {'v1': vel[0], 'v2': vel[1]} # just for readability make a dict

    ad1 = params.ad1
    ad2 = params.ad2
    adr = ad1 + np.random.rand() * (ad2 - ad1)
    # sample lamda based on adr
    lams = np.load("utils/lambda.npy")
    ads = np.load("utils/ads.npy")
    means = np.mean(ads, axis=1)
    lam = lams[closest(means, adr)]

    params.lam = lam

    x_g, y_g = np.meshgrid(x, y)

    # create a source function
    ng = params.ng
    p = np.zeros(ng)
        
    min_act = 1E-3 # avoid zeros
    max_act = 1
        
    for i in range(ng):
        alpha = np.random.rand()
        if alpha > vf:
            p[i] = (max_act - min_act) * np.random.rand() + min_act

    all_zeros = not np.any(p)
    if all_zeros:
        randidx = np.random.randint(0, len(p))
        p[randidx] = (max_act - min_act) * np.random.rand() + min_act

    
    source = rbf(x_g, y_g, p, ng=ng, sigma=std, spacing=space) # linear combination of rbfs

    nx = x.shape[0]
    ny = y.shape[0]
    ikx = fft_coef(nx).reshape(1,nx)
    ikx = np.repeat(ikx, ny, axis=0)
    iky = fft_coef(ny).reshape(ny,1)
    iky = np.repeat(iky, nx, axis=1)
    ikx2 = ikx**2
    iky2 = iky**2
    

    f_hat = np.fft.fft2(source) # RHS
    
    diff_factor = ikx2*K['k11'] + iky2*K['k22'] + 2*ikx*iky*K['k12']
    diff_factor *= (4.0 * np.pi**2) / (params.Lx * params.Ly) # not a 2pi domain, but unit cube
    adv_factor = v['v1']*ikx + v['v2']*iky
    adv_factor *= (2.0 * np.pi) / (params.Lx)  # implicity assumed Lx = Ly; TODO: be careful
    factor = (1 - params.lam) * params.diff_coef_scale * diff_factor - params.lam * params.adv_coef_scale * adv_factor

    factor = np.where(factor == 0, 0, -1/factor) # zeroth mode; set to zero 
    u_hat = factor * f_hat
    
    u = np.real(np.fft.ifft2(u_hat))
    u = u - np.mean(u.flatten())  # remove the dc component

    au = (params.lam) * params.adv_coef_scale * advection_op(u, v, nx, ny, params)
    du = (1 - params.lam) * params.diff_coef_scale * diffusion_op(u, 1, K, nx, ny, params)
    au = np.linalg.norm(au)
    du = np.linalg.norm(du)
    ratio = au/du
    
     
    k = np.array([K['k11'], K['k22'], K['k12']])
    v = np.array([v['v1'], v['v2']]) # for dataset creating and interfacing with training code

    return u, source, k, v, ratio


def create_hdf5(path, dat, ten, other):
    with h5py.File(path, "a") as f:
        try:
            f.create_dataset('fields', dat.shape, dtype='<f4', data=dat)
            f.create_dataset('tensor', ten.shape, dtype='<f4', data=ten)
            f.create_dataset('other', other.shape, dtype='<f4', data=other)
        except:
            del f['fields']
            del f['tensor']
            del f['other']
            f.create_dataset('fields', dat.shape, dtype='<f4', data=dat)
            f.create_dataset('tensor', ten.shape, dtype='<f4', data=ten)
            f.create_dataset('other', other.shape, dtype='<f4', data=other)


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


def create_hdf5_datasets(path, n_samples, nx, ny, tensor_dim, other_dim, chunk_samples):
    f = h5py.File(path, "w")
    field_chunks = _chunk_shape(n_samples, chunk_samples, 2, nx, ny)
    tensor_chunks = _chunk_shape(n_samples, chunk_samples, tensor_dim)
    other_chunks = _chunk_shape(n_samples, chunk_samples, other_dim)

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
    other_ds = f.create_dataset(
        'other',
        shape=(n_samples, other_dim),
        dtype='<f4',
        chunks=other_chunks,
    )
    return f, fields_ds, tensor_ds, other_ds

def num_ex(ntrain, idx, num_vfs):
    nex = (ntrain//num_vfs if idx is not num_vfs-1 else ntrain - (num_vfs-1)*(ntrain//num_vfs))
    return nex

def closest(lst, k):
    lst = np.asarray(lst)
    idx = (np.abs(lst - k)).argmin()
    return idx

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", default=100, type=int, help="number of training examples")
    parser.add_argument("--nval", default=25, type=int, help="number of validation examples")
    parser.add_argument("--ntest", default=25, type=int, help="number of testing examples")
    parser.add_argument("--n", default=128, type=int, help="grid size nxn")
    parser.add_argument("--ng", default=144, type=int, help="number of gaussians in grid")
    parser.add_argument("--sparse", action='store_true', help="creates sparse gaussians")
    parser.add_argument("--datapath", default="./", type=str, help="path to root dir to store data")
    parser.add_argument("--adr1", default=0.2, type=float, help="sample AD ratios starting from adr1")
    parser.add_argument("--adr2", default=1, type=float, help="sample AD ratios  ending at adr2")
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
    params["Lx"] = lx
    params["Ly"] = ly
    params["adv_coef_scale"] = 1
    params["diff_coef_scale"] = 0.01
    params['ad1'] = args.adr1
    params['ad2'] = args.adr2
    params["ng"] = args.ng
    params['sparse'] = args.sparse
    params = SimpleNamespace(**params)
    std = 1/32
    space = 2*std
    x_g, y_g = np.meshgrid(x, y)

    os.makedirs(args.datapath, exist_ok=True)
    train_path = os.path.join(args.datapath, "_train_adr{}_{}_32k.h5".format(params.ad1, params.ad2))
    val_path = os.path.join(args.datapath, "_val_adr{}_{}_4k.h5".format(params.ad1, params.ad2))
    test_path = os.path.join(args.datapath, "_test_adr{}_{}_4k.h5".format(params.ad1, params.ad2))

    train_file, train_fields_ds, train_tensor_ds, train_other_ds = create_hdf5_datasets(
        train_path, ntrain, x.shape[0], y.shape[0], 5, 1, args.h5_chunk_samples
    )
    val_file, val_fields_ds, val_tensor_ds, val_other_ds = create_hdf5_datasets(
        val_path, nval, x.shape[0], y.shape[0], 5, 1, args.h5_chunk_samples
    )
    test_file, test_fields_ds, test_tensor_ds, test_other_ds = create_hdf5_datasets(
        test_path, ntest, x.shape[0], y.shape[0], 5, 1, args.h5_chunk_samples
    )

    vfs = [0.2, 0.4, 0.6, 0.8]
    num_vfs = len(vfs)
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
                u, source, k, v, ratios = advdiff(x, y, std, space, vf, params)
                train_fields_ds[sim_train, 0] = source
                train_fields_ds[sim_train, 1] = u
                train_tensor_ds[sim_train, 0:3] = k * params.diff_coef_scale * (1 - params.lam)
                train_tensor_ds[sim_train, 3:5] = v * params.adv_coef_scale * params.lam
                train_other_ds[sim_train] = ratios
                sim_train += 1
                if sim_train % args.h5_chunk_samples == 0:
                    train_file.flush()
                if args.progress_every > 0 and ((sim_train - train_last_report) >= args.progress_every or sim_train == ntrain):
                    _report_progress("train", sim_train, ntrain, train_start)
                    train_last_report = sim_train

            nex = num_ex(nval, idx, num_vfs)
            print("For vf {}, there are {} val examples".format(vf, nex))
            for _ in range(nex):
                u, source, k, v, ratios = advdiff(x, y, std, space, vf, params)
                val_fields_ds[sim_val, 0] = source
                val_fields_ds[sim_val, 1] = u
                val_tensor_ds[sim_val, 0:3] = k * params.diff_coef_scale * (1 - params.lam)
                val_tensor_ds[sim_val, 3:5] = v * params.adv_coef_scale * params.lam
                val_other_ds[sim_val] = ratios
                sim_val += 1
                if sim_val % args.h5_chunk_samples == 0:
                    val_file.flush()
                if args.progress_every > 0 and ((sim_val - val_last_report) >= args.progress_every or sim_val == nval):
                    _report_progress("val", sim_val, nval, val_start)
                    val_last_report = sim_val

            nex = num_ex(ntest, idx, num_vfs)
            print("For vf {}, there are {} test examples".format(vf, nex))
            for _ in range(nex):
                u, source, k, v, ratios = advdiff(x, y, std, space, vf, params)
                test_fields_ds[sim_test, 0] = source
                test_fields_ds[sim_test, 1] = u
                test_tensor_ds[sim_test, 0:3] = k * params.diff_coef_scale * (1 - params.lam)
                test_tensor_ds[sim_test, 3:5] = v * params.adv_coef_scale * params.lam
                test_other_ds[sim_test] = ratios
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
