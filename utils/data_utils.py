"""
  data loaders
"""
import re
import time
import os, sys
import logging
import h5py
import glob
import torch
import random
import numpy as np
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.utils.data.distributed import DistributedSampler


def _coerce_bool(val):
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.strip().lower() in ['1', 'true', 'yes', 'y', 'on']
    return bool(val)


def get_data_loader(params, location, distributed, train=True, pack=False):
    transform = torch.from_numpy
    dataset = PDESolns(params, location, transform, train)
    sampler = DistributedSampler(dataset, shuffle=train) if distributed else None
    if train:
        batch_size = params.local_batch_size
    else:
        batch_size = params.local_valid_batch_size
    if not pack:
        shuffle = bool(train and sampler is None and _coerce_bool(getattr(params, 'train_shuffle', False)))
        dataloader = DataLoader(dataset,
                                batch_size=int(batch_size),
                                num_workers=params.num_data_workers,
                                shuffle=shuffle,
                                sampler=sampler,
                                drop_last=True,
                                pin_memory=torch.cuda.is_available())
    else:
        # data is small, pack it all onto the gpu
        packed_x = []
        packed_y = []
        for idx in range(len(dataset)):
            x, y = dataset[idx]
            packed_x.append(x)
            packed_y.append(y)

        X = torch.stack(packed_x, dim=0).float().to(params.device)
        y = torch.stack(packed_y, dim=0).float().to(params.device)
        tensor_dataset = TensorDataset(X, y)
        dataloader = torch.utils.data.DataLoader(tensor_dataset, batch_size=int(batch_size), shuffle=True)
    return dataloader, dataset, sampler


class PDESolns(Dataset):
    def __init__(self, params, location, transform, train):
        self.transform = transform
        self.params = params
        self.location = location
        self.train = train
        self.random_train_subset = False
        self.sample_indices = None
        if hasattr(self.params, "subsample") and (self.train):
            self.subsample = self.params.subsample
        else:
            self.subsample = 1 # subsample only if training
        self.use_bc_channels = _coerce_bool(getattr(self.params, 'use_bc_channels', False))
        self.scales = None
        self._get_files_stats()
        file = self._open_file(self.location)
        self.data = file['fields']
        if 'tensor' in list(file.keys()):
            self.tensor = file['tensor']
        else:
            self.tensor = None
        if 'bc' in list(file.keys()):
            self.bc = file['bc']
        else:
            self.bc = None

    def _get_files_stats(self):
        self.file = self.location
        with h5py.File(self.file, 'r') as _f:
            logging.info("Getting file stats from {}".format(self.file))
            full_n_samples = _f['fields'].shape[0]
            self.n_samples = full_n_samples
            self.img_shape_x = _f['fields'].shape[2]
            self.img_shape_y = _f['fields'].shape[3]
            self.in_channels = _f['fields'].shape[1]-1
            if 'tensor' in list(_f.keys()):
                self.tensor_shape = _f['tensor'].shape[1]
            else:
                self.tensor_shape = 0
            self.has_bc = 'bc' in list(_f.keys())
            self.bc_channels = 0
            if self.has_bc:
                bc_shape = _f['bc'].shape
                expected_shape = (_f['fields'].shape[0], 2, self.img_shape_x, self.img_shape_y)
                if bc_shape != expected_shape:
                    raise ValueError(
                        "bc dataset in {} must have shape {}, got {}".format(
                            self.file, expected_shape, bc_shape
                        )
                    )
                self.bc_channels = 2
        target_n_samples = int(full_n_samples / self.subsample)
        if self.train and self.subsample > 1 and _coerce_bool(getattr(self.params, 'random_train_subset', False)):
            subset_seed = getattr(self.params, 'subset_seed', None)
            if subset_seed is None:
                subset_seed = getattr(self.params, 'seed', 0)
            rng = np.random.default_rng(subset_seed)
            self.sample_indices = np.sort(rng.choice(full_n_samples, size=target_n_samples, replace=False))
            self.n_samples = len(self.sample_indices)
            self.random_train_subset = True
            logging.info(
                "Using seeded random train subset: %d / %d samples (seed=%s)",
                self.n_samples,
                full_n_samples,
                str(subset_seed),
            )
        else:
            self.n_samples = target_n_samples
        if self.use_bc_channels and not self.has_bc:
            logging.warning(
                "use_bc_channels=true but dataset {} has no 'bc' key; continuing without BC channels.".format(
                    self.location
                )
            )
        input_channels = self.in_channels + self.tensor_shape
        if self.use_bc_channels and self.has_bc:
            input_channels += self.bc_channels
        logging.info("Found data at path {}. Number of examples: {}. Image Shape: {} x {}".format(self.location, self.n_samples, self.img_shape_x, self.img_shape_y))
        logging.info("Input channels from loader: {} (fields={} tensor={} bc={})".format(
            input_channels,
            self.in_channels,
            self.tensor_shape,
            (self.bc_channels if (self.use_bc_channels and self.has_bc) else 0),
        ))
        if hasattr(self.params, "scales_path"):
            self.scales = np.load(self.params.scales_path)
            self.scales = np.array([s if s != 0 else 1 for s in self.scales]) 
            self.scales = self.scales.astype('float32')
            measure_x = self.scales[-2] / self.img_shape_x
            measure_y = self.scales[-1] / self.img_shape_y
            self.measure = measure_x * measure_y
            logging.info("Scales for PDE are (source, tensor, sol, domain): {}".format(self.scales))
            logging.info("Measure of the set is lx/nx * ly/ny =  {}/{} * {}/{}".format(self.scales[-2], self.img_shape_x, self.scales[-1], self.img_shape_y))

    def __len__(self):
        return self.n_samples

    def _open_file(self, path):
        return h5py.File(path, 'r')

    def __getitem__(self, idx):
        if self.sample_indices is not None:
            local_idx = int(self.sample_indices[idx])
        else:
            local_idx = int(idx*self.subsample)
        X = (self.data[local_idx,0:self.in_channels])
        if self.tensor is not None: # append coefficient tensor to channels
            tensor = []
            for tidx in range(self.tensor_shape):
                coef = np.full((1, self.img_shape_x, self.img_shape_y), self.tensor[local_idx,tidx])
                tensor.append(coef)
            X = np.concatenate([X] + tensor, axis=0).astype('float32')
        else:
            X = X.astype('float32')

        if self.use_bc_channels and self.bc is not None:
            # BC convention is [g, m] where g is boundary value map and m is boundary mask.
            bc = self.bc[local_idx].astype('float32')
            X = np.concatenate([X, bc], axis=0).astype('float32')

        if self.scales is not None:
            f_norm = np.linalg.norm(X[0]) * self.measure
            f_scaling = f_norm / self.scales[0]
            # Keep existing normalization semantics: global source-based scaling first.
            # Tensor channels then receive an additional channel-wise scaling step below.
            X = X / f_scaling # ensures that 10f and 10k for example, have the same input
            # scale the tensors
            tensor_start = self.in_channels
            tensor_end = tensor_start + self.tensor_shape
            if self.tensor_shape > 0:
                X[tensor_start:tensor_end] = X[tensor_start:tensor_end] / self.scales[
                    self.in_channels:(self.in_channels + self.tensor_shape), None, None
                ]

        X = self.transform(X)
        y = self.transform(self.data[local_idx,self.in_channels:])
        return X, y
