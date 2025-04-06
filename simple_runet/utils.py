import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.optim.lr_scheduler import StepLR
from os.path import join
from collections import OrderedDict
import h5py
from torch.utils.data import Dataset #, DataLoader
from tqdm import tqdm
import os
import json
from torch.utils.data import Dataset, DataLoader
import psutil


class RSE_loss(object):
    def __init__(self, p=2, d=None, split=False, weights=None):
        super(RSE_loss, self).__init__()
        #Dimension and Lp-norm type are postive
        assert p > 0
        self.d = d
        self.p = p
        if weights == None:
            weights = [1, 1]
        self.p_weight, self.s_weight = weights[0], weights[1]
        self.split = split

    def rel_square_error(self, y_pred, y):
        diff_norms = torch.norm(y_pred-y, p=self.p, dim=self.d)
        y_norms = torch.norm(y, p=self.p, dim=self.d)
        return torch.mean(diff_norms/y_norms)

    def __call__(self, y_pred, y):
        if self.split:
            p_pred, s_pred = torch.split(y_pred, (1, 1), dim=2)
            p_true, s_true = torch.split(y, (1, 1), dim=2)
            loss = self.p_weight*self.rel_square_error(p_pred, p_true)
            loss += self.s_weight*self.rel_square_error(s_pred, s_true)
        else:
            loss = self.rel_square_error(y_pred, y)
        return loss


class Dataset_Task4(Dataset):
    def __init__(self, folders, root_to_data, num_years=5, interval=1, total_step=61, dx=2, split_index=None):
        self.folders = folders
        self.dx = dx
        self.split_index = split_index
        self.total_step = total_step
        self.interval = interval
        self.num_steps = num_years * (4 // interval)
        self.root_to_data = root_to_data
        self.step_index = np.array(range(0, self.total_step, self.interval))[:self.num_steps+1]
        self.s, self.p, self.m = self._dataloader()
        self.len = self.m.shape[0]

    def __getitem__(self, index):
        sample = self.s[index], self.p[index], self.m[index]
        return sample

    def __len__(self):
        return self.len

    def _dataloader(self):
        
        plume, press, perm = [], [], []
        
        for folder in self.folders:
            path_to_data = join(self.root_to_data, folder)
            s = torch.load(join(path_to_data, 'processed_plume_all.pt'))[:, self.step_index, ...]
            p = torch.load(join(path_to_data, 'processed_pressure_all.pt'))[:, self.step_index, ...]
            k = torch.load(join(path_to_data, 'processed_static_all.pt'))[:, None]
            if self.dx is not None:
                s = s[:, :, self.dx:-self.dx, self.dx:-self.dx, :]
                p = p[:, :, self.dx:-self.dx, self.dx:-self.dx, :]
            if self.split_index is not None:
                s = s[self.split_index, ...]
                p = p[self.split_index, ...]
                k = k[self.split_index, ...]
            plume.append(s)
            press.append(p)
            perm.append(k)

        plume = torch.vstack(plume) 
        press = torch.vstack(press)
        perm  = torch.vstack(perm)
        perm = torch.nn.functional.pad(perm, pad=(0,0,2,2,2,2), mode='constant', value=0)

        return plume, press, perm
