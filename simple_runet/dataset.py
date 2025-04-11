import os
import torch
import torch.nn as nn
import numpy as np

from os.path import join
from tqdm import tqdm
from torch.utils.data import Dataset

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from threading import Lock


class DatasetCase1(Dataset):
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


class DatasetCase2(Dataset): # (512, 128, 1) - 2D reservoir
    def __init__(self, sample_index, dir_to_database, timestep, pred_length=8):
        self.sample_index = sample_index
        self.dir_to_database = dir_to_database
        self.timestep = timestep
        self.pred_length = pred_length
        self.total_step = len(timestep)

        # create a dictionary of item keys
        self.precomputed_keys = {}
        self._precompute_keys()
        self.nsample = len(self.precomputed_keys)

        # Load data in parallel
        self.M, self.S, self.P, self.C, self.indices = self.load_data_in_parallel()
    
    def __len__(self):
        return self.nsample
    
    def _precompute_keys(self):
        pred_length = self.pred_length
        total_step = self.total_step
        idx = 0
        for sample_index, real_index in enumerate(self.sample_index):
            for step_start in range(total_step - pred_length):
                step_index = list(range(step_start, step_start + self.pred_length + 1)) # [initial step + prediction steps]
                # Store the precomputed information in the dictionary
                self.precomputed_keys[idx] = (sample_index, real_index, step_index)
                idx += 1

    def _load_data(self, idx):
        perm  = np.load(os.path.join(self.dir_to_database, f'real{idx}_perm_real.npy'))
        plume = np.load(os.path.join(self.dir_to_database, f'real{idx}_plume3d.npy'))[self.timestep]
        press = np.load(os.path.join(self.dir_to_database, f'real{idx}_press3d.npy'))[self.timestep]
        cntrl = np.load(os.path.join(self.dir_to_database, f'real{idx}_input_control.npy'))[:len(self.timestep[1:]),:]
        return perm, plume, press, cntrl, idx

    def load_data_in_parallel(self):
        perm, plume, press, cntrl, indices = [], [], [], [], []
        
        lock = Lock()
        with tqdm(total=len(self.sample_index)) as pbar:
            def task_with_progress(index):
                result = self._load_data(index)
                with lock:
                    pbar.update(1)
                return result
        
            with ThreadPoolExecutor() as executor:
                results = list(executor.map(task_with_progress, self.sample_index))
                results = sorted(results, key=lambda x: x[4])
        
                for reali in results:
                    _perm, _plume, _press, _cntrl, _index = reali
                    perm.append(_perm)
                    plume.append(_plume)
                    press.append(_press)
                    cntrl.append(_cntrl)
                    indices.append(_index)
                
        # Convert to tensors and preprocess
        perm  = np.array(perm).transpose((0, 1, 4, 2, 3))[:,:,:,4:-4,:]
        nlogk = (np.log10(perm) - np.log10(1.0)) / np.log10(2000)
        plume = np.array(plume)[:,:,:,4:-4,:]    
        press = (np.array(press)[:,:,:,4:-4,:] - 9810) / (33110 - 9810)
        cntrl = np.array(cntrl)
        indices = np.array(indices)
        
        data = (torch.tensor(nlogk, dtype=torch.float32), # m
                torch.tensor(plume, dtype=torch.float32), # s
                torch.tensor(press, dtype=torch.float32), # p
                torch.tensor(cntrl, dtype=torch.float32), # c
                indices
               )
        return data
    
    def __getitem__(self, idx):
        sample_index, real_index, step_index = self.precomputed_keys[idx]
                
        s0 = self.S[sample_index, step_index[0:1]]
        p0 = self.P[sample_index, step_index[0:1]]
        st = self.S[sample_index, step_index[1:]].unsqueeze(dim=1)
        pt = self.P[sample_index, step_index[1:]].unsqueeze(dim=1)

        static = self.M[sample_index]
        states = torch.cat((p0, s0), dim=0)
        contrl = torch.zeros(st.size(), dtype=torch.float32)
        contrl[:, :, 100, 255, 0] = 1
        output = torch.cat((pt, st), dim=1)
        return (contrl, states, static), output
    

