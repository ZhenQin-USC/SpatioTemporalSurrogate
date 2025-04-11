import os
import torch
import torch.nn as nn
import numpy as np

from os.path import join
from tqdm import tqdm
from collections import defaultdict

from torch.optim.lr_scheduler import StepLR
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
    

class GeneralTrainer:
    def __init__(self, model, train_config, pixel_loss=None, regularizer=None, device=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        self.model = model.to(self.device)
        self.train_config = train_config
        self.num_epochs = train_config.get('num_epochs', 200)
        self.verbose = train_config.get('verbose', 1)
        self.learning_rate = train_config.get('learning_rate', 1e-4)
        self.step_size = train_config.get('step_size', 100)
        self.gamma = train_config.get('gamma', 0.95)
        self.weight_decay = train_config.get('weight_decay', 0.0)
        self.gradient_clip = train_config.get('gradient_clip', False)
        self.gradient_clip_val = train_config.get('gradient_clip_val', None)

        # Loss function
        self.loss_fn = nn.MSELoss() if pixel_loss is None else pixel_loss
        self.regularizer = regularizer
        self.regularizer_weight = train_config.get('regularizer_weight', None)

        # Optimizer and LR scheduler
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.scheduler = StepLR(self.optimizer, step_size=self.step_size, gamma=self.gamma)

    def forward(self, data):
        """
        Must be implemented by subclass. Should return:
        {
            'tensors': {...},    # any tensor-like outputs
            'losses': {...}      # all loss terms, including 'loss', 'loss_pixel', etc.
        }
        """
        raise NotImplementedError

    def load_checkpoint(self, path_to_ckpt):
        checkpoint = torch.load(join(path_to_ckpt, 'checkpoint.pt'), map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def save_checkpoint(self, epoch, path_to_model, loss_tracker_dict, learning_rate_list, save_epoch_ckpt=True):
        if save_epoch_ckpt:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }, join(path_to_model, f'checkpoint{epoch}.pt'))

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, join(path_to_model, 'checkpoint.pt'))

        # Save losses
        loss_np_dict = {k: np.array(v) for k, v in loss_tracker_dict.items()}
        loss_np_dict['learning_rate'] = np.array(learning_rate_list)
        np.savez(join(path_to_model, 'loss_epoch.npz'), **loss_np_dict)

    def _run_one_epoch(self, dataloader, epoch=None, train=True):
        mode = 'train' if train else 'valid'
        self.model.train() if train else self.model.eval()
        loop = tqdm(dataloader, desc=f"{mode.capitalize()}ing") if self.verbose else dataloader

        running_losses = defaultdict(float)
        num_batches = 0

        for data in loop:
            if train:
                self.optimizer.zero_grad()

            output = self.forward(data)
            losses = output['losses']

            loss = losses['loss']  # main loss
            if train:
                loss.backward()
                if self.gradient_clip and self.gradient_clip_val:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
                
                self.optimizer.step()
                self.scheduler.step()

            # Update statistics
            for k, v in losses.items():
                running_losses[k] += v.item()
            num_batches += 1

            if self.verbose:
                loop.set_postfix({k: v / num_batches for k, v in running_losses.items()})

        avg_losses = {k: v / num_batches for k, v in running_losses.items()}
        return avg_losses

    def train(self, train_loader, valid_loader, path_to_model, ckpt_epoch=5):
        loss_tracker_dict = defaultdict(list)
        learning_rate_list = []

        for epoch in range(self.num_epochs):
            train_losses = self._run_one_epoch(train_loader, epoch, train=True)
            valid_losses = self._run_one_epoch(valid_loader, epoch, train=False)

            for k, v in train_losses.items():
                loss_tracker_dict[f'train_{k}'].append(v)
            for k, v in valid_losses.items():
                loss_tracker_dict[f'valid_{k}'].append(v)
            learning_rate_list.append(self.optimizer.param_groups[0]['lr'])

            if self.verbose:
                train_loss_str = ' | '.join([f"{k}: {v:.4f}" for k, v in train_losses.items()])
                valid_loss_str = ' | '.join([f"{k}: {v:.4f}" for k, v in valid_losses.items()])
                print(f"Epoch {epoch+1:03d}: Train - {train_loss_str} | Valid - {valid_loss_str}")

            self.save_checkpoint(
                epoch=epoch,
                path_to_model=path_to_model,
                loss_tracker_dict=loss_tracker_dict,
                learning_rate_list=learning_rate_list,
                save_epoch_ckpt=((epoch + 1) % ckpt_epoch == 0)
            )

        return loss_tracker_dict

    def test(self, test_loader):
        self.model.eval()
        all_outputs = defaultdict(list)
        running_losses = defaultdict(float)
        num_batches = 0

        loop = tqdm(test_loader, desc="Testing") if self.verbose else test_loader

        for data in loop:
            with torch.no_grad():
                output = self.forward(data)

            for k, v in output['losses'].items():
                running_losses[k] += v.item()
            for k, v in output['tensors'].items():
                all_outputs[k].append(v.cpu())

            num_batches += 1
            if self.verbose:
                loop.set_postfix({k: f"{v / num_batches:.4f}" for k, v in running_losses.items()})

        averaged_losses = {k: v / num_batches for k, v in running_losses.items()}
        output_tensors = {k: torch.cat(v, dim=0) for k, v in all_outputs.items()}

        return {
            'losses': averaged_losses,
            'tensors': output_tensors
        }


class Trainer(GeneralTrainer):
    def __init__(self, model, train_config, pixel_loss=None, regularizer=None, device=None):
        super().__init__(model, train_config, pixel_loss, regularizer, device)

    def _extract_data_instance(self, data):
        _s, _p, _m = data
        states = torch.cat((_s[:, [0]], _p[:, [0]]), dim=1).to(self.device) # (B, 2, C, X, Y, Z)
        static = _m.to(self.device)
        outputs = torch.cat((_s[:, 1:][:, :, None], _p[:, 1:][:, :, None]), dim=2).to(self.device)
        return states, static, outputs

    def forward(self, data):
        states, static, outputs = self._extract_data_instance(data)
        preds = self.model(states, static)

        loss_pixel = self.loss_fn(preds, outputs)
        if self.regularizer is not None:
            loss_reg = self.regularizer_weight * self.regularizer(preds, outputs)
        else:
            loss_reg = torch.tensor(0.0, device=self.device)

        total_loss = loss_pixel + loss_reg

        return {
            'tensors': {
                'preds': preds,
                'outputs': outputs,
                'static': static,
                'states': states,
            },
            'losses': {
                'loss': total_loss,
                'loss_pixel': loss_pixel.detach(),
                'loss_auxillary': loss_reg.detach()
            }
        }


class TrainerCase1(GeneralTrainer):  # Trainer for Case 1 - 3D Dataset
    def __init__(self, model, train_config, wells=None, **kwargs):
        super().__init__(model, train_config, **kwargs)
        self.wells = wells or [(42, 42, 16), (42, 42, 17), (27, 27, 16), (27, 27, 17)]

    def _extract_data_instance(self, data):
        _s, _p, _m = data  # (s, p) ~ (B, T + 1, X, Y, Z), (m) ~ (B, 1, X, Y, Z)
        _u = torch.zeros_like(_s[:, 1:])  # (B, T, X, Y, Z)
        for ix, iy, iz in self.wells:
            _u[..., ix, iy, iz] = 1
        contrl = _u[:, :, None].to(self.device)  # (B, T, 1, X, Y, Z)
        states = torch.cat((_s[:, [0]], _p[:, [0]]), dim=1).to(self.device)  # (B, 2, X, Y, Z)
        static = _m.to(self.device)  # (B, 1, X, Y, Z)
        outputs = torch.cat((_s[:, 1:][:, :, None], _p[:, 1:][:, :, None]), dim=2).to(self.device)  # (B, T, 2, X, Y, Z)
        return contrl, states, static, outputs

    def forward(self, data):
        contrl, states, static, outputs = self._extract_data_instance(data)
        preds = self.model(contrl, states, static)

        loss_pixel = self.loss_fn(preds, outputs)
        loss_reg = self.regularizer(preds, outputs) if self.regularizer else torch.tensor(0.0, device=self.device)
        total_loss = loss_pixel + self.regularizer_weight * loss_reg

        return {
            'tensors': {
                'preds': preds.detach().cpu(),
                'outputs': outputs.detach().cpu(),
                'states': states.detach().cpu(),
                'static': static.detach().cpu()
            },
            'losses': {
                'loss': total_loss,
                'loss_pixel': loss_pixel.detach(),
                'loss_auxillary': loss_reg.detach()
            }
        }


class TrainerCase2(GeneralTrainer):  # Trainer for Case 2 - 2D Dataset
    def __init__(self, model, train_config, wells=None, **kwargs):
        super().__init__(model, train_config, **kwargs)

    def _extract_data_instance(self, data):
        inputs, outputs = data
        contrl, states, static = inputs
        contrl = contrl.to(self.device)
        states = states.to(self.device)
        static = static.to(self.device)
        outputs = outputs.to(self.device)
        return contrl, states, static, outputs

    def forward(self, data):
        contrl, states, static, outputs = self._extract_data_instance(data)
        preds = self.model(contrl, states, static)

        loss_pixel = self.loss_fn(preds, outputs)
        loss_reg = self.regularizer(preds, outputs) if self.regularizer else torch.tensor(0.0, device=self.device)
        total_loss = loss_pixel + self.regularizer_weight * loss_reg

        return {
            'tensors': {
                'preds': preds.detach().cpu(),
                'outputs': outputs.detach().cpu(),
                'states': states.detach().cpu(),
                'static': static.detach().cpu()
            },
            'losses': {
                'loss': total_loss,
                'loss_pixel': loss_pixel.detach(),
                'loss_auxillary': loss_reg.detach()
            }
        }


class Trainer_LSDA(GeneralTrainer):
    def _extract_data_instance(self, data):
        inputs, outputs = data
        contrl, _state, static = inputs
        contrl = contrl.to(self.device)
        static = static.to(self.device)
        states = torch.cat((_state.unsqueeze(dim=1), outputs), dim=1)[:,:,-1:,...].to(self.device)
        return contrl, states, static

    def forward(self, data):
        contrl, states, static = self._extract_data_instance(data)
        m_recon, x_recon, x_pred = self.model(contrl, states, static)

        m_recon_loss = self.loss_fn(m_recon, static)
        x_recon_loss = self.loss_fn(x_recon, states)
        x_pred_loss = self.loss_fn(x_pred, states[:,1:])
        total = m_recon_loss + x_recon_loss + x_pred_loss

        tensors = {
                'x_pred': x_pred.detach().cpu(),
                'x_recon': x_recon.detach().cpu(),
                'm_recon': m_recon.detach().cpu(),
                'static': static.detach().cpu(),
                'states': states.detach().cpu(),
                'contrl': contrl.detach().cpu(),
            }
        
        losses = {
            'loss': total,
            'm_recon_loss': m_recon_loss.detach(),
            'x_recon_loss': x_recon_loss.detach(),
            'x_pred_loss': x_pred_loss.detach(),
        }

        return {
            'tensors': tensors,
            'losses': losses
        }

