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
from kornia.contrib import distance_transform


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


class Dataset_Task4_with_Boundary_Mask(Dataset_Task4):
    def __init__(self, folders, root_to_data, device, threshold = 1e-2, num_years=5, interval=1, total_step=61, dx=2, split_index=None):
        super().__init__(folders, root_to_data, num_years=num_years, interval=interval, total_step=total_step,
                         dx=dx, split_index=split_index)
        self.threshold = threshold
        self.device = device
        self.map = self._create_signed_distance_map()

    def _create_signed_distance_map(self):
        # Step 1: create binary mask
        B, T, X, Y, Z = self.s.shape
        s_reshaped = self.s.permute(0, 1, 4, 2, 3)  # (B, T, Z, X, Y)
        s_reshaped = s_reshaped.reshape(B, T * Z, X, Y)  # (B, T*Z, X, Y)
        image = s_reshaped.to(self.device)  # (B, T*Z, X, Y)
        bin_mask = (image >= self.threshold).float()

        # Step 2: compute unsigned distance maps
        phi_in = distance_transform(1.0 - bin_mask)  # outside: distance to foreground
        phi_out  = distance_transform(bin_mask)        # inside: distance to background

        # Step 3: combine into signed distance map
        signed_phi = phi_out.clone()
        signed_phi[bin_mask.bool()] = -phi_in[bin_mask.bool()]

        # Step 4: reshape to original dimensions
        signed_dist_map = signed_phi.reshape(B, T, Z, X, Y)  # (B, T, Z, X, Y)
        signed_dist_map = signed_dist_map.permute(0, 1, 3, 4, 2)  # (B, T, X, Y, Z)

        return signed_dist_map

    def __getitem__(self, index):
        sample = self.s[index], self.p[index], self.m[index], self.map[index]
        return sample


class Dataset_Task4_with_Contour_Mask(Dataset_Task4):
    def __init__(self, folders, root_to_data, device, thresholds = [1e-2], num_years=5, interval=1, total_step=61, dx=2, split_index=None):
        super().__init__(folders, root_to_data, num_years=num_years, interval=interval, total_step=total_step,
                         dx=dx, split_index=split_index)
        self.thresholds = thresholds
        self.device = device
        maps = [self._create_signed_distance_map(threshold) for threshold in self.thresholds]
        self.map = torch.cat(maps, dim=-1)  

    def _create_signed_distance_map(self, threshold):
        # Step 1: create binary mask
        B, T, X, Y, Z = self.s.shape
        s_reshaped = self.s.permute(0, 1, 4, 2, 3)  # (B, T, Z, X, Y)
        s_reshaped = s_reshaped.reshape(B, T * Z, X, Y)  # (B, T*Z, X, Y)
        image = s_reshaped.to(self.device)  # (B, T*Z, X, Y)
        bin_mask = (image >= threshold).float()

        # Step 2: compute unsigned distance maps
        phi_in = distance_transform(1.0 - bin_mask)  # outside: distance to foreground
        phi_out  = distance_transform(bin_mask)        # inside: distance to background

        # Step 3: combine into signed distance map
        signed_phi = phi_out.clone()
        signed_phi[bin_mask.bool()] = -phi_in[bin_mask.bool()]

        # Step 4: reshape to original dimensions
        signed_dist_map = signed_phi.reshape(B, T, Z, X, Y)  # (B, T, Z, X, Y)
        signed_dist_map = signed_dist_map.permute(0, 1, 3, 4, 2)  # (B, T, X, Y, Z)

        return signed_dist_map

    def __getitem__(self, index):
        sample = self.s[index], self.p[index], self.m[index], self.map[index]
        return sample


def load_data(path_to_data, p_postprocess=None):
    un = torch.load(join(path_to_data, 'processed_rate_all.pt'))[:, :, None]
    s = torch.load(join(path_to_data, 'processed_plume_all.pt'))[:, :, None]
    p = torch.load(join(path_to_data, 'processed_pressure_all.pt'))[:, :, None]
    static = torch.load(join(path_to_data, 'processed_static_all.pt'))

    if p_postprocess != None:
        p = p_postprocess(p)
    uw = torch.zeros_like(un, dtype=torch.float32)
    contrl = torch.cat((uw, un), dim=2)
    states = torch.cat((p, s), dim=2)
    return contrl, states, static


def memory_usage_psutil():
    # return the memory usage in percentage like top
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss/(1e3)**3
    print('Memory Usage in Gb: {:.2f}'.format(mem))  # in GB 
    return mem


def load_model(model, learning_rate, path_to_model):
    # Load model and optimizer state
    checkpoint = torch.load(path_to_model)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=250, gamma=0.975)
    return model, optimizer, scheduler


def save_model(model, model_name, save_path):
    ''' save the model '''
    torch.save(model.state_dict(), save_path + model_name + '.pt')


def extract_data_instance(data, device):
    _s, _p, _m = data
    _states = torch.cat((_s[:,[0]], _p[:,[0]]), dim=1).to(device)
    _static = _m.to(device)
    outputs = torch.cat((_s[:,1:][:,:,None], _p[:,1:][:,:,None]), dim=2).to(device)
    return _states, _static, outputs


def train_one_epoch(model, device, train_loader, epoch, NUM_EPOCHS, 
                    optimizer, scheduler, loss_fn, verbose=0, 
                    gradient_clip=False, gradient_clip_val=None):
    batch_loss = 0.
    # last_loss = 0.

    if verbose == 1:
        loop = tqdm(train_loader)   
    elif verbose == 0:
        loop = train_loader

    for i, data in enumerate(loop):
        # Data instance 
        _states, _static, outputs = extract_data_instance(data, device)

        # Zero gradients for every batch
        optimizer.zero_grad()

        # Make predictions for every batch
        preds = model(_states, _static)

        # Compute the loss and its gradients
        loss = loss_fn(preds, outputs)
        loss.backward()

        # Gradient Clip
        if gradient_clip:
            if gradient_clip_val==None:
                pass
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_val)

        # Updates the parameters
        optimizer.step()

        # Adjust learning weights
        scheduler.step()

        # Gather data and report
        batch_loss += loss.item()

        if verbose == 1:
            loop.set_description(f"Epoch [{epoch}/{NUM_EPOCHS}]")
            loop.set_postfix(loss=loss.item()) 
        elif verbose == 0:
            pass

    return batch_loss/(i+1), optimizer, scheduler


def validate_model(model, device, valid_loader, loss_fn, metrics):

    model.eval()
    running_vloss = 0.0
    i = 0
    running_metrics = []
    for i, vdata in enumerate(valid_loader):

        _metrics_val = []
        _states, _static, outputs = extract_data_instance(vdata, device)
        
        with torch.no_grad():
            preds = model(_states, _static)
            vloss = loss_fn(preds, outputs)
            
            for _metrics in metrics:
                metric = _metrics(preds, outputs)
                _metrics_val.append(metric.item())

            running_metrics.append(_metrics_val)

        running_vloss += vloss.item() #float(vloss)
    
    return running_vloss/(i+1), list(np.array(running_metrics).mean(axis=0))


def train(model, device, EPOCHS, train_loader, valid_loader, path_to_model, 
          learning_rate=1e-3, step_size=250, gamma=0.975, verbose=0, loss_fn = nn.MSELoss(),
          gradient_clip=False, gradient_clip_val=None, if_track_validate=True, weight_decay=0):
    
    metrics = [RSE_loss(p=1), RSE_loss(p=2), nn.MSELoss()]
    metrics_name = ['L1_RSE', 'L2_RSE', 'MSE']

    train_loss_list = []
    valid_loss_list = []
    learning_rate_list = []
    metrics_val = []

    epoch_number = 0

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma, verbose=False)

    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        
        # Training 
        model.train(True)
        batch_loss, optimizer, scheduler = train_one_epoch(
            model, device, train_loader, epoch, EPOCHS, optimizer, scheduler, loss_fn, 
            verbose=verbose, gradient_clip=gradient_clip, gradient_clip_val=gradient_clip_val
            )
        train_loss_list.append(batch_loss)
        if if_track_validate == True:
            # Validation
            batch_vloss, _metrics_val = validate_model(model, device, valid_loader, loss_fn, metrics)
        else:
            batch_vloss = 0.0
            _metrics_val = 0.0
            
        valid_loss_list.append(batch_vloss)
        metrics_val.append(_metrics_val)
        print('Epoch {}: Loss train {} valid {}. LR {}'.
              format(epoch, batch_loss, batch_vloss, scheduler.get_last_lr()))
        print(_metrics_val)
        learning_rate_list.append(float(optimizer.param_groups[0]['lr']))
        
        # Save checkpoint every 5 epoch
        if (epoch+1)%5 == 0:
            for param_group in optimizer.param_groups:
                print(param_group['lr'])
        
            print('save model!!!') 
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, join(path_to_model, 'checkpoint{}.pt'.format(epoch_number)))
            
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, join(path_to_model, 'checkpoint.pt'))

        np.savez(join(path_to_model,'loss_epoch.npz'), 
                 train_loss=np.array(train_loss_list), 
                 valid_loss=np.array(valid_loss_list), 
                 metrics=np.array(metrics_val),
                 learning_rate=np.array(learning_rate_list)
                 )
        
        epoch_number += 1

    return train_loss_list, valid_loss_list


def test(model, data_loader, device, path_to_saved_data, metrics, use_cuda=True):
    mse_loss_fn, rse_loss_fn, l1_rse_loss_fn = metrics
    # data_loader = DataLoader(dataset, sampler=sampler)
    model.eval()
    if use_cuda:
        model.to(device)
    else:
        model.to('cpu')
    preds, trues = [], []
    mse_losses, rse_losses = [], []
    p_mse_losses, s_mse_losses = [], []
    p_rse_losses, s_rse_losses = [], []
    p_l1rse_losses, s_l1rse_losses = [], []
    loop = tqdm(data_loader)
    for data in loop:
        if use_cuda:
            _contrl, _states, _static, outputs = extract_data_instance(
                data, device)
        else:
            _contrl, _states, _static, outputs = extract_data_instance(data, 'cpu')

        with torch.no_grad():
            pred, _ = model(_contrl, _states, _static)
            _mse_loss = mse_loss_fn(pred, outputs).to('cpu').item()
            _rse_loss = rse_loss_fn(pred, outputs).to('cpu').item()

        trues.append(outputs.to('cpu').detach().numpy())
        preds.append(pred.to('cpu').detach().numpy())
        _mse_loss = mse_loss_fn(pred, outputs).to('cpu').item()
        _rse_loss = rse_loss_fn(pred, outputs).to('cpu').item()
        _p_mse_losses = mse_loss_fn(pred[:, :, 0], outputs[:, :, 0]).to('cpu').item()
        _s_mse_losses = mse_loss_fn(pred[:, :, 1], outputs[:, :, 1]).to('cpu').item()
        _p_rse_losses = rse_loss_fn(pred[:, :, 0], outputs[:, :, 0]).to('cpu').item()
        _s_rse_losses = rse_loss_fn(pred[:, :, 1], outputs[:, :, 1]).to('cpu').item()
        _p_l1rse_losses = l1_rse_loss_fn(pred[:, :, 0], outputs[:, :, 0]).to('cpu').item()
        _s_l1rse_losses = l1_rse_loss_fn(pred[:, :, 1], outputs[:, :, 1]).to('cpu').item()
        del _contrl, _states, _static, outputs, pred, _  # , state0

        # print(_mse_loss, _rse_loss)
        mse_losses.append(_mse_loss)
        rse_losses.append(_rse_loss)
        p_mse_losses.append(_p_mse_losses)
        s_mse_losses.append(_s_mse_losses)
        p_rse_losses.append(_p_rse_losses)
        s_rse_losses.append(_s_rse_losses)
        p_l1rse_losses.append(_p_l1rse_losses)
        s_l1rse_losses.append(_s_l1rse_losses)
        loop.set_postfix(mse=_mse_loss, rse=_rse_loss,
                         p_mse=_p_mse_losses, s_mse=_s_mse_losses,
                         p_rse=_p_rse_losses, s_rse=_s_rse_losses)
    # Collect Testing Results
    mse_losses = np.array(mse_losses)
    rse_losses = np.array(rse_losses)
    p_mse_losses = np.array(p_mse_losses)
    s_mse_losses = np.array(s_mse_losses)
    p_rse_losses = np.array(p_rse_losses)
    s_rse_losses = np.array(s_rse_losses)
    preds = np.vstack(preds)
    trues = np.vstack(trues)
    p_pred, s_pred = preds[:, :, 0], preds[:, :, 1]
    p_true, s_true = trues[:, :, 0], trues[:, :, 1]
    print(mse_losses.shape, rse_losses.shape)
    print(p_pred.shape, s_pred.shape, p_true.shape, s_true.shape)
    print(preds.shape, trues.shape)
    np.savez(path_to_saved_data,
             mse_loss=mse_losses,
             rse_lose=rse_losses,
             p_mse_losses=p_mse_losses,
             s_mse_losses=s_mse_losses,
             p_rse_losses=p_rse_losses,
             s_rse_losses=s_rse_losses,
             preds=preds,
             trues=trues)
    # matdict = {'preds': preds, 'trues': trues}
    # sio.savemat(join(path_to_model, '{}_test_result.mat'.format(folder)), matdict)
    del preds, trues, data_loader

    return p_pred, s_pred, p_true, s_true


