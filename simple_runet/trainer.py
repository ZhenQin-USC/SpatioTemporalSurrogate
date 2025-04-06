import torch
import torch.nn as nn
import numpy as np
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from os.path import join
from torch.utils.data import Dataset


class Dataset(Dataset):
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


class Trainer: # Base class for training and testing Simple-RUNET models
    def __init__(self, model, train_config, pixel_loss=None, regularizer=None, device=None):
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
            if device is None else device
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

    def _extract_data_instance(self, data):
        _s, _p, _m = data
        states = torch.cat((_s[:, [0]], _p[:, [0]]), dim=1).to(self.device) # initial state
        static = _m.to(self.device)
        outputs = torch.cat((_s[:, 1:][:, :, None], _p[:, 1:][:, :, None]), dim=2).to(self.device)
        return states, static, outputs

    def load_model(self, path_to_ckpt):
        checkpoint = torch.load(join(path_to_ckpt, 'checkpoint.pt'), map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

    def save_checkpoint(self, epoch, path_to_model, 
                        train_loss_list, train_pixel_loss, train_aux_loss,
                        valid_loss_list, valid_pixel_loss, valid_aux_loss,
                        learning_rate_list, save_epoch_ckpt=True):
        
        if save_epoch_ckpt:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }, join(path_to_model, f'checkpoint{epoch}.pt'))

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, join(path_to_model, 'checkpoint.pt'))
        
        np.savez(join(path_to_model, 'loss_epoch.npz'),
                 train_loss=np.array(train_loss_list),
                 train_pixel=np.array(train_pixel_loss),
                 train_aux=np.array(train_aux_loss),
                 valid_loss=np.array(valid_loss_list),
                 valid_pixel=np.array(valid_pixel_loss),
                 valid_aux=np.array(valid_aux_loss),
                 learning_rate=np.array(learning_rate_list))

    def _train_one_epoch(self, train_loader, epoch):
        self.model.train()
        loop = tqdm(train_loader) if self.verbose == 1 else train_loader

        batch_loss, pixel_loss, auxillary_loss = 0.0, 0.0, 0.0

        for i, data in enumerate(loop):
            self.optimizer.zero_grad()
            out = self.forward(data)
            loss = out['loss']
            loss.backward()

            if self.gradient_clip and self.gradient_clip_val:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)

            self.optimizer.step()
            self.scheduler.step()

            batch_loss += loss.item()
            pixel_loss += out['loss_pixel'].item()
            auxillary_loss += out['loss_auxillary'].item()

            if self.verbose == 1:
                loop.set_description(f"Epoch [{epoch+1}/{self.num_epochs}]")
                loop.set_postfix(
                    loss=batch_loss / (i + 1),
                    pixel_loss=pixel_loss / (i + 1),
                    aux_loss=auxillary_loss / (i + 1)
                )

        return batch_loss / (i + 1), pixel_loss / (i + 1), auxillary_loss / (i + 1)

    def validate(self, valid_loader):
        self.model.eval()
        batch_vloss, pixel_vloss, auxillary_vloss = 0.0, 0.0, 0.0
        loop = tqdm(valid_loader, desc="Validating") if self.verbose == 1 else valid_loader

        for i, data in enumerate(loop):
            with torch.no_grad():
                out = self.forward(data)
                batch_vloss += out['loss'].item()
                pixel_vloss += out['loss_pixel'].item()
                auxillary_vloss += out['loss_auxillary'].item()
            
            if self.verbose == 1:
                loop.set_postfix(
                    loss=batch_vloss / (i + 1),
                    pixel_loss=pixel_vloss / (i + 1),
                    aux_loss=auxillary_vloss / (i + 1)
                )

        return batch_vloss / (i + 1), pixel_vloss / (i + 1), auxillary_vloss / (i + 1)

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
            'loss': total_loss,
            'loss_pixel': loss_pixel.detach(),
            'loss_auxillary': loss_reg.detach(),
            'preds': preds.detach().cpu(),
            'outputs': outputs.detach().cpu(),
            'static': static.detach().cpu(),
            'states': states.detach().cpu()
        }

    def train(self, train_loader, valid_loader, path_to_model, ckpt_epoch=5):
        train_loss_list, valid_loss_list = [], []
        train_pixel_loss, train_aux_loss = [], []
        valid_pixel_loss, valid_aux_loss = [], []
        learning_rate_list = []

        torch.autograd.set_detect_anomaly(True)

        for epoch in range(self.num_epochs):
            self.optimizer.zero_grad()
            train_loss, pixel_loss, aux_loss = self._train_one_epoch(train_loader, epoch)

            train_loss_list.append(train_loss)
            train_pixel_loss.append(pixel_loss)
            train_aux_loss.append(aux_loss)
            learning_rate_list.append(self.optimizer.param_groups[0]['lr'])

            val_loss, val_pixel, val_aux = self.validate(valid_loader)
            valid_loss_list.append(val_loss)
            valid_pixel_loss.append(val_pixel)
            valid_aux_loss.append(val_aux)

            print(f"Epoch {epoch:03d}: Train - total {train_loss:.6f} | pixel {pixel_loss:.6f} | aux {aux_loss:.6f}")
            print(f"             Valid - total {val_loss:.6f} | pixel {val_pixel:.6f} | aux {val_aux:.6f}")
            print(f"             LR: {self.scheduler.get_last_lr()}")

            self.save_checkpoint(
                epoch=epoch,
                path_to_model=path_to_model,
                train_loss_list=train_loss_list,
                train_pixel_loss=train_pixel_loss,
                train_aux_loss=train_aux_loss,
                valid_loss_list=valid_loss_list,
                valid_pixel_loss=valid_pixel_loss,
                valid_aux_loss=valid_aux_loss,
                learning_rate_list=learning_rate_list,
                save_epoch_ckpt=((epoch + 1) % ckpt_epoch == 0)
            )

        return train_loss_list, valid_loss_list

    def test(self, test_loader):
        self.model.eval()
        preds, states, static = [], [], []
        batch_loss, pixel_loss, aux_loss = 0.0, 0.0, 0.0

        loop = tqdm(test_loader) if self.verbose == 1 else test_loader

        for i, data in enumerate(loop):

            with torch.no_grad():
                out = self.forward(data)

            batch_loss += out['loss'].item()
            pixel_loss += out['loss_pixel'].item()
            aux_loss += out['loss_auxillary'].item()

            preds.append(out['preds'])
            states.append(out['outputs'])
            static.append(out['static'])

            if self.verbose == 1:
                loop.set_description("Testing")
                loop.set_postfix(
                    loss=batch_loss / (i + 1),
                    pixel_loss=pixel_loss / (i + 1),
                    aux_loss=aux_loss / (i + 1)
                )

        return {
            'preds': torch.vstack(preds),
            'states': torch.vstack(states),
            'static': torch.vstack(static),
            'test_loss': batch_loss / (i + 1),
            'pixel_loss': pixel_loss / (i + 1),
            'aux_loss': aux_loss / (i + 1)
        }


class Trainer_RUNET(Trainer): # Trainer for RUNET models
    def __init__(self, model, train_config, wells=None, **kwargs):
        super().__init__(model, train_config, **kwargs)
        self.wells = wells or [(42, 42, 16), (42, 42, 17), (27, 27, 16), (27, 27, 17)]

    def _extract_data_instance(self, data):
        _s, _p, _m = data # (s, p) ~ (B, T + 1, X, Y, Z), (m) ~ (B, 1, X, Y, Z)
        _u = torch.zeros_like(_s[:, 1:]) # Placeholder for the control input (B, T, X, Y, Z)
        for ix, iy, iz in self.wells:
            _u[..., ix, iy, iz] = 1
        contrl = _u[:, :, None].to(self.device) # Control input (B, T, 1, X, Y, Z)
        states = torch.cat((_s[:, [0]], _p[:, [0]]), dim=1).to(self.device) # initial state
        static = _m.to(self.device)
        outputs = torch.cat((_s[:, 1:][:, :, None], _p[:, 1:][:, :, None]), dim=2).to(self.device)
        return contrl, states, static, outputs

    def forward(self, data):
        contrl, states, static, outputs = self._extract_data_instance(data)

        preds = self.model(contrl, states, static)

        loss_pixel = self.loss_fn(preds, outputs)

        if self.regularizer is not None:
            loss_reg = self.regularizer_weight * self.regularizer(preds, outputs)
        else:
            loss_reg = torch.tensor(0.0, device=self.device)

        total_loss = loss_pixel + loss_reg
        return {
            'loss': total_loss,
            'loss_pixel': loss_pixel.detach(),
            'loss_auxillary': loss_reg.detach(),
            'preds': preds.detach().cpu(),
            'outputs': outputs.detach().cpu(),
            'static': static.detach().cpu(),
            'states': states.detach().cpu()
        }



class Trainer_RUNET_2D(Trainer): # Trainer for 2D Dataset
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

        if self.regularizer is not None:
            loss_reg = self.regularizer_weight * self.regularizer(preds, outputs)
        else:
            loss_reg = torch.tensor(0.0, device=self.device)

        total_loss = loss_pixel + loss_reg
        return {
            'loss': total_loss,
            'loss_pixel': loss_pixel.detach(),
            'loss_auxillary': loss_reg.detach(),
            'preds': preds.detach().cpu(),
            'outputs': outputs.detach().cpu(),
            'static': static.detach().cpu(),
            'states': states.detach().cpu()
        }

