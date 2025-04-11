import os
import torch
import torch.nn as nn
import numpy as np

from os.path import join
from tqdm import tqdm
from collections import defaultdict

from torch.optim.lr_scheduler import StepLR


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

