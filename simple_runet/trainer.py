import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
import numpy as np
from tqdm import tqdm
from os.path import join


class Trainer:
    def __init__(self, model, train_config, pixel_loss=None, regularizer=None, device=None):
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
            if device is None else device
        self.model = model.to(self.device)
        self.train_config = train_config

        self.verbose = train_config.get('verbose', 1)

        self.gradient_clip = train_config.get('gradient_clip', False)
        self.gradient_clip_val = train_config.get('gradient_clip_val', None)

        # Loss function
        self.loss_fn = nn.MSELoss() if pixel_loss is None else pixel_loss
        self.regularizer = regularizer
        self.regularizer_weight = train_config.get('regularizer_weight', None)

        # Optimizer and LR scheduler
        self._optimizer = torch.optim.Adam(self.model.parameters(),
                                           lr=train_config.get('learning_rate'),
                                           weight_decay=train_config.get('weight_decay'))
        self._scheduler = StepLR(self._optimizer,
                                 step_size=train_config['step_size'],
                                 gamma=train_config['gamma'])

    def _extract_data_instance(self, data):
        _s, _p, _m = data
        _states = torch.cat((_s[:, [0]], _p[:, [0]]), dim=1).to(self.device) # initial state
        _static = _m.to(self.device)
        outputs = torch.cat((_s[:, 1:][:, :, None], _p[:, 1:][:, :, None]), dim=2).to(self.device)
        return _states, _static, outputs

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
                'optimizer_state_dict': self._optimizer.state_dict(),
            }, join(path_to_model, f'checkpoint{epoch}.pt'))

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self._optimizer.state_dict(),
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
            _states, _static, outputs = self._extract_data_instance(data)

            self._optimizer.zero_grad()
            out = self.forward(_states, _static, outputs)
            loss = out['loss']
            loss.backward()

            if self.gradient_clip and self.gradient_clip_val is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)

            self._optimizer.step()
            self._scheduler.step()

            batch_loss += loss.item()
            pixel_loss += out['loss_pixel'].item()
            auxillary_loss += out['loss_auxillary'].item()

            if self.verbose == 1:
                loop.set_description(f"Epoch [{epoch}/{self.train_config['num_epoch']}]")
                loop.set_postfix(
                    loss=batch_loss / (i + 1),
                    pixel_loss=pixel_loss / (i + 1),
                    aux_loss=auxillary_loss / (i + 1)
                )

        return batch_loss / (i + 1), pixel_loss / (i + 1), auxillary_loss / (i + 1)

    def _validate(self, valid_loader):
        self.model.eval()
        batch_vloss, pixel_vloss, auxillary_vloss = 0.0, 0.0, 0.0

        for i, data in enumerate(valid_loader):
            _states, _static, outputs = self._extract_data_instance(data)

            with torch.no_grad():
                out = self.forward(_states, _static, outputs)
                batch_vloss += out['loss'].item()
                pixel_vloss += out['loss_pixel'].item()
                auxillary_vloss += out['loss_auxillary'].item()

        return batch_vloss / (i + 1), pixel_vloss / (i + 1), auxillary_vloss / (i + 1)

    def forward(self, _states, _static, outputs):
        
        preds = self.model(_states, _static)

        loss_pixel = self.loss_fn(preds, outputs)

        if self.regularizer is not None:
            loss_reg = self.regularizer_weight * self.regularizer(preds, outputs)
        else:
            loss_reg = torch.tensor(0.0, device=self.device)

        total_loss = loss_pixel + loss_reg
        return {
            'preds': preds,
            'loss': total_loss,
            'loss_pixel': loss_pixel.detach(),
            'loss_auxillary': loss_reg.detach(),
        }

    def train(self, train_loader, valid_loader, path_to_model, ckpt_epoch=5):
        train_loss_list, valid_loss_list = [], []
        train_pixel_loss, train_aux_loss = [], []
        valid_pixel_loss, valid_aux_loss = [], []
        learning_rate_list = []

        torch.autograd.set_detect_anomaly(True)

        for epoch in range(self.train_config['num_epoch']):
            self._optimizer.zero_grad()
            train_loss, pixel_loss, aux_loss = self._train_one_epoch(train_loader, epoch)

            train_loss_list.append(train_loss)
            train_pixel_loss.append(pixel_loss)
            train_aux_loss.append(aux_loss)
            learning_rate_list.append(self._optimizer.param_groups[0]['lr'])

            val_loss, val_pixel, val_aux = self._validate(valid_loader)
            valid_loss_list.append(val_loss)
            valid_pixel_loss.append(val_pixel)
            valid_aux_loss.append(val_aux)

            print(f"Epoch {epoch:03d}: Train - total {train_loss:.6f} | pixel {pixel_loss:.6f} | aux {aux_loss:.6f}")
            print(f"             Valid - total {val_loss:.6f} | pixel {val_pixel:.6f} | aux {val_aux:.6f}")
            print(f"             LR: {self._scheduler.get_last_lr()}")

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
            _states, _static, outputs = self._extract_data_instance(data)

            with torch.no_grad():
                out = self.forward(_states, _static, outputs)

            batch_loss += out['loss'].item()
            pixel_loss += out['loss_pixel'].item()
            aux_loss += out['loss_auxillary'].item()

            preds.append(out['preds'].cpu())
            states.append(outputs.cpu())
            static.append(_static.cpu())

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
