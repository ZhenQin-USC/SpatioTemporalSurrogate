import os
import torch
import random
import torch.nn as nn
import numpy as np
import torch.optim as optim
from os.path import join
from datetime import datetime
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from .losses import RelativeError, PerceptualLoss


class Trainer:
    def __init__(
        self,
        model,
        device,
        train_loader,
        valid_loader,
        path_to_model,
        learning_rate=1e-3,
        step_size=250,
        gamma=0.975,
        loss_fn=None,
        gradient_clip=False,
        gradient_clip_val=None,
        weight_decay=0,
        metrics=None,
        metrics_names=None,
        checkpoint_interval=None,
        **kwargs
        ):
        """
        Initializes the Trainer class.

        Parameters:
        - model: The neural network model to train.
        - device: The device to run the model on ('cpu' or 'cuda').
        - train_loader: DataLoader for training data.
        - valid_loader: DataLoader for validation data.
        - path_to_model: Directory path to save the model checkpoints.
        - learning_rate: Initial learning rate for the optimizer.
        - step_size: Period of learning rate decay.
        - gamma: Multiplicative factor of learning rate decay.
        - loss_fn: Loss function to use (default is nn.MSELoss).
        - gradient_clip: Whether to apply gradient clipping.
        - gradient_clip_val: Value to clip gradients at.
        - weight_decay: Weight decay (L2 penalty) for the optimizer.
        - metrics: List of metric functions for evaluation.
        - metrics_names: List of names corresponding to the metrics.
        """
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.path_to_model = path_to_model
        self.learning_rate = learning_rate
        self.step_size = step_size
        self.gamma = gamma
        self.loss_fn = loss_fn if loss_fn else nn.MSELoss()
        self.gradient_clip = gradient_clip
        self.gradient_clip_val = gradient_clip_val
        self.weight_decay = weight_decay
        self.checkpoint_interval = max(1, checkpoint_interval) if checkpoint_interval is not None else 1

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.scheduler = StepLR(self.optimizer, step_size=self.step_size, gamma=self.gamma, last_epoch=-1, verbose=False)

        self.metrics = metrics if metrics else [RelativeError(p=1), RelativeError(p=2), nn.MSELoss()]
        self.metrics_names = metrics_names if metrics_names else ['L1_RSE', 'L2_RSE', 'MSE']

        self.train_loss_list = []
        self.valid_loss_list = []
        self.learning_rate_list = []
        self.metrics_val = []
        self.epoch_number = 0
        
        os.makedirs(self.path_to_model, exist_ok=True) 
        self.log_file_path = join(self.path_to_model, 'training.log') 
        with open(self.log_file_path, 'w') as log_file:
            log_file.write("Training Log Initialized.\n")
            
    def log(self, message):
        """
        Logs a message to both the console and the log file with a timestamp.

        Parameters:
        - message: The message to log.
        """
        # Get the current timestamp
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        # Format the log message
        log_message = f"[{timestamp}] {message}"
        
        # Write to the log file
        with open(self.log_file_path, 'a') as log_file:
            log_file.write(log_message + '\n')
        
        # Print to the console
        print(log_message)

    def extract_data_instance(self, data):
        """
        Extracts and moves data to the specified device.

        Parameters:
        - data: A tuple containing inputs and outputs.

        Returns:
        - _contrl, _states, _static: Input tensors moved to device.
        - outputs: Output tensors moved to device.
        """
        inputs, outputs = data
        _contrl, _states, _static = inputs
        _contrl = _contrl.to(self.device)
        _states = _states.to(self.device)
        _static = _static.to(self.device)
        outputs = outputs.to(self.device)
        return _contrl, _states, _static, outputs

    def train_one_epoch(self, epoch, num_epochs, verbose=0):
        """
        Trains the model for one epoch.

        Parameters:
        - epoch: Current epoch number.
        - num_epochs: Total number of epochs.
        - verbose: Verbosity level (0 or 1).

        Returns:
        - Average training loss for the epoch.
        """
        self.model.train()
        batch_loss = 0.0

        if verbose == 1:
            loop = tqdm(self.train_loader)
        else:
            loop = self.train_loader

        for i, data in enumerate(loop):
            _contrl, _states, _static, outputs = self.extract_data_instance(data)
            self.optimizer.zero_grad()
            preds, _ = self.model(_contrl, _states, _static)
            loss = self.loss_fn(preds, outputs)
            loss.backward()

            if self.gradient_clip and self.gradient_clip_val:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)

            self.optimizer.step()
            self.scheduler.step()
            batch_loss += loss.item()

            if verbose == 1:
                loop.set_description(f"Epoch [{epoch+1}/{num_epochs}]")
                loop.set_postfix(loss=loss.item())

        avg_loss = batch_loss / (i + 1)
        return avg_loss

    def validate(self, verbose=0):
        """
        Validates the model on the validation dataset.

        Parameters:
        - verbose: Verbosity level (0 or 1).

        Returns:
        - Average validation loss.
        - List of average metric values.
        """
        self.model.eval()
        running_vloss = 0.0
        running_metrics = []

        if verbose == 1:
            loop = tqdm(self.valid_loader, desc="Validating")
        else:
            loop = self.valid_loader

        with torch.no_grad():
            for i, vdata in enumerate(loop):
                _contrl, _states, _static, voutputs = self.extract_data_instance(vdata)
                vpreds, _ = self.model(_contrl, _states, _static)
                vloss = self.loss_fn(vpreds, voutputs)
                running_vloss += vloss.item()

                _metrics_val = []
                for metric_fn in self.metrics:
                    metric = metric_fn(vpreds, voutputs)
                    _metrics_val.append(metric.item())
                running_metrics.append(_metrics_val)
                
                if verbose == 1:
                    loop.set_postfix(loss=vloss.item())

        avg_vloss = running_vloss / (i + 1)
        avg_metrics = np.mean(running_metrics, axis=0).tolist()
        return avg_vloss, avg_metrics

    def train(self, num_epochs, verbose=0, if_track_validate=True, resume_training=False, curr_epoch=None):
        """
        Trains the model for a specified number of epochs.

        Parameters:
        - num_epochs: Number of epochs to train.
        - verbose: Verbosity level (0 or 1).
        - if_track_validate: Whether to perform validation after each epoch.
        - resume_training: Boolean flag to indicate whether to resume training from checkpoints.
        """
        # Check if there is an existing checkpoint to resume training
        latest_checkpoint_path = join(self.path_to_model, 'checkpoint.pt')
        if resume_training and os.path.exists(latest_checkpoint_path):
            self.log(f"Found existing checkpoint at {latest_checkpoint_path}. Resuming training...")
            self.load_checkpoint(latest_checkpoint_path)
            self.log(f"Resumed from epoch {self.epoch_number}.")
            if curr_epoch is not None:
                self.epoch_number = curr_epoch 
            else:
                raise ValueError("Current epoch number should be provided for continuing training")
        else:
            if resume_training:
                self.log("No existing checkpoint found. Cannot resume training.")
            else:
                self.log("Starting training from scratch. Ignoring any existing checkpoints.")
                
        # Begin training
        start_epoch = self.epoch_number
        for epoch in range(start_epoch, num_epochs):
            train_loss = self.train_one_epoch(epoch, num_epochs, verbose=verbose)
            self.train_loss_list.append(train_loss)

            if if_track_validate:
                valid_loss, metrics_values = self.validate()
                self.valid_loss_list.append(valid_loss)
                self.metrics_val.append(metrics_values)
            else:
                valid_loss = None
                metrics_values = None

            current_lr = self.optimizer.param_groups[0]['lr']
            self.learning_rate_list.append(current_lr)
            self.log(f"Epoch {epoch+1}/{num_epochs}: Train Loss = {train_loss:.6f}, Valid Loss = {valid_loss:.6f}, LR = {current_lr:.6f}")

            if metrics_values:
                metrics_str = ', '.join(
                    [f"{name}: {value:.6f}" for name, value in zip(self.metrics_names, metrics_values)]
                )
                self.log(f"Validation Metrics: {metrics_str}")
            
            # Save checkpoint based on the interval
            self.save_checkpoint(epoch)
            self.epoch_number += 1

    def save_checkpoint(self, epoch):
        """
        Saves the model checkpoint.

        Parameters:
        - epoch: Current epoch number.
        """
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(), 
        }

        # Save the intermediate checkpoint every k epochs, k = checkpoint_interval
        if self.checkpoint_interval and (epoch + 1) % self.checkpoint_interval == 0:
            checkpoint_path = join(self.path_to_model, f'checkpoint_epoch{epoch+1}.pt')
            torch.save(checkpoint, checkpoint_path)

        # Save the latest checkpoint every epoch
        latest_checkpoint_path = join(self.path_to_model, 'checkpoint.pt')
        torch.save(checkpoint, latest_checkpoint_path)

        # Save training logs every epoch
        np.savez(
            join(self.path_to_model, 'training_logs.npz'),
                 train_loss=np.array(self.train_loss_list),
                 valid_loss=np.array(self.valid_loss_list),
                 metrics=np.array(self.metrics_val),
                 learning_rate=np.array(self.learning_rate_list)
            )
        
    def load_training_logs(self):
        """
        Loads training logs from the saved .npz file.
    
        Returns:
        - None
        """
        logs_path = join(self.path_to_model, 'training_logs.npz')
        if os.path.exists(logs_path):
            logs = np.load(logs_path, allow_pickle=True)
            self.train_loss_list = logs.get('train_loss', []).tolist()
            self.valid_loss_list = logs.get('valid_loss', []).tolist()
            self.metrics_val = logs.get('metrics', []).tolist()
            self.learning_rate_list = logs.get('learning_rate', []).tolist()
            self.log("Training logs loaded successfully.")
        else:
            self.log("No training logs found. Initializing empty logs.")
            self.train_loss_list = []
            self.valid_loss_list = []
            self.metrics_val = []
            self.learning_rate_list = []

    def load_checkpoint(self, checkpoint_path):
        """
        Loads the model, optimizer, and scheduler state from a checkpoint.

        Parameters:
        - checkpoint_path: Path to the checkpoint file.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict']) 
        self.epoch_number = checkpoint['epoch']
        self.log(f"Checkpoint loaded. Resumed from epoch {self.epoch_number}.")
    
        # Load training logs
        self.load_training_logs()

    def save_model(self, model_name):
        """
        Saves the model state dictionary.

        Parameters:
        - model_name: Name of the model file.
        """
        save_path = join(self.path_to_model, model_name + '.pt')
        torch.save(self.model.state_dict(), save_path)

