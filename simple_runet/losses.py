import torch
import torch.nn as nn
import kornia 
from typing import (Union, Optional, Dict, List, Tuple)
from .lpips import LPIPS


class RelativeError(object):
    def __init__(self, p=2, d=None):
        """
        Initializes the RelativeError instance.
        
        Parameters:
        p (int): The norm type to use (e.g., L2 norm where p=2). Must be > 0.
        d (int or tuple of ints, optional): The dimension(s) over which to compute the norm. Default is None (compute over the entire tensor).
        """
        assert p > 0, "Lp-norm type `p` must be greater than 0"  # Ensure p is positive
        self.p = p  # Lp-norm type
        self.d = d  # Dimension along which to compute the norm

    def __call__(self, y_pred, y, epsilon=1e-8):
        """
        Computes the relative error between predicted and true values.
        
        Parameters:
        y_pred (torch.Tensor): Predicted tensor.
        y (torch.Tensor): True tensor.
        epsilon (float): Small value added to prevent division by zero. Default is 1e-8.
        
        Returns:
        torch.Tensor: Mean relative error across the specified dimension.
        """

        diff_norms = torch.norm(y_pred - y, p=self.p, dim=self.d) # Compute the Lp-norm of the difference
        y_norms = torch.norm(y, p=self.p, dim=self.d) + epsilon # Compute the Lp-norm of the true values, with epsilon to avoid division by zero
        return torch.mean(diff_norms / y_norms) # Compute the relative error and return the mean value


class RSELoss(object):
    def __init__(self, p=2, d=None, split=False, weights=None):
        """
        Initializes the RSELoss instance.
        
        Parameters:
        p (int): The norm type to use (e.g., L2 norm where p=2). Must be > 0.
        d (int or tuple of ints, optional): The dimension(s) over which to compute the norm. Default is None (compute over the entire tensor).
        split (bool): Whether to split the input tensors into primary and secondary components. Default is False.
        weights (list of floats, optional): Weights for combining losses if split is True. Default is [1, 1].
        """
        super().__init__()
        self.rel_error = RelativeError(p=p, d=d)  # Use the RelativeError class to compute relative errors
        self.split = split  # Whether to split the input tensor
        if weights is None:
            weights = [1, 1]
        self.p_weight, self.s_weight = weights[0], weights[1]

    def __call__(self, y_pred, y):
        """
        Calculates the RSE loss value for the given predicted and true tensors.
        
        Parameters:
        y_pred (torch.Tensor): Predicted tensor.
        y (torch.Tensor): True tensor.
        
        Returns:
        torch.Tensor: Computed RSE loss value.
        """
        if self.split: # Split the input tensors into p_pred/s_pred and p_true/s_true along the 3rd dimension (dim=2)
            p_pred, s_pred = torch.split(y_pred, (1, 1), dim=2)
            p_true, s_true = torch.split(y, (1, 1), dim=2)
            
            # Compute the weighted relative error loss for both components
            loss = self.p_weight * self.rel_error(p_pred, p_true)
            loss += self.s_weight * self.rel_error(s_pred, s_true)
        else:
            # If not split, compute the relative error between the entire predicted and true tensors
            loss = self.rel_error(y_pred, y)

        return loss


class PerceptualLoss(nn.Module):

    def __init__(
        self,
        spatial_dims: int,
        network_type: str = "alex",
        is_fake_3d: bool = True,
        fake_3d_ratio: float = 0.5,
        cache_dir: Optional[str] = None,
        pretrained: bool = True,
    ):
        super().__init__()

        if spatial_dims not in [2, 3]:
            raise NotImplementedError("Perceptual loss is implemented only in 2D and 3D.")

        if (spatial_dims == 2 or is_fake_3d) and "medicalnet_" in network_type:
            raise ValueError(
                "MedicalNet networks are only compatible with ``spatial_dims=3``."
                "Argument is_fake_3d must be set to False."
            )

        if cache_dir:
            torch.hub.set_dir(cache_dir)

        self.spatial_dims = spatial_dims
        self.perceptual_function = LPIPS(pretrained=pretrained, net=network_type, verbose=False)
        self.is_fake_3d = is_fake_3d
        self.fake_3d_ratio = fake_3d_ratio

    def _calculate_axis_loss(self, input: torch.Tensor, target: torch.Tensor, spatial_axis: int) -> torch.Tensor:
        """
        Calculate perceptual loss in one of the axis used in the 2.5D approach. After the slices of one spatial axis
        is transformed into different instances in the batch, we compute the loss using the 2D approach.

        Args:
            input: input 5D tensor. BNHWD
            target: target 5D tensor. BNHWD
            spatial_axis: spatial axis to obtain the 2D slices.
        """

        def batchify_axis(x: torch.Tensor, fake_3d_perm: tuple) -> torch.Tensor:
            """
            Transform slices from one spatial axis into different instances in the batch.
            """
            slices = x.float().permute((0,) + fake_3d_perm).contiguous()
            slices = slices.view(-1, x.shape[fake_3d_perm[1]], x.shape[fake_3d_perm[2]], x.shape[fake_3d_perm[3]])

            return slices

        preserved_axes = [2, 3, 4]
        preserved_axes.remove(spatial_axis)

        channel_axis = 1
        input_slices = batchify_axis(x=input, fake_3d_perm=(spatial_axis, channel_axis) + tuple(preserved_axes))
        indices = torch.randperm(input_slices.shape[0])[: int(input_slices.shape[0] * self.fake_3d_ratio)].to(
            input_slices.device
        )
        input_slices = torch.index_select(input_slices, dim=0, index=indices)
        target_slices = batchify_axis(x=target, fake_3d_perm=(spatial_axis, channel_axis) + tuple(preserved_axes))
        target_slices = torch.index_select(target_slices, dim=0, index=indices)

        axis_loss = torch.mean(self.perceptual_function(input_slices, target_slices))

        return axis_loss

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: the shape should be BNHW[D].
            target: the shape should be BNHW[D].
        """
        if target.shape != input.shape:
            raise ValueError(f"ground truth has differing shape ({target.shape}) from input ({input.shape})")

        if self.spatial_dims == 3 and self.is_fake_3d:
            # Compute 2.5D approach
            loss_sagittal = self._calculate_axis_loss(input, target, spatial_axis=2)
            loss_coronal = self._calculate_axis_loss(input, target, spatial_axis=3)
            loss_axial = self._calculate_axis_loss(input, target, spatial_axis=4)
            loss = loss_sagittal + loss_axial + loss_coronal
        else:
            # 2D and real 3D cases
            loss = self.perceptual_function(input, target)

        return torch.mean(loss)


class SSIMLoss(nn.Module):

    def __init__(self, window_size, max_val=1.0, eps=1e-12, reduction='mean', padding='same', mode='both'):
        """
        Parameters:
        - window_size: Size of the 3D SSIM window
        - max_val: Normalization value of data
        - reduction: 'mean', 'sum', or 'none'
        - mode: 'both', 'pressure', or 'saturation'
        """
        super().__init__()
        self.window_size = window_size
        self.max_val = max_val
        self.eps = eps
        self.reduction = reduction
        self.padding = padding
        assert mode in ['both', 'pressure', 'saturation'], "mode must be 'both', 'pressure', or 'saturation'"
        self.mode = mode

    def __call__(self, preds, trues):
        """
        Computes the SSIM loss between predictions and targets.

        Parameters:
        - preds: Predicted outputs. (B, T, 2*F, X, Y, Z) = [p_trues || s_trues]
        - trues: Ground truth outputs. (B, T, 2*F, X, Y, Z) = [p_preds || s_preds]

        Returns:
        - SSIM loss value.
        """
        B, T, X, Y, Z = trues.size(0), trues.size(1), trues.size(3), trues.size(4), trues.size(5)

        # Chunk into [primary || secondary]
        p_trues, s_trues = torch.chunk(trues, chunks=2, dim=2)  # (B, T, F, X, Y, Z)
        p_preds, s_preds = torch.chunk(preds, chunks=2, dim=2)  # (B, T, F, X, Y, Z)

        F = p_trues.size(2)  # Number of frames

        # Merge T and F into channel dimension
        p_trues = p_trues.view(B, T * F, X, Y, Z)
        s_trues = s_trues.view(B, T * F, X, Y, Z)
        p_preds = p_preds.view(B, T * F, X, Y, Z)
        s_preds = s_preds.view(B, T * F, X, Y, Z)

        # Selectively compute SSIM
        losses = []
        if self.mode in ['both', 'pressure']:
            p_ssim = kornia.losses.ssim3d_loss(
                p_preds, p_trues,
                window_size=self.window_size,
                max_val=self.max_val,
                reduction=self.reduction
            )
            losses.append(p_ssim)

        if self.mode in ['both', 'saturation']:
            s_ssim = kornia.losses.ssim3d_loss(
                s_preds, s_trues,
                window_size=self.window_size,
                max_val=self.max_val,
                reduction=self.reduction
            )
            losses.append(s_ssim)

        return sum(losses) / len(losses)


class PerceptualLoss3D(nn.Module):
    """
    Wrapper for perceptual loss that allows for the use of different perceptual loss functions.
    """
    def __init__(self, device, num_img_for_perceptual: int = 16):
        super().__init__()
        self._perceptual_loss = PerceptualLoss(spatial_dims=2, network_type="vgg").to(device)
        self.num_img_for_perceptual = num_img_for_perceptual

    def __call__(self, preds, trues):
        """
        Computes the perceptual loss between predictions and targets.

        Parameters:
        - preds: Predicted outputs. (B, T, 2*F, X, Y, Z) = [p_trues || s_trues]
        - trues: Ground truth outputs. (B, T, 2*F, X, Y, Z) = [p_preds || s_preds]

        Returns:
        - Perceptual loss value.
        """
        B, T, X, Y, Z = trues.size(0), trues.size(1), trues.size(3), trues.size(4), trues.size(5)

        p_trues, s_trues = torch.chunk(trues, chunks=2, dim=2)  # (B, T, F, X, Y, Z)
        p_preds, s_preds = torch.chunk(preds, chunks=2, dim=2)  # (B, T, F, X, Y, Z)
        
        p_trues_reshaped = p_trues.permute(0, -1, 1, 2, 3, 4).contiguous().view(-1, 1, X, Y) # (B*Z*T*F, X, Y)
        p_preds_reshaped = p_preds.permute(0, -1, 1, 2, 3, 4).contiguous().view(-1, 1, X, Y)
        s_trues_reshaped = s_trues.permute(0, -1, 1, 2, 3, 4).contiguous().view(-1, 1, X, Y)
        s_preds_reshaped = s_preds.permute(0, -1, 1, 2, 3, 4).contiguous().view(-1, 1, X, Y)

        total_images = p_trues_reshaped.size(0)
        indices = torch.randperm(total_images)[:self.num_img_for_perceptual]

        p_ploss = self._perceptual_loss(p_trues_reshaped[indices], p_preds_reshaped[indices])

        s_ploss = self._perceptual_loss(s_trues_reshaped[indices], s_preds_reshaped[indices])
        
        return p_ploss, s_ploss
