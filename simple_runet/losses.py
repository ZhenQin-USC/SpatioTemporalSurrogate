import torch
import torch.nn as nn
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


