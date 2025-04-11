import torch


def get_kernels_3d(filter_type: str) -> torch.Tensor:
    """
    Returns: Tensor of shape (N, 1, 3, 3, 3)
    """
    filter_type = filter_type.lower()
    if filter_type == 'sobel':
        return _sobel_kernels()
    elif filter_type == 'scharr':
        return _scharr_kernels()
    elif filter_type == 'central':
        return _central_kernels()
    elif filter_type == 'laplacian':
        return _laplacian_kernel()  # 7-point stencil
    elif filter_type in ['laplacian27', 'lap27']:
        return laplacian_kernel_27()
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")


def _sobel_kernels():
    gx = torch.tensor([
        [[-1, 0, 1], [-3, 0, 3], [-1, 0, 1]],
        [[-3, 0, 3], [-6, 0, 6], [-3, 0, 3]],
        [[-1, 0, 1], [-3, 0, 3], [-1, 0, 1]],
    ]) / 32.0
    gy = gx.permute(1, 0, 2).clone()
    gz = gx.permute(2, 1, 0).clone()
    return torch.stack([gx, gy, gz]).unsqueeze(1)


def _scharr_kernels():
    gx = torch.tensor([
        [[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]],
        [[-10, 0, 10], [-30, 0, 30], [-10, 0, 10]],
        [[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]],
    ]) / 256.0
    gy = gx.permute(1, 0, 2).clone()
    gz = gx.permute(2, 1, 0).clone()
    return torch.stack([gx, gy, gz]).unsqueeze(1)


def _central_kernels():
    dx = torch.zeros((3, 3, 3))
    dx[1, 1, 2] = 0.5
    dx[1, 1, 0] = -0.5
    dy = dx.permute(1, 0, 2).clone()
    dz = dx.permute(2, 1, 0).clone()
    return torch.stack([dx, dy, dz]).unsqueeze(1)


def _laplacian_kernel():
    lap = torch.zeros((3, 3, 3))
    lap[1, 1, 1] = -6
    lap[0, 1, 1] = lap[2, 1, 1] = 1
    lap[1, 0, 1] = lap[1, 2, 1] = 1
    lap[1, 1, 0] = lap[1, 1, 2] = 1
    return lap.unsqueeze(0).unsqueeze(0)  # (1,1,3,3,3)


def laplacian_kernel_27():
    """
    Returns: Tensor of shape (1, 1, 3, 3, 3)
    27-point Laplacian stencil from the image, scaled by 1/26.
    """
    core = torch.tensor([
        [[2, 3, 2],
         [3, 6, 3],
         [2, 3, 2]],

        [[3, 6, 3],
         [6, -88, 6],
         [3, 6, 3]],

        [[2, 3, 2],
         [3, 6, 3],
         [2, 3, 2]]
    ], dtype=torch.float32)

    return (core / 26.0).unsqueeze(0).unsqueeze(0)  # shape (1,1,3,3,3)
