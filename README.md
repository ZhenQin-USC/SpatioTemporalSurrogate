# SpatioTemporalSurrogate

A surrogate modeling framework for learning spatiotemporal physical fields, powered by deep neural networks (e.g., Recurrent U-Net), and designed for CO₂ plume prediction.

## Features
- ✅ Supports 3D structured inputs: `(B, T, C, X, Y, Z)`
- ✅ Modular training framework (Trainer class)
- ✅ Multi-field loss support (SSIM, Gradient, Perceptual)
- ✅ Easily configurable via Python
- ✅ Compatible with PyTorch and MONAI

---

## Project Structure:
```
simple_runet/ 
├── __init__.py
├── losses.py   # MultiFieldLoss family 
├── trainer.py  # Trainer class 
├── unet.py     # U-Net & RUNet definitions 
├── registry.py # Loss registration 
├── utils.py    # Misc tools 
├── lpips.py 
├── pretrained_networks.py
├── get_kernels_3d.py 
├── requirements.txt 
└── README.md
```

---

## Key packages:

- `torch`  
- `torchvision`  
- `kornia`  
- `matplotlib`  
- `numpy`  
