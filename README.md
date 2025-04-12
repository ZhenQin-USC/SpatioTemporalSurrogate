# SpatioTemporalSurrogate

A surrogate modeling framework for learning spatiotemporal physical fields, powered by deep neural networks (e.g., Recurrent U-Net), and designed for CO₂ plume prediction.

## Features
- ✅ Supports 4D structured inputs using hierarchical deep learning model: `(B, T, C, X, Y, Z)`
- ✅ Modular training framework (Trainer class)
- ✅ Loss support (SSIM, Gradient, Perceptual)

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
## How to run:

Run Case1-Train.ipynb for a simple 3D toy example.

Run Case2-Train.ipynb for a simple 2D toy example.

---

## Key packages:

- `torch`  
- `torchvision`  
- `kornia`  
- `matplotlib`  
- `numpy`  
