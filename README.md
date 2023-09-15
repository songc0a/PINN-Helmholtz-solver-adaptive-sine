# PINN-Helmholtz-solver-adaptive-sine
**This repository reproduces the results of the submitted paper "A versatile framework to solve the Helmholtz equation using physics-informed neural networks." to Geophysical Journal International.**

# Overview

We applied the physics-informed neural networks (PINNs) to solve the Helmholtz equation for isotropic and anisotropic media. The proposed method has resilience and versatility in predicting frequency-domain wavefields for different media and model shapes.


# Installation of Tensorflow1

CPU usage: pip install --pre "tensorflow==1.15.*"

GPU usage: pip install --pre "tensorflow-gpu==1.15.*"

# Code explanation

Helm_pinn_tanh.py: Solving the Helmholtz equation using PINN with tanh activation function for isotropic media

Helm_pinn_sine_adaptive.py: Solving the Helmholtz equation using PINN with adpative sine activation function for isotropic media

Helm_pinn_sine_fixed.py: Solving the Helmholtz equation using PINN with fixed sine activation function for isotropic media

Helm_pinn_sine_vti_adaptive.py: Solving the Helmholtz equation using PINN with fixed sine activation function for VTI media

Helm_pinn_ps_sine_tti_topo.py: Solving the Helmholtz equation using PINN with fixed sine activation function for TTI media

Part of Matlab codes are included in my other repository

# Citation information

If you find our codes and publications helpful, please kindly cite the following publication.

@article{song2023weighted,
  title={Weighted Envelope Correlation-based Waveform Inversion Using Automatic Differentiation},
  author={Song, Chao and Wang, Yanghua and Richardson, Alan and Liu, Cai},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  volume={61},
  year={2023},
  publisher={IEEE}
}

# contact information
If there are any problems, please contact me through my emails: chao.song@kaust.edu.sa;csong1@ic.ac.uk
