Paper Information
Title: Cross-Spectral Cross-Attention Dehazing Network for Hyperspectral Remote Sensing Images
Status: Submitted 
Research Type: Hyperspectral Image Dehazing


Network Overview
This paper proposes a Cross-Spectral Cross-Attention Network, dubbed CSCANet, for hyperspectral RS image dehazing. This proposed algorithm introduces an innovative spectral-wise attention mechanism, which adaptively transfers haze-robust features from longer-wavelength bands to steer the restoration of heavily degraded short-wavelength spectral channels. In our proposed CSCANet, there are mainly four modules including band grouping module (BGM), band exchanging module (BEM), cross-spectral cross-attention module (CSCAM), and multi-scale residual module (MRM). First, a BGM is presented to divide all the spectral bands into three band groups according to wavelength ranges. Second, based on a new band scoring strategy, we design a BEM to exchange spectral bands between each pair of adjacent band groups. Third, we propose a novel CSCAM which calculates the attention map for exchanged adjacent band groups, to handle their fog-induced spectral distortion relationships. So that, the dehazing of the shorter-wavelength spectral band data is naturally guided by the longer-wavelength one. Finally, an MRM is developed, considering the heterogeneous land cover characteristics in RS scenes, to enhance the network robustness and boost the dehazing performance. In addition, we also propose two synthetic hazy HSI datasets based on the famous atmospheric scattering model. One is named as HSIDeD-AVI (HSI Dehazing Dataset - AVIRIS), derived from the widely used Airborne Visible/Infrared Imaging Spectrometer (AVIRIS) dataset. And the other one is HSIDeD-UAV (HSI Dehazing Dataset - Unmanned Aerial Vehicle), created from hyperspectral RS images captured not by airplanes but by UAV platforms.


Dependencies
This project is implemented in Python using the following major libraries:
torch
einops
os
matplotlib
scipy
glob
PIL
numpy
h5py
cv2
tifffile


Dataset
The dataset created by the authors mentioned in the paper can be accessed at the following link:DOI 10.5281/zenodo.15675446.


Note
The code is provided for academic and research purposes only.
This version is shared as part of the submission process and may be updated upon acceptance.