import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2

from Diffraction_H import Diffraction_propagation, get_amplitude, get_phase

# input zernike and crop
zernike_pha = cv2.imread('test_img/exp/4.23_zernike1.png', cv2.IMREAD_GRAYSCALE) / 255
zernike_pha = torch.tensor(zernike_pha, dtype=torch.float64) * 2 * torch.pi
zernike_pha = zernike_pha[40:1040, 460:1460]
plt.imshow(zernike_pha.numpy(), cmap='gray')
plt.show()

# input src
src = cv2.imread('test_img/exp/src.bmp', cv2.IMREAD_GRAYSCALE) / 255
src = cv2.resize(src, (600, 600), interpolation=cv2.INTER_AREA)
padding_size = (200, 200)
src = np.pad(src, [padding_size, padding_size], mode='constant', constant_values=0)
plt.imshow(src, cmap='gray')
plt.show()

# input ref and abe
ref = cv2.imread('test_img/exp/ref.bmp', cv2.IMREAD_GRAYSCALE) / 255
ref = cv2.resize(ref, (600, 600), interpolation=cv2.INTER_AREA)
padding_size = (200, 200)
ref = np.pad(ref, [padding_size, padding_size], mode='constant', constant_values=0)
plt.imshow(ref, cmap='gray')
plt.show()
abe = cv2.imread('test_img/exp/abe.bmp', cv2.IMREAD_GRAYSCALE) / 255
abe = cv2.resize(abe, (600, 600), interpolation=cv2.INTER_AREA)
padding_size = (200, 200)
abe = np.pad(abe, [padding_size, padding_size], mode='constant', constant_values=0)
plt.imshow(abe, cmap='gray')
plt.show()

# Start propagation
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
lambda_ = 532e-9
pi = torch.tensor(torch.pi, dtype=torch.float64)
k = (2 * pi / lambda_)
dx = 8e-6
d = 0.1
src = torch.tensor(src, dtype=torch.float64)
obj_field = torch.sqrt(src)*torch.exp(1j*zernike_pha)
free_d0 = Diffraction_propagation(obj_field, d, dx, lambda_).to(device)
free_d0_amp = get_amplitude(free_d0)
free_d0_ph = get_phase(free_d0)
plt.imshow(free_d0_amp.cpu().data.numpy(), cmap='gray')
plt.show()
