import os
from unit import phasemap_8bit
import cv2
from torch import nn
import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
from Diffraction_H import get_0_2pi, Diffraction_propagation, get_amplitude, get_phase
from Zernike import generate_zer_poly
import imageio
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
lambda_ = 532e-9  # mm
pi = torch.tensor(np.pi, dtype=torch.float64)
k = (2 * pi / lambda_)
dx = 8e-6  # m
d0 = 0.03  # m
size = 1000
mask_size = (size, size)
iter_num = 500
x = torch.linspace(-size / 2, size / 2, size, dtype=torch.float64) * dx
y = torch.linspace(-size / 2, size / 2, size, dtype=torch.float64) * dx
X, Y = torch.meshgrid(x, y, indexing='xy')
rho = torch.sqrt(X ** 2 + (Y ** 2))
Phi = torch.atan2(Y, X)
img_path = 'gray_grid10.bmp'
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) / 255.0
img = torch.tensor(img, dtype=torch.float64, device=device)
zernike_pha = cv2.imread('../test_img/exp/4.23_zernike1.png', cv2.IMREAD_GRAYSCALE) / 255
zernike_pha = torch.tensor(zernike_pha, dtype=torch.float64, device=device) * 2 * torch.pi
# slm_plane = img*torch.exp(1j*zernike_pha)
slm_plane = Diffraction_propagation(img, d0, dx, lambda_, device=device)
slm_plane_fft = torch.fft.fftshift(torch.fft.fft2(img)) * torch.exp(1j*zernike_pha)
ori_abe = get_amplitude(slm_plane_fft)
ori_pha = get_phase(slm_plane_fft)
plt.imshow(ori_abe.cpu(), cmap='gray')
plt.show()
# plt.imsave('ori_abe.png', ori_abe.cpu().numpy())
plt.imshow(ori_pha.cpu(), cmap='gray')
plt.show()
# imageio.imsave('ori_pha.png', phasemap_8bit(ori_pha, inverted=True))

def random_phase_recovery(sensor_abe, random_phase, d0, dx, lambda_, iter_num, method, device):
    if method=='ASM':
        init_u = Diffraction_propagation(sensor_abe, -d0, dx, lambda_, device=device)  # Back prop
        init_u = torch.mean(init_u, dim=1)  # 1,1080,1920
        pbar = tqdm(range(iter_num))
        for i in pbar:
            sensor_p = Diffraction_propagation(init_u.unsqueeze(0) * torch.exp(1j * random_phase), d0, dx, lambda_,
                                               device=device)
            sensor_angle = get_phase(sensor_p)
            new_sensor = sensor_abe * torch.exp(1j * sensor_angle)
            # new_sensor = ((sensor_abe - get_amplitude(sensor_p)) * torch.rand(1, 8, 1, 1, device=device) + sensor_abe)
            # * torch.exp(1j * sensor_angle)
            new_slm = Diffraction_propagation(new_sensor, -d0, dx, lambda_, device=device)  # Back prop
            init_u = torch.mean(new_slm * torch.exp(-1j * random_phase), dim=1)
        return init_u
    if method=='FFT':
        init_u = torch.fft.ifftshift(torch.fft.ifft2(sensor_abe))
        init_u = torch.mean(init_u, dim=1)  # 1,1080,1920
        pbar = tqdm(range(iter_num))
        for i in pbar:
            sensor_p = torch.fft.fftshift(torch.fft.fft2(init_u.unsqueeze(0) * torch.exp(1j * random_phase)))

            sensor_angle = get_phase(sensor_p)
            new_sensor = sensor_abe * torch.exp(1j * sensor_angle)
            # new_sensor = ((sensor_abe - get_amplitude(sensor_p)) * torch.rand(1, 8, 1, 1, device=device) + sensor_abe)
            # * torch.exp(1j * sensor_angle)
            new_slm = torch.fft.ifft2(torch.fft.ifftshift(new_sensor))  # Back prop
            init_u = torch.mean(new_slm * torch.exp(-1j * random_phase), dim=1)
        return init_u




vortex = torch.zeros(8, 1000, 1000)
for l in range(8):
    vortex[l] = get_0_2pi((l + 1) * 4 * Phi)
pad_left = pad_right = (1920 - 1000) // 2
pad_top = pad_bottom = (1080 - 1000) // 2
vortex = F.pad(vortex, (pad_left, pad_right, pad_top, pad_bottom), "constant", 0).to(device)
vortex = vortex.unsqueeze(0)

phase = torch.rand(12, 108, 192, dtype=torch.float64, device=device)
phase = F.interpolate(phase.unsqueeze(0), size=(1080, 1920), mode='bicubic', align_corners=False)
slm_plane = slm_plane_fft * torch.exp(1j * phase)
# sensor_plane = Diffraction_propagation(slm_plane, d0, dx, lambda_, device=device)
# sensor_abe = get_amplitude(sensor_plane)  # 1,8,1080,1920
sensor_plane = torch.fft.fftshift(torch.fft.fft2(slm_plane))
sensor_abe = get_amplitude(sensor_plane)
#  Get sensor intensity and random phase, then recovery intensity and phase of slm plane field
recovery_field = random_phase_recovery(sensor_abe, phase, d0, dx, lambda_, iter_num, 'FFT', device)
est_abe = get_amplitude(recovery_field[0])
plt.imshow(est_abe.cpu(), cmap='gray')
plt.show()
# plt.imsave('est_abe.png', est_abe.cpu().numpy())
est_pha = get_phase(recovery_field[0])
plt.imshow(est_pha.cpu(), cmap='gray')
plt.show()

# imageio.imsave('est_pha.png', phasemap_8bit(ori_pha, inverted=True))
