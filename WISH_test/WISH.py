import os
from unit import phasemap_8bit, pad_tensor, creat_obj
import cv2
from torch import nn
import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
from Diffraction_H import get_0_2pi, Diffraction_propagation, get_amplitude, get_phase, lens_phase, get_hologram
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
n_max = 15
zer_radius = 400
iter_num = 500
x = torch.linspace(-size / 2, size / 2, size, dtype=torch.float64) * dx
y = torch.linspace(-size / 2, size / 2, size, dtype=torch.float64) * dx
X, Y = torch.meshgrid(x, y, indexing='xy')
rho = torch.sqrt(X ** 2 + (Y ** 2))
Phi = torch.atan2(Y, X)
img_path = 'usaf.png'
img = creat_obj(img_path, size, radius=500, binaty_inv=2, if_obj=True,
                device=device)  # 0: Inverse; 1: NoInverse; 2: None

plt.imshow(img.data.cpu().numpy(), cmap='gray')
plt.show()
zer_path = 'parameter/zernike_stack_{}_{}.pth'.format(n_max, zer_radius)
if os.path.exists(zer_path):
    zernike_stack = torch.load(zer_path).to(device)  # zur_num,1,1000,1000
    zer_num = zernike_stack.shape[0]
else:
    print('Generate Zernike polynomial')
    zernike_stack, zer_num = generate_zer_poly(size=1000, dx=dx, n_max=n_max, radius=zer_radius, device=device)
zernike_pha = get_0_2pi((torch.rand(zer_num, 1, 1, 1, dtype=torch.float64, device=device) * zernike_stack).sum(dim=0))
imageio.imsave('abe_pha.png', phasemap_8bit(zernike_pha, inverted=False))
# slm_plane = img*torch.exp(1j*zernike_pha)
slm_plane = Diffraction_propagation(img, d0, dx, lambda_, device=device)
slm_plane_fft = torch.fft.fftshift(torch.fft.fft2(img)) * torch.exp(1j * zernike_pha)
sensor_plane_fft = torch.fft.fft2(torch.fft.fftshift(slm_plane_fft))
ori_abe = get_amplitude(sensor_plane_fft)
ori_pha = get_phase(sensor_plane_fft)
plt.imshow(ori_abe[0].cpu(), cmap='gray')
plt.show()
plt.imsave('usaf_amp_abe.png', ori_abe[0].cpu())
plt.imshow(ori_pha[0].cpu(), cmap='gray')
plt.show()
# imageio.imsave('ori_pha.png', phasemap_8bit(ori_pha, inverted=True))

free_d0 = Diffraction_propagation(img, 0.1, dx, lambda_, 'Angular Spectrum', device)
free_d0_amp = get_amplitude(free_d0)
free_d0_ph = get_phase(free_d0)
len1_phs = lens_phase(X, Y, k, 0.1).to(device)  # lens1 phase
# len1_phs = pad_tensor(len1_phs, 1080, 1920, 0)
new_ph = get_0_2pi(free_d0_ph - len1_phs)
free_d1_field = get_hologram(free_d0_amp, new_ph)
free_d1 = Diffraction_propagation(free_d1_field, 0.1, dx, lambda_, 'Angular Spectrum', device)
free_d1_amp = get_amplitude(free_d1)
free_d1_ph = get_phase(free_d1)
zer_phase = get_0_2pi(zernike_pha + free_d1_ph)
zer_field = free_d1_amp * torch.exp(1j * zer_phase)
free_d2 = Diffraction_propagation(zer_field, 0.1, dx, lambda_, 'Angular Spectrum', device)
free_d2_amp = get_amplitude(free_d2)
free_d2_ph = get_phase(free_d2)
len2_phs = lens_phase(X, Y, k, 0.1).to(device)  # lens2 phase
# len2_phs = pad_tensor(len2_phs, 1080, 1920, 0)
new_ph = get_0_2pi(free_d2_ph - len2_phs)
free_d2_field = get_hologram(free_d2_amp, new_ph)
free_d3 = Diffraction_propagation(free_d2_field, 0.1, dx, lambda_, 'Angular Spectrum', device)
free_d3_amp = get_amplitude(free_d3)
free_d3_ph = get_phase(free_d3)  # 0-2pi
# plt.imshow(free_d3_amp.squeeze(0).squeeze(0).cpu().data.numpy(), cmap='gray')
# plt.show()
# plt.imsave('usaf_abe_amp.png', free_d3_amp.squeeze(0).squeeze(0).cpu().numpy())
# plt.imshow(free_d3_ph.squeeze(0).squeeze(0).cpu().data.numpy(), cmap='gray')
# plt.show()
# imageio.imsave('usaf_abe_pha.png', phasemap_8bit(free_d3_ph, inverted=True))


def random_phase_recovery(sensor_abe, random_phase, d0, dx, lambda_, iter_num, method, device):
    if method == 'ASM':
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
    if method == 'FFT':
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



phase = torch.rand(8, 100, 100, dtype=torch.float64, device=device)
phase = F.interpolate(phase.unsqueeze(0), size=(1000, 1000), mode='bicubic', align_corners=False)
slm_plane = slm_plane_fft * torch.exp(1j * phase)
# sensor_plane = Diffraction_propagation(slm_plane, d0, dx, lambda_, device=device)
# sensor_abe = get_amplitude(sensor_plane)  # 1,8,1080,1920
sensor_plane = torch.fft.fftshift(torch.fft.fft2(slm_plane))
sensor_abe = get_amplitude(sensor_plane)
#  Get sensor intensity and random phase, then recovery intensity and phase of slm plane field
recovery_slm_field = random_phase_recovery(sensor_abe, phase, d0, dx, lambda_, iter_num, 'FFT', device)
est_abe_pha = get_phase(recovery_slm_field / torch.fft.fftshift(torch.fft.fft2(img)))
plt.imshow(est_abe_pha[0].cpu(), cmap='gray')
plt.title('slm plane pha')
plt.show()
imageio.imsave('est_abe_pha.png', phasemap_8bit(est_abe_pha, inverted=False))
recovery_sensor_field = torch.fft.fft2(recovery_slm_field * torch.exp(-1j * est_abe_pha))
sensor_intensity = get_amplitude(recovery_sensor_field[0])
plt.imshow(sensor_intensity.cpu(), cmap='gray')
plt.title('recovery sensor plane intensity')
plt.show()
plt.imsave('recovery_usaf.png', sensor_intensity.cpu().numpy())
# without_cor_sensor = torch.fft.fft2(recovery_slm_field)
# sensor_abe = get_amplitude(without_cor_sensor[0])
# plt.imshow(sensor_abe.cpu(), cmap='gray')
# plt.title('abe sensor plane intensity')
# plt.show()
# est_pha = get_phase(recovery_field[0])
# plt.imshow(est_pha.cpu(), cmap='gray')
# plt.show()

# imageio.imsave('est_pha.png', phasemap_8bit(ori_pha, inverted=True))
