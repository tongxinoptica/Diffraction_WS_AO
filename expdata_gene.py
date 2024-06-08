import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from Diffraction_H import get_0_2pi, get_amplitude, random_phase_recovery, get_phase, second_iterate, \
    Diffraction_propagation
from Zernike import generate_zer_poly
from unit import pad_array, phasemap_8bit, pad_tensor
import imageio
import cv2

# aline_a = np.ones((600,600))
# aline_a = pad_array(aline_a, 1080, 1920, 0)
# aline_a = (aline_a * 255).astype(np.uint8)
# imageio.imwrite('./exp_data_gene/aline_a2.png', aline_a.squeeze(0))

# for i in range(1,12):
#     img = cv2.imread('./exp_data_gene/gt/{}.png'.format(i), cv2.IMREAD_GRAYSCALE) / 255
#     img = cv2.resize(img, (600,600), interpolation=cv2.INTER_CUBIC)
#     img = (img >= 0.5).astype(np.float32)
#     # img = (img + 1) / 2
#     img = pad_array(img, 1080, 1920, 0)
#     img = (img * 255).astype(np.uint8)
#     imageio.imwrite('./exp_data_gene/obj{}.png'.format(i), img.squeeze(0))

# zernike_stack, zer_num = generate_zer_poly(size=600, dx=8e-6, n_max=15, radius=250)  # 0-2pi
# for i in range(1,11):
#     zernike_pha = get_0_2pi((torch.rand(zer_num, 1, 1, 1, dtype=torch.float64) * 1 * zernike_stack).sum(dim=0))
#     zernike_pha = pad_tensor(zernike_pha, 1080,1920,0)
#     imageio.imsave('./exp_data_gene/abe/{}.png'.format(i), phasemap_8bit(zernike_pha, inverted=False))
#     for j in range(1, 11):
#         phase = torch.rand(1, 60, 60, dtype=torch.float64)
#         phase = F.interpolate(phase.unsqueeze(0), size=(600, 600), mode='bicubic', align_corners=False)
#         phase = pad_tensor(phase[0], 1080,1920,0)
#         phase = phase * torch.pi * 2
#         imageio.imsave('./exp_data_gene/rand_p/{}_{}.png'.format(i,j), phasemap_8bit(phase, inverted=False))
#
#         slm_p = phase + zernike_pha
#         imageio.imsave('./exp_data_gene/slm_p/{}_{}.png'.format(i,j), phasemap_8bit(slm_p, inverted=False))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dx = 8e-6  # m
d0 = 0.05
lambda_ = 532e-9
obj = cv2.imread('./exp_data_gene/obj2/obj1.png', cv2.IMREAD_GRAYSCALE) / 255
# obj = pad_array(obj, 1080,1920)
obj = torch.tensor(obj, dtype=torch.float64, device=device)
slmp1 = cv2.imread('./exp_data_gene/slm_p/1_1.png', cv2.IMREAD_GRAYSCALE) / 255
slmp1 = torch.tensor(slmp1, dtype=torch.float64) * 2 * torch.pi
slmp2 = cv2.imread('./exp_data_gene/slm_p/1_2.png', cv2.IMREAD_GRAYSCALE) / 255
slmp2 = torch.tensor(slmp2, dtype=torch.float64) * 2 * torch.pi
slmp3 = cv2.imread('./exp_data_gene/slm_p/1_3.png', cv2.IMREAD_GRAYSCALE) / 255
slmp3 = torch.tensor(slmp3, dtype=torch.float64) * 2 * torch.pi
p1 = cv2.imread('./exp_data_gene/asm_p/1_1.png', cv2.IMREAD_GRAYSCALE) / 255
p1 = torch.tensor(p1, dtype=torch.float64) * 2 * torch.pi
p2 = cv2.imread('./exp_data_gene/asm_p/1_2.png', cv2.IMREAD_GRAYSCALE) / 255
p2 = torch.tensor(p2, dtype=torch.float64) * 2 * torch.pi
p3 = cv2.imread('./exp_data_gene/asm_p/1_3.png', cv2.IMREAD_GRAYSCALE) / 255
p3 = torch.tensor(p3, dtype=torch.float64) * 2 * torch.pi
abe = cv2.imread('./exp_data_gene/abe/1.png', cv2.IMREAD_GRAYSCALE) / 255
abe = torch.tensor(abe, dtype=torch.float64, device=device)
abe = abe * torch.pi * 2
slmp = torch.stack([slmp1, slmp2, slmp3], dim=0).to(device)
phase = torch.stack([p1, p2, p3], dim=0).to(device)
obj = obj[:, 420:1500]
slmp = slmp[:, :, 420:1500]
# phase = phase[:, :, 420:1500]

size = 1080
x = torch.linspace(-size / 2, size / 2, size, dtype=torch.float64) * dx
y = torch.linspace(-size / 2, size / 2, size, dtype=torch.float64) * dx
X, Y = torch.meshgrid(x, y, indexing='xy')
rho = torch.sqrt(X ** 2 + Y ** 2)
phi = torch.atan2(Y, X)
vortex_phase = torch.exp(1j * 4 * phi).to(device)
slm_plane = Diffraction_propagation(obj, 2 * d0, dx, lambda_, device=device)
plt.imshow(get_amplitude(slm_plane).cpu(), cmap='gray')
plt.show()
plt.imshow(get_phase(slm_plane).cpu(), cmap='gray')
plt.show()

# phase = torch.rand(3, 108, 108, dtype=torch.float64, device=device)
# phase = F.interpolate(phase.unsqueeze(0), size=(1080, 1080), mode='bicubic', align_corners=False)
# phase = phase[0] * torch.pi * 2

slm_plane = slm_plane * torch.exp(1j*phase)
sensor_plane = Diffraction_propagation(slm_plane, d0, dx, lambda_, device=device)
sensor_abe = get_amplitude(sensor_plane)
# sensor_abe = sensor_abe / sensor_abe.amax(dim=(2, 3), keepdim=True)
recovery_slm_field = random_phase_recovery(get_amplitude(slm_plane), sensor_abe.unsqueeze(0), phase, d0, dx, lambda_, 1000, 'ASM', device)
ori = Diffraction_propagation(recovery_slm_field, -2.0*d0, dx, lambda_, device=device)
plt.imshow(get_amplitude(recovery_slm_field[0]).cpu(), cmap='gray')
plt.show()
plt.imshow(get_phase(recovery_slm_field[0]).cpu(), cmap='gray')
plt.show()
'''
######FFT###########
slm_plane = torch.fft.fftshift(torch.fft.fft2(obj)) * torch.exp(1j * slmp)
sensor_plane = torch.fft.fft2(torch.fft.fftshift(slm_plane))
sensor_abe = get_amplitude(sensor_plane)
sensor_abe = sensor_abe / sensor_abe.amax(dim=(1, 2), keepdim=True)
noise_level = 0.  # Adjustable parameter  for noise level
noise = torch.randn(sensor_abe.shape, dtype=torch.float64) * noise_level
plt.imshow(sensor_abe[0].cpu(), cmap='gray')
plt.show()
plt.imshow(sensor_abe[1].cpu(), cmap='gray')
plt.show()
plt.imshow(sensor_abe[2].cpu(), cmap='gray')
plt.show()
'''

# recovery_slm_field = random_phase_recovery(sensor_abe.unsqueeze(0), phase, d0, dx, lambda_, 40, 'FFT',device=device)
# final_slm_field = second_iterate(obj, recovery_slm_field, sensor_abe, phase, 60, 'FFT', device=device)
# est_abe_pha = get_phase(final_slm_field / torch.fft.fftshift(torch.fft.fft2(obj)))
# plt.imshow(est_abe_pha[0].cpu(), cmap='gray')
# plt.show()
# sensor = get_amplitude(torch.fft.ifft2(final_slm_field * torch.exp(-1j * est_abe_pha)))
# plt.imshow(sensor[0].cpu(), cmap='gray')
# plt.show()
