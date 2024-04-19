import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage import draw
from Diffraction_H import lens_phase, Diffraction_propagation, get_phase, get_amplitude, get_hologram, get_0_2pi
from Zernike import zernike
from unit import creat_obj, phasemap_8bit
import imageio

lambda_ = 633e-9  #
k = 2 * np.pi / lambda_
f = 0.1  # focal length
d0 = 0.5 * f  # dis between abe and obj
d1 = 0.5 * f  # dis between obj and lens1
d2 = 2 * f  # dis between lens1 and lens2
d3 = 1 * f  # dis between lens2 and observe
size = 1000  #
dx = 8e-6  #
sample_path = 'grid_4.png'
if_obj = True
Add_zernike = True
phase_correct = False
# Create object and binary
obj = creat_obj(sample_path, size, binaty_inv=True, if_obj=if_obj)
# plt.imshow(obj.data.numpy(), cmap='gray')
# plt.show()

x = torch.linspace(-size / 2, size / 2, size) * dx
y = torch.linspace(size / 2, -size / 2, size) * dx
X, Y = torch.meshgrid(x, y, indexing='xy')

# Create zernike aberration
rho = torch.sqrt(X ** 2 + Y ** 2)
theta = torch.arctan2(Y, X)
mask = rho > dx * 350
rho[mask] = 0.0
rho = rho / torch.max(rho)
zer = 2 * zernike(5, 3, rho, theta) + 2 * zernike(4, 0, rho, theta) + 3 * zernike(7, -3, rho, theta) + \
      5 * zernike(6, 5, rho, theta) + 3 * zernike(9, 5, rho, theta) + 5 * zernike(11, 5, rho, theta)
zer = zer % (2 * torch.pi)  # 0-2pi
zer[mask] = 0.0
# plt.imshow(zer.data.numpy(), cmap='gray')
# plt.show()
# imageio.imwrite('./test_img/zer_1.png', phasemap_8bit(zer))
# plt.imsave('./test_img/zer_1.png', zer.data.numpy(), cmap='gray')

# Propagate to abe
obj_field = obj.unsqueeze(0).unsqueeze(0)
free_d0 = Diffraction_propagation(obj_field, d0, dx, lambda_)
free_d0_amp = get_amplitude(free_d0)
free_d0_ph = get_phase(free_d0)
if Add_zernike:
    # Propagate abe to lens
    zer_phase = zer.unsqueeze(0).unsqueeze(0) + free_d0_ph  # Add zernike phase
    zer_phase = get_0_2pi(zer_phase)
    zer_field = free_d0_amp * torch.exp(1j * zer_phase)  # Zernike field
    free_d1 = Diffraction_propagation(zer_field, d1, dx, lambda_)
    free_d1_amp = get_amplitude(free_d1)
    free_d1_ph = get_phase(free_d1)
    # plt.imshow(free_d0_amp.squeeze(0).squeeze(0), cmap='gray')
    # plt.show()
else:
    # propagate to lens
    free_d1 = Diffraction_propagation(free_d0, d1, dx, lambda_)
    free_d1_amp = get_amplitude(free_d1)
    free_d1_ph = get_phase(free_d1)

len1_phs = lens_phase(X, Y, k, f)  # lens1 phase
new_ph = get_0_2pi(free_d1_ph - len1_phs)
free_d1_field = get_hologram(free_d1_amp, new_ph)

# Propagate lens1 to lens2
free_d2 = Diffraction_propagation(free_d1_field, d2, dx, lambda_)
free_d2_amp = get_amplitude(free_d2)
free_d2_ph = get_phase(free_d2)
len2_phs = lens_phase(X, Y, k, f)  # lens1 phase
new_ph = get_0_2pi(free_d2_ph - len2_phs)
free_d2_field = get_hologram(free_d2_amp, new_ph)

# Propagate lens2 to observe
free_d3 = Diffraction_propagation(free_d2_field, d3, dx, lambda_)
free_d3_amp = get_amplitude(free_d3)
free_d3_ph = get_phase(free_d3)  # 0-2pi
# free_d3_ph = free_d3_ph / (2 * torch.pi)

plt.imshow(free_d3_amp.squeeze(0).squeeze(0).data.numpy(), cmap='gray')
plt.show()
# plt.imsave('./test_img/g4_f_zer_amp.png', free_d3_amp.squeeze(0).squeeze(0).data.numpy(),cmap='gray')
plt.imshow(free_d3_ph.squeeze(0).squeeze(0).data.numpy(), cmap='gray')
plt.show()
# imageio.imwrite('./test_img/g8_0.5f_nozer_ph.png', phasemap_8bit(free_d3_ph, False))
# plt.imsave('./test_img/obj_0.5f_zer_ph.png', free_d3_ph.squeeze(0).squeeze(0).numpy(), cmap='gray')

if phase_correct:
    d4 = 0.5 * f
    ph_zer = cv2.imread('test_img/g4_0.5f_zer_ph.png', cv2.IMREAD_GRAYSCALE)
    ph_nozer = cv2.imread('test_img/g4_0.5f_nozer_ph.png', cv2.IMREAD_GRAYSCALE)
    ph_zer = (torch.tensor(ph_zer) / 255) * 2 * torch.pi
    ph_nozer = (torch.tensor(ph_nozer) / 255) * 2 * torch.pi
    del_phase = get_0_2pi(ph_nozer - ph_zer).unsqueeze(0).unsqueeze(0)  # 0-2pi
    plt.imshow(del_phase.squeeze(0).squeeze(0).data.numpy(), cmap='gray')
    plt.show()
    corr_ph = get_0_2pi(free_d3_ph + ph_nozer - ph_zer)
    free_d4_field = free_d3_amp * torch.exp(1j * corr_ph)
    free_d4 = Diffraction_propagation(free_d4_field, d4, dx, lambda_)
    free_d4_amp = get_amplitude(free_d4)
    free_d4_ph = get_phase(free_d4)  # 0-2pi
    plt.imshow(free_d4_amp.squeeze(0).squeeze(0).data.numpy(), cmap='gray')
    plt.show()
    # plt.imsave('./test_img/g4_f_com_amp.png', free_d4_amp.squeeze(0).squeeze(0).data.numpy(),cmap='gray')
