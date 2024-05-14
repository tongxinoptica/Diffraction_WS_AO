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
d0 = 0.1  # m
size = 1000
mask_size = (size, size)
zer_radius = 500
pupil_radium = 400
n_max = 15
w = 3e-3
learning_rate = 0.01
batch = 1
zer_path = '../parameter/zernike_stack_{}_{}.pth'.format(n_max, zer_radius)
holo_path1 = 'gray_grid/test_10cm.png'
holo_path2 = 'gray_grid/test_9cm.png'
# Define zernike aberration
if os.path.exists(zer_path):
    zernike_stack = torch.load(zer_path).to(device)  # zur_num,1,1000,1000
    zer_num = zernike_stack.shape[0]
    x = torch.linspace(-size / 2, size / 2, size, dtype=torch.float64) * dx
    y = torch.linspace(size / 2, -size / 2, size, dtype=torch.float64) * dx
    X, Y = torch.meshgrid(x, y, indexing='xy')
    rho = torch.sqrt(X ** 2 + (Y ** 2))
    mask = rho <= dx * 400
    mask = mask.unsqueeze(0).unsqueeze(0)
    zernike_stack = zernike_stack * mask
    print('zer_num = {}'.format(zer_num))
else:
    print('Generate Zernike polynomial')
    zernike_stack, zer_num = generate_zer_poly(size=size, dx=dx, n_max=n_max, radius=zer_radius, device=device)
zer_num = zernike_stack.shape[0]
coeff = torch.rand(batch, zer_num, 1, 1, 1, dtype=torch.float64, device=device)
zernike_stack = zernike_stack.unsqueeze(0)
zer_pha = get_0_2pi((coeff * zernike_stack).sum(dim=1))  # size=(batch,1,1000,1000)
#  Load holo
in_phase1 = cv2.imread(holo_path1, cv2.IMREAD_GRAYSCALE) / 255.0
in_phase1 = torch.tensor(in_phase1[40:1040, 460:1460], dtype=torch.float64, device=device) * 2 * torch.pi
in_phase1 = get_0_2pi(in_phase1.unsqueeze(0).unsqueeze(0))
obj_field = torch.exp(1j * in_phase1) * torch.exp(1j * zer_pha)
mix_pha = get_phase(obj_field)
imageio.imsave('mix_phase.png', phasemap_8bit(mix_pha, inverted=True))
abe1_complex = Diffraction_propagation(obj_field, d0, dx, lambda_, device=device)
abe1 = get_amplitude(abe1_complex)  # Capture image with abe
plt.imshow(abe1[0].squeeze(0).detach().numpy(), cmap='gray')
plt.title('abe1')
plt.show()
in_phase2 = cv2.imread(holo_path1, cv2.IMREAD_GRAYSCALE) / 255.0
in_phase2 = torch.tensor(in_phase2, dtype=torch.float64, device=device) * 2 * torch.pi
in_phase2 = get_0_2pi(in_phase2.unsqueeze(0).unsqueeze(0))
padding_left = 460
padding_right = 460
padding_top = 40
padding_bottom = 40
zer_pha = F.pad(zer_pha, (padding_left, padding_right, padding_top, padding_bottom), "constant", 0)
obj_field = torch.exp(1j * in_phase2) * torch.exp(1j * zer_pha)
mix_pha2 = get_phase(obj_field)
imageio.imsave('mix_phase2.png', phasemap_8bit(mix_pha2, inverted=True))
abe2_complex = Diffraction_propagation(obj_field, d0, dx, lambda_, device=device)
abe2 = get_amplitude(abe2_complex)  # Capture image with abe
plt.imshow(abe2[0].squeeze(0).detach().numpy(), cmap='gray')
plt.title('abe2')
plt.show()
