import os
from torch import nn
from torchvision import models
import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from unit import sobel_grad
from Diffraction_H import get_0_2pi, Diffraction_propagation, get_amplitude
from Zernike import generate_zer_poly
from resnet50 import ResNet50

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = 'cpu'
lambda_ = 532e-9  # mm
pi = torch.tensor(np.pi, dtype=torch.float64)
k = (2 * pi / lambda_)
dx = 8e-6  # m
d0 = 0.10  # m
size = 1000
mask_size = (size, size)
Zer_radius = 400
pupil_radium = 400
n_max = 15
w = 3e-3
batch = 4
zer_path = '../parameter/zernike_stack_{}_{}.pth'.format(n_max, Zer_radius)

# Define zernike aberration
if os.path.exists(zer_path):
    zernike_stack = torch.load(zer_path).to(device)  # zur_num,1,1000,1000
    zer_num = zernike_stack.shape[0]
else:
    print('Generate Zernike polynomial')
    zernike_stack, zer_num = generate_zer_poly(size=size, dx=dx, n_max=n_max, radius=Zer_radius)

coeff = torch.rand(batch, zer_num, 1, 1, 1, dtype=torch.float64, device=device)
zernike_stack = zernike_stack.unsqueeze(0)
zer_pha = get_0_2pi((coeff * zernike_stack).sum(dim=1)).squeeze(0)  # size=(batch,1,1000,1000)

#  Load holo
in_phase = cv2.imread('../test.png', cv2.IMREAD_GRAYSCALE) / 255
in_phase = torch.tensor(in_phase[40:1040, 460:1460], dtype=torch.float64, device=device) * 2 * torch.pi
in_phase = get_0_2pi(in_phase.unsqueeze(0).unsqueeze(0))
obj_field = torch.exp(1j * in_phase)
abe_complex = Diffraction_propagation(obj_field * torch.exp(1j*zer_pha), d0, dx, lambda_, device=device)
abe = get_amplitude(abe_complex)  # Capture image with abe
ref_complex = Diffraction_propagation(obj_field, d0, dx, lambda_, device=device)
ref = get_amplitude(ref_complex)  # Capture image without abe
abe_gx, abe_gy = sobel_grad(abe)
ref_gx, ref_gy = sobel_grad(ref)
delta_intensity = abe - ref
delta_gx = abe_gx - ref_gx
delta_gy = abe_gy - ref_gy  # size=(batch,1,1000,1000)
input = torch.cat((delta_intensity, delta_gx, delta_gy), dim=1)
#  Load resnet50
model = ResNet50().to(device)
output = model(input.float())
print(output.shape)

