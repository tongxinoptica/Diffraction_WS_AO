import os
from torch import nn
import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
from Diffraction_H import get_0_2pi, Diffraction_propagation, get_amplitude
from Zernike import generate_zer_poly
from resnet50 import ResNet50
from generate_data import train_data, random_abe

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
lambda_ = 532e-9  # mm
pi = torch.tensor(np.pi, dtype=torch.float64)
k = (2 * pi / lambda_)
dx = 8e-6  # m
d0 = 0.10  # m
size = 1000
mask_size = (size, size)
zer_radius = 400
pupil_radium = 400
n_max = 10
w = 3e-3
learning_rate = 0.01
batch = 1
cut = True
zer_path = '../parameter/zernike_stack_{}_{}.pth'.format(n_max, zer_radius)
holo_path = 'gray_grid/test_10cm.png'
tur_path = 'grid10/3.jpg'
# Define zernike aberration
if os.path.exists(zer_path):
    zernike_stack = torch.load(zer_path).to(device)  # zur_num,1,1000,1000
    zer_num = zernike_stack.shape[0]
    if cut:
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


#  Load resnet50
loss_mse = nn.MSELoss()
loss_l1 = nn.L1Loss()
model = ResNet50(num_classes=zer_num).to(device)
model.load_state_dict(torch.load('gray_grid/u50_240000.pth', map_location=device))
print('Weight Loaded')
input, coeff, obj_field, ref, abe = train_data(batch, zernike_stack, holo_path, d0, dx, lambda_, device)
# input, obj_field, ref, abe = random_abe(tur_path, size, holo_path, d0, dx, lambda_, device)
zer_phase = get_0_2pi((coeff * zernike_stack).sum(dim=1))
plt.imshow(zer_phase[0].squeeze(0).detach().numpy(), cmap='gray')
plt.title('abe')
plt.show()
output = model(input.float().to(device))
out_coeff = output.unsqueeze(4)  # size=(batch,zer_num,1,1,1)
est_zer_phase = get_0_2pi((out_coeff * zernike_stack).sum(dim=1))
plt.imshow(est_zer_phase[0].squeeze(0).detach().numpy(), cmap='gray')
plt.title('est_abe')
plt.show()
out_ref_com = Diffraction_propagation(obj_field * torch.exp(-1j * est_zer_phase), d0, dx, lambda_, device=device)
out_ref = get_amplitude(out_ref_com)
plt.imshow(out_ref[0].squeeze(0).detach().numpy(), cmap='gray')
plt.title('cor')
plt.show()
plt.imshow(ref[0].squeeze(0).detach().numpy(), cmap='gray')
plt.title('raw')
plt.show()
plt.imshow(abe[0].squeeze(0).detach().numpy(), cmap='gray')
plt.title('abe')
plt.show()
delta_phase = torch.sum(torch.abs(torch.exp(1j*zer_phase)) - torch.abs(torch.exp(1j*est_zer_phase)))
print(delta_phase)

