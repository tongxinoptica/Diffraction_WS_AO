import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
from tqdm import tqdm

from Diffraction_H import Diffraction_propagation, get_amplitude, get_phase, get_0_2pi
from Zernike import generate_zer_poly

n = 1
# input zernike and crop
zernike_pha = cv2.imread('test_img/exp/4.23_zernike{}.png'.format(n), cv2.IMREAD_GRAYSCALE) / 255
zernike_pha = torch.tensor(zernike_pha, dtype=torch.float64) * 2 * torch.pi
zernike_pha = zernike_pha[40:1040, 460:1460]
plt.imshow(zernike_pha.numpy(), cmap='gray')
plt.show()

# input src
src = cv2.imread('test_img/exp/src.bmp', cv2.IMREAD_GRAYSCALE) / 255
src = cv2.resize(src, (1000, 1000), interpolation=cv2.INTER_AREA)
plt.imshow(src, cmap='gray')
plt.show()

# input ref and abe
ref = cv2.imread('test_img/exp/ref.bmp', cv2.IMREAD_GRAYSCALE) / 255
ref = cv2.resize(ref, (1000, 1000), interpolation=cv2.INTER_AREA)
# padding_size = (200, 200)
# ref = np.pad(ref, [padding_size, padding_size], mode='constant', constant_values=0)
plt.imshow(ref, cmap='gray')
plt.show()
abe = cv2.imread('test_img/exp/abe{}.bmp'.format(n), cv2.IMREAD_GRAYSCALE) / 255
abe = cv2.resize(abe, (1000, 1000), interpolation=cv2.INTER_AREA)
# padding_size = (200, 200)
# abe = np.pad(abe, [padding_size, padding_size], mode='constant', constant_values=0)
plt.imshow(abe, cmap='gray')
plt.show()

# Start propagation
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
lambda_ = 532e-9
pi = torch.tensor(torch.pi, dtype=torch.float64)
k = (2 * pi / lambda_)
dx = 8e-6
d = 0.1
num_iters = 2000

src = torch.tensor(src, dtype=torch.float64).to(device)
ref = torch.tensor(ref, dtype=torch.float64).to(device)
abe = torch.tensor(abe, dtype=torch.float64).to(device)
params = torch.nn.Parameter(torch.zeros_like(src))
optimizer = torch.optim.Adam([params], lr=0.01)
initial_lr = optimizer.param_groups[0]['lr']
loss_fn = torch.nn.MSELoss()
best_loss = 10.0

# Correct propagation phase with exp and sim
train_pro_pha = False
if train_pro_pha:
    pbar = tqdm(range(num_iters))
    for i in pbar:
        optimizer.zero_grad()
        obj_field = (src + 1e-9) * torch.exp(1j * 2 * pi * params)
        free_d0 = Diffraction_propagation(obj_field, d, dx, lambda_).to(device)
        free_d0_amp = get_amplitude(free_d0)
        free_d0_ph = get_phase(free_d0)
        current_lr = optimizer.param_groups[0]['lr']
        if i % 500 == 0:
            optimizer.param_groups[0]['lr'] = current_lr * 0.5
        with torch.no_grad():
            s = (free_d0_amp * ref).mean() / \
                (ref ** 2).mean()
        loss_val = loss_fn(s * free_d0_amp, ref)
        loss_val.backward()
        optimizer.step()
        with torch.no_grad():
            if loss_val < best_loss:
                pro_pha = params
                best_loss = loss_val.item()
                best_amp = s * free_d0_amp
        pbar.set_postfix(loss=f'{loss_val:.6f}', refresh=True)

    torch.save(pro_pha, 'parameter/grid_10_pro_pha.pth')
    plt.imshow(best_amp.cpu().data.numpy(), cmap='gray')
    plt.title('{}'.format(n))
    plt.show()

# Iterate zernike phase
pro_pha = torch.load('parameter/grid_10_pro_pha.pth')
# Create zernike polynomial
n_max = 15
Zer_radius = 400
zer_path = 'parameter/zernike_stack_{}_{}.pth'.format(n_max, Zer_radius)
if os.path.exists(zer_path):
    zernike_stack = torch.load(zer_path).to(device)  # zur_num,1,1000,1000
    zer_num = zernike_stack.shape[0]
else:
    print('Generate Zernike polynomial')
    zernike_stack, zer_num = generate_zer_poly(size=1000, dx=dx, n_max=n_max, radius=Zer_radius)

coeff = torch.zeros(zer_num, 1, 1, 1, device=device, dtype=torch.float64)
params = torch.nn.Parameter(coeff)
optimizer = torch.optim.Adam([params], lr=0.04)
pbar = tqdm(range(num_iters))
for i in pbar:
    optimizer.zero_grad()
    zer_pha = get_0_2pi((params * zernike_stack).sum(dim=0)).squeeze(0)
    obj_field = (src + 1e-9) * torch.exp(1j * 2 * pi * pro_pha) * torch.exp(1j * zer_pha)
    free_d0 = Diffraction_propagation(obj_field, d, dx, lambda_).to(device)
    free_d0_amp = get_amplitude(free_d0)
    free_d0_ph = get_phase(free_d0)
    current_lr = optimizer.param_groups[0]['lr']
    if i % 500 == 0:
        optimizer.param_groups[0]['lr'] = current_lr * 0.5
    with torch.no_grad():
        s = (free_d0_amp * abe).mean() / \
            (abe ** 2).mean()
    loss_val = loss_fn(s * free_d0_amp, abe)
    loss_val.backward()
    optimizer.step()
    with torch.no_grad():
        if loss_val < best_loss:
            pro_pha = params
            best_loss = loss_val.item()
            best_amp = s * free_d0_amp
    pbar.set_postfix(loss=f'{loss_val:.6f}', refresh=True)
pro_pha = get_0_2pi(pro_pha % (2 * pi))
plt.imshow(pro_pha.cpu().data.numpy(), cmap='gray')
plt.title('{}'.format(n))
plt.show()
