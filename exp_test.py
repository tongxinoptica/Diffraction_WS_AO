import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
from tqdm import tqdm
from pytorch_msssim import ssim
from Diffraction_H import Diffraction_propagation, get_amplitude, get_phase, get_0_2pi
from Zernike import generate_zer_poly
from unit import twossim


n = 2
# input zernike and crop
zernike_pha = cv2.imread('test_img/exp/4.23_zernike{}.png'.format(n), cv2.IMREAD_GRAYSCALE) / 255
zernike_pha = torch.tensor(zernike_pha, dtype=torch.float64) * 2 * torch.pi
zernike_pha = zernike_pha[40:1040, 460:1460]
plt.imshow(zernike_pha.numpy(), cmap='gray')
plt.show()

# input src
src = cv2.imread('test_img/exp/process_grid10/src.bmp', cv2.IMREAD_GRAYSCALE)
src = cv2.resize(src, (1000, 1000), interpolation=cv2.INTER_AREA)
src = src.astype(np.float64) / 255
src = (src - src.mean()) / src.std()
src = (src - np.min(src)) / (np.max(src) - np.min(src))
plt.imshow(src, cmap='gray')
plt.show()

# input ref and abe
ref = cv2.imread('test_img/exp/process_grid10/ref.bmp', cv2.IMREAD_GRAYSCALE)
ref = cv2.resize(ref, (1000, 1000), interpolation=cv2.INTER_AREA)
ref = ref.astype(np.float64) / 255
ref = (ref - ref.mean()) / ref.std()
ref = (ref - np.min(ref)) / (np.max(ref) - np.min(ref))
# padding_size = (200, 200)
# ref = np.pad(ref, [padding_size, padding_size], mode='constant', constant_values=0)
plt.imshow(ref, cmap='gray')
plt.show()
abe = cv2.imread('test_img/exp/process_grid10/abe{}.bmp'.format(n), cv2.IMREAD_GRAYSCALE)
abe = cv2.resize(abe, (1000, 1000), interpolation=cv2.INTER_AREA)
abe = abe.astype(np.float64) / 255
abe = (abe - abe.mean()) / abe.std()
abe = (abe - np.min(abe)) / (np.max(abe) - np.min(abe))
# padding_size = (200, 200)
# abe = np.pad(abe, [padding_size, padding_size], mode='constant', constant_values=0)
plt.imshow(abe, cmap='gray')
plt.show()

# Start propagation
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
lambda_ = 532e-9
pi = torch.tensor(torch.pi, dtype=torch.float64)
k = (2 * pi / lambda_)
dx = 8e-6
d = 0.1
size = 1000
n_max = 15
Zer_radius = 400
num_iters = 2000
x = torch.linspace(-size / 2, size / 2, size, dtype=torch.float64) * dx
y = torch.linspace(size / 2, -size / 2, size, dtype=torch.float64) * dx
X, Y = torch.meshgrid(x, y, indexing='xy')
rho = torch.sqrt((X) ** 2 + ((Y) ** 2))
mask = rho > dx * 500

src = torch.tensor(src, dtype=torch.float64).to(device)
ref = torch.tensor(ref, dtype=torch.float64).to(device)
abe = torch.tensor(abe, dtype=torch.float64).to(device)
#abe[mask] = 0

# Correct propagation phase with exp and sim
train_pro_pha = False
if train_pro_pha:
    params = torch.nn.Parameter(torch.zeros_like(src))
    optimizer = torch.optim.Adam([params], lr=0.1)
    initial_lr = optimizer.param_groups[0]['lr']
    loss_fn = torch.nn.MSELoss()
    best_loss = 10.0
    pbar = tqdm(range(num_iters))
    for i in pbar:
        optimizer.zero_grad()
        obj_field = src * torch.exp(1j * 2 * pi * params)
        free_d0 = Diffraction_propagation(obj_field, d, dx, lambda_).to(device)
        free_d0_amp = get_amplitude(free_d0)
        # free_d0_amp[mask] = 0
        current_lr = optimizer.param_groups[0]['lr']
        if i % 500 == 0:
            optimizer.param_groups[0]['lr'] = current_lr * 0.9
        with torch.no_grad():
            s = (free_d0_amp * ref).mean() / \
                (ref ** 2).mean()
        # free_d0_amp = (free_d0_amp - torch.mean(free_d0_amp)) / torch.std(free_d0_amp)
        # free_d0_amp = (free_d0_amp - torch.min(free_d0_amp)) / (torch.max(free_d0_amp) - torch.min(free_d0_amp))
        # loss_val = loss_fn(free_d0_amp, ref) + 1 - twossim(free_d0_amp, ref) + \
        #            torch.abs(torch.mean(free_d0_amp) - torch.mean(ref))
        loss_val = loss_fn(s * free_d0_amp, ref)
        loss_val.backward()
        optimizer.step()
        with torch.no_grad():
            if loss_val < best_loss:
                pro_pha = params.clone().detach()
                best_loss = loss_val.item()
                best_amp = s * free_d0_amp
        pbar.set_postfix(loss=f'{best_loss:.6f}', refresh=True)

    torch.save(pro_pha, 'parameter/grid_10_pro_pha.pth')
    plt.imshow(best_amp.cpu().data.numpy(), cmap='gray')
    plt.title('{}'.format(n))
    plt.show()

# Iterate zernike phase
pro_pha = torch.load('parameter/grid_10_pro_pha.pth', map_location=device)
pro_pha.requires_grad_(False)

# Create zernike polynomial

zer_path = 'parameter/zernike_stack_{}_{}.pth'.format(n_max, Zer_radius)
if os.path.exists(zer_path):
    zernike_stack = torch.load(zer_path).to(device)  # zur_num,1,1000,1000
    zer_num = zernike_stack.shape[0]
else:
    print('Generate Zernike polynomial')
    zernike_stack, zer_num = generate_zer_poly(size=1000, dx=dx, n_max=n_max, radius=Zer_radius)
obj0 = src * torch.exp(1j * 2 * pi * pro_pha)
coeff = torch.zeros(zer_num, 1, 1, 1, device=device, dtype=torch.float64)
params = torch.nn.Parameter(coeff, requires_grad=True)
optimizer = torch.optim.Adam([params], lr=0.01)
best_loss = 10.0
loss_fn = torch.nn.MSELoss()
pbar = tqdm(range(1000))
for i in pbar:
    optimizer.zero_grad()
    zer_pha = get_0_2pi((params * zernike_stack).sum(dim=0)).squeeze(0).to(device)
    obj_field = src * torch.exp(1j * zer_pha)
    free_d0 = Diffraction_propagation(obj_field, d, dx, lambda_).to(device)
    free_d0_amp = get_amplitude(free_d0)
    #free_d0_amp[mask] = 0
    current_lr = optimizer.param_groups[0]['lr']
    if i % 500 == 0:
        optimizer.param_groups[0]['lr'] = current_lr * 1
    # with torch.no_grad():
    #     s = (free_d0_amp * abe).mean() / \
    #         (abe ** 2).mean()
    free_d0_amp = (free_d0_amp - torch.mean(free_d0_amp)) / torch.std(free_d0_amp)
    free_d0_amp = (free_d0_amp - torch.min(free_d0_amp)) / (torch.max(free_d0_amp) - torch.min(free_d0_amp))

    loss_val = 1 - twossim(free_d0_amp, abe) + loss_fn(free_d0_amp, abe) + \
               torch.abs(torch.mean(free_d0_amp) - torch.mean(abe))
    # loss_val = loss_fn(free_d0_amp, abe)
    loss_val.backward()
    optimizer.step()
    # with torch.no_grad():
    #     params.data.clamp(0, 1)
    with torch.no_grad():
        if loss_val < best_loss:
            best_para = params.clone().detach()
            estimate_abe = get_0_2pi((best_para * zernike_stack).sum(dim=0)).squeeze(0)
            best_loss = loss_val.item()
            best_amp = 1 * free_d0_amp.clone().detach()
            loss_percent = torch.sum(torch.abs(free_d0_amp - abe)) / torch.sum(abe)
            print(loss_percent)
            # plt.imshow(best_amp.cpu().data.numpy(), cmap='gray')
            # plt.show()
    pbar.set_postfix(loss=f'{best_loss:.6f}', refresh=True)
estimate_abe = get_0_2pi(estimate_abe % (2 * pi))
plt.imshow(estimate_abe.cpu().data.numpy(), cmap='gray')
plt.title('{}'.format(n))
plt.show()
plt.imshow(best_amp.cpu().data.numpy(), cmap='gray')
plt.show()
