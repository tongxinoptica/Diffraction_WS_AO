import os
import numpy as np
import imageio
from skimage import exposure
from unit import cal_grad
from unit import phasemap_8bit
import cv2
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from Diffraction_H import Diffraction_propagation, get_amplitude, get_phase, get_0_2pi
from Zernike import generate_zer_poly

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
lambda_ = 532e-9  # mm
pi = torch.tensor(torch.pi, dtype=torch.float64)
k = (2 * pi / lambda_)
dx = 8e-6  # m
d0 = 0.10  # m
size = 1000
mask_size = (size, size)
Zer_radius = 400
pupil_radium = 400
n_max = 15
w = 3e-3

in_pupil = torch.ones(mask_size, dtype=torch.float64).to(device)
x = torch.linspace(-size / 2, size / 2, size, dtype=torch.float64) * dx
y = torch.linspace(size / 2, -size / 2, size, dtype=torch.float64) * dx
X, Y = torch.meshgrid(x, y, indexing='xy')
rho = torch.sqrt(X ** 2 + Y ** 2)
angle = torch.atan2(Y, X)
mask = rho > dx * pupil_radium
# in_pupil[mask] = 0.0
gaussian = torch.exp(-rho ** 2 / w ** 2)
in_phase = cv2.imread('test.png', cv2.IMREAD_GRAYSCALE) / 255
in_phase = torch.tensor(in_phase, dtype=torch.float64, device=device) * 2 * torch.pi
in_phase = get_0_2pi(in_phase)
# plt.imshow(in_phase, cmap='gray')
# plt.show()

zer_path = 'parameter/zernike_stack_{}_{}.pth'.format(n_max, Zer_radius)
if os.path.exists(zer_path):
    zernike_stack = torch.load(zer_path).to(device)  # zur_num,1,1000,1000
    zer_num = zernike_stack.shape[0]
else:
    print('Generate Zernike polynomial')
    zernike_stack, zer_num = generate_zer_poly(size=size, dx=dx, n_max=n_max, radius=Zer_radius)

coeff = torch.rand(zer_num, 1, 1, 1, device=device, dtype=torch.float64)
zer_pha = get_0_2pi((coeff * zernike_stack).sum(dim=0)).squeeze(0)
padding_left = 460
padding_right = 460
padding_top = 40
padding_bottom = 40
zer_pha = F.pad(zer_pha, (padding_left, padding_right, padding_top, padding_bottom), "constant", 0)
plt.imshow(zer_pha.cpu(), cmap='gray')
plt.show()

random = torch.randn(1000, 1000)
obj_field = torch.exp(1j * in_phase)
free_d0 = Diffraction_propagation(obj_field, d0, dx, lambda_, device='cpu')
ref = get_amplitude(free_d0)

plt.imshow(ref.cpu().numpy(), cmap='gray')
plt.title('ref')
plt.show()
# plt.imsave('cor.png', I1, cmap='gray')
free_d1 = Diffraction_propagation(obj_field * torch.exp(1j*zer_pha), d0, dx, lambda_).to(device)
I0 = get_amplitude(free_d1)
I0 = (I0 - torch.mean(I0)) / torch.std(I0)
I0 = (I0 - torch.min(I0)) / (torch.max(I0) - torch.min(I0))
abe = exposure.match_histograms(I0.cpu().data.numpy(), ref.cpu().data.numpy())
# plt.imshow(I0.cpu().numpy(), cmap='gray')
# plt.show()
plt.imshow(abe, cmap='gray')
plt.title('abe')
plt.show()
# plt.imsave('ref.png', I0, cmap='gray')
abe = abe[40:1040, 460:1460]
ref = ref[40:1040, 460:1460]
delta_i = np.abs(abe - ref.numpy())
plt.imshow(delta_i, cmap='gray')
plt.show()
abe_gx, abe_gy = cal_grad(abe)
plt.imshow(abe_gx, cmap='gray')
plt.show()
plt.imshow(abe_gy, cmap='gray')
plt.show()
abe_grad = np.sqrt(abe_gx**2+abe_gy**2)
plt.imshow(abe_grad, cmap='gray')
plt.show()

ref_gx, ref_gy = cal_grad(ref)
plt.imshow(ref_gx, cmap='gray')
plt.show()
plt.imshow(ref_gy, cmap='gray')
plt.show()
ref_grad = np.sqrt(ref_gx**2+ref_gy**2)
plt.imshow(ref_grad, cmap='gray')
plt.show()

plt.imshow(np.abs(ref_gx - abe_gx), cmap='gray')
plt.show()
'''
img = cv2.imread('test_img/grid_10_slm.png', cv2.IMREAD_GRAYSCALE) / 255
img = torch.tensor(img, dtype=torch.float64, device=device)
num_iters = 5000
params = torch.nn.Parameter(torch.zeros_like(img))
optimizer = torch.optim.Adam([params], lr=0.04)
initial_lr = optimizer.param_groups[0]['lr']
loss_fn = torch.nn.MSELoss()
best_loss = 10.0
pbar = tqdm(range(num_iters))
for i in pbar:
    optimizer.zero_grad()
    obj1 = torch.exp(1j * params)  # Zernike
    free_d1 = Diffraction_propagation(obj1, d0, dx, lambda_)
    free_d1_amp = get_amplitude(free_d1)
    with torch.no_grad():
        s = (free_d1_amp * img).mean() / \
            (free_d1_amp ** 2).mean()
    loss_val = loss_fn(1 * free_d1_amp, img)
    # coeff_loss = torch.sum(torch.abs(params - coeff))
    loss_val.backward()
    optimizer.step()
    current_lr = optimizer.param_groups[0]['lr']
    if i % 400 == 0:
        optimizer.param_groups[0]['lr'] = current_lr * 0.8
    with torch.no_grad():
        if loss_val < best_loss:
            best_para = params.clone().detach()
            best_loss = loss_val.item()
            best_amp = 1 * free_d1_amp
    pbar.set_postfix(loss=f'{loss_val:.6f}', refresh=True)
print(best_loss)
plt.imshow(best_amp.squeeze(0).squeeze(0).cpu().data.numpy(), cmap='gray')
plt.show()
plt.imshow(best_para.cpu().data.numpy(), cmap='gray')
plt.show()
imageio.imwrite('test.png', phasemap_8bit(best_para))
'''
