import os

from unit import phasemap_8bit, pad_tensor, creat_obj
import cv2
from torch import nn
import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
from Diffraction_H import get_0_2pi, Diffraction_propagation, get_amplitude, get_phase, lens_phase, get_hologram, \
    random_phase_recovery, second_iterate
from Zernike import generate_zer_poly
import imageio
import torch.nn.functional as F

def again(re_obj, init_u, sensor_abe, random_phase, iter_num, method, device):
    if method == 'FFT':
        est_abe_pha = get_phase(init_u / re_obj)
        init_u = re_obj * torch.exp(1j*est_abe_pha)
        pbar = tqdm(range(iter_num))
        for i in pbar:
            sensor_p = torch.fft.fftshift(torch.fft.fft2(init_u.unsqueeze(0) * torch.exp(1j * random_phase)))
            sensor_angle = get_phase(sensor_p)
            # new_sensor = sensor_abe * torch.exp(1j * sensor_angle)
            new_sensor = ((sensor_abe - get_amplitude(sensor_p)) * torch.rand(1, 4, 1, 1, device=device) + sensor_abe) * torch.exp(1j * sensor_angle)
            new_slm = torch.fft.ifft2(torch.fft.ifftshift(new_sensor))  # Back prop
            init_u = torch.mean(new_slm * torch.exp(-1j * random_phase), dim=1)
            pha = get_phase(init_u / re_obj)
            init_u = re_obj * torch.exp(1j * pha)

        return init_u

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
lambda_ = 532e-9  # m
pi = torch.tensor(np.pi, dtype=torch.float64)
k = (2 * pi / lambda_)
dx = 8e-6  # m
d0 = 0.03  # m
size = 768
n_max = 15
zer_radius = 400
iter_num = 100
x = torch.linspace(-size / 2, size / 2, size, dtype=torch.float64) * dx
y = torch.linspace(-size / 2, size / 2, size, dtype=torch.float64) * dx
X, Y = torch.meshgrid(x, y, indexing='xy')
rho = torch.sqrt(X ** 2 + (Y ** 2))
Phi = torch.atan2(Y, X)
img_path = 'castle.png'
# img = creat_obj(img_path, size, radius=500, binaty_inv=2, if_obj=True, device=device)  # 0: Inverse; 1: NoInverse; 2: None

zer_path = '../parameter/zernike_stack_{}_{}.pth'.format(n_max, zer_radius)
if os.path.exists(zer_path):
    zernike_stack = torch.load(zer_path).to(device)  # zur_num,1,1000,1000
    zer_num = zernike_stack.shape[0]
else:
    print('Generate Zernike polynomial')
    zernike_stack, zer_num = generate_zer_poly(size=1000, dx=dx, n_max=n_max, radius=zer_radius, device=device)
zernike_pha = get_0_2pi(
    (torch.rand(zer_num, 1, 1, 1, dtype=torch.float64, device=device) * 1 * zernike_stack).sum(dim=0))
zernike_pha = zernike_pha[0][116:884, 116:884]
plt.imshow(zernike_pha.cpu(), cmap='gray')
plt.show()

img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) / 255
img = cv2.resize(img, (768,768), interpolation=cv2.INTER_CUBIC)
img = torch.tensor(img, dtype=torch.float64, device=device)
slm_plane = get_hologram(img, zernike_pha)

plt.imshow(get_amplitude(slm_plane).cpu(), cmap='gray')
plt.show()

phase = torch.rand(4, 70, 70, dtype=torch.float64, device=device)
phase = F.interpolate(phase.unsqueeze(0), size=(768, 768), mode='bicubic', align_corners=False)
phase = phase * torch.pi * 2
slm_plane = slm_plane * torch.exp(1j*phase)
slm_plane = Diffraction_propagation(slm_plane, 0.01*d0, dx, lambda_, device=device)
# sensor_plane = Diffraction_propagation(slm_plane, d0, dx, lambda_, device=device)

sensor_plane = torch.fft.fftshift(torch.fft.fft2(slm_plane))
sensor_abe = get_amplitude(sensor_plane)
sensor_abe = sensor_abe / sensor_abe.amax(dim=(2, 3), keepdim=True)
plt.imshow(sensor_abe[0][0].cpu(), cmap='gray')
plt.show()
recovery_slm_field = random_phase_recovery(sensor_abe, phase, d0, dx, lambda_, 800, 'FFT', device)
final_slm_field = again(img, recovery_slm_field, sensor_abe, phase, 800, 'FFT', device)
plt.imshow(get_amplitude(final_slm_field[0]).cpu(), cmap='gray')
plt.show()
plt.imshow(get_phase(final_slm_field[0]).cpu(), cmap='gray')
plt.show()










