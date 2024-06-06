import os
import time

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
size = 768
n_max = 15
zer_radius = 400
iter_num = 100
x = torch.linspace(-size / 2, size / 2, size, dtype=torch.float64) * dx
y = torch.linspace(-size / 2, size / 2, size, dtype=torch.float64) * dx
X, Y = torch.meshgrid(x, y, indexing='xy')
rho = torch.sqrt(X ** 2 + (Y ** 2))
Phi = torch.atan2(Y, X)
img_path = 'usaf.png'
# img = creat_obj(img_path, size, radius=500, binaty_inv=2, if_obj=True, device=device)  # 0: Inverse; 1: NoInverse; 2: None
# img = img[200:800,200:800]
# img = torch.ones(768, 768, dtype=torch.float64, device=device)
# img = pad_tensor(img, 768, 768, 0)
# img = img.squeeze(0)
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) / 255
img = cv2.resize(img, (768,768), interpolation=cv2.INTER_CUBIC)
img = torch.tensor(img, device=device)
plt.imshow(img.data.cpu().numpy(), cmap='gray')
plt.show()
zer_path = '../parameter/zernike_stack_{}_{}.pth'.format(n_max, zer_radius)
if os.path.exists(zer_path):
    zernike_stack = torch.load(zer_path).to(device)  # zur_num,1,1000,1000
    zer_num = zernike_stack.shape[0]
else:
    print('Generate Zernike polynomial')
    zernike_stack, zer_num = generate_zer_poly(size=1000, dx=dx, n_max=n_max, radius=zer_radius, device=device)
zernike_pha = get_0_2pi(
    (torch.rand(zer_num, 1, 1, 1, dtype=torch.float64, device=device) * 2 * zernike_stack).sum(dim=0))
zernike_pha = zernike_pha[0][116:884, 116:884]
plt.imshow(zernike_pha.cpu(), cmap='gray')
plt.show()
# imageio.imsave('abe_pha.png', phasemap_8bit(zernike_pha, inverted=False))
# slm_plane = img*torch.exp(1j*zernike_pha)
# slm_plane = Diffraction_propagation(img, d0, dx, lambda_, device=device)
test = torch.fft.fftshift(torch.fft.fft2(img))
slm_plane_fft = test * torch.exp(1j * zernike_pha)
sensor_plane_fft = torch.fft.fft2(torch.fft.fftshift(slm_plane_fft))
ori_abe = get_amplitude(sensor_plane_fft)
ori_pha = get_phase(sensor_plane_fft)
plt.imshow(ori_abe.cpu(), cmap='gray')
plt.show()

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
            sensor_p = torch.fft.fft2(torch.fft.fftshift(init_u.unsqueeze(0) * torch.exp(1j * random_phase)))

            sensor_angle = get_phase(sensor_p)
            new_sensor = sensor_abe * torch.exp(1j * sensor_angle)
            # new_sensor = ((sensor_abe - get_amplitude(sensor_p)) * torch.rand(1, 8, 1, 1, device=device) + sensor_abe)
            # * torch.exp(1j * sensor_angle)
            new_slm = torch.fft.ifftshift(torch.fft.ifft2(new_sensor))  # Back prop
            init_u = torch.mean(new_slm * torch.exp(-1j * random_phase), dim=1)
        return init_u


def second_iterate(re_obj, init_u, sensor_abe, random_phase, iter_num, method, device):
    if method == 'FFT':
        est_abe_pha = get_phase(init_u / torch.fft.fftshift(torch.fft.fft2(re_obj)))
        init_u = torch.fft.fftshift(torch.fft.fft2(re_obj)) * torch.exp(1j * est_abe_pha)

        pbar = tqdm(range(iter_num))
        for i in pbar:
            sensor_p = torch.fft.fft2(torch.fft.fftshift(init_u.unsqueeze(0) * torch.exp(1j * random_phase)))

            sensor_angle = get_phase(sensor_p)
            new_sensor = sensor_abe * torch.exp(1j * sensor_angle)
            # new_sensor = ((sensor_abe - get_amplitude(sensor_p)) * torch.rand(1, 3, 1, 1, device=device) + sensor_abe) * torch.exp(1j * sensor_angle)
            new_slm = torch.fft.ifftshift(torch.fft.ifft2(new_sensor))  # Back prop
            init_u = torch.mean(new_slm * torch.exp(-1j * random_phase), dim=1)
            pha = get_phase(init_u / torch.fft.fftshift(torch.fft.fft2(re_obj)))
            init_u = torch.fft.fftshift(torch.fft.fft2(re_obj)) * torch.exp(1j * pha)

            # sensor_p2 = torch.fft.fftshift(torch.fft.fft2(init_u))
            # new_sensor2 = ((re_obj - get_amplitude(sensor_p2)) * torch.rand(1, device=device) + re_obj) * torch.exp(1j*get_phase(sensor_p2))
            # init_u = torch.fft.ifft2(torch.fft.ifftshift(new_sensor2))

        return init_u


phase = torch.rand(3, 100, 100, dtype=torch.float64, device=device)
phase = F.interpolate(phase.unsqueeze(0), size=(768, 768), mode='bicubic', align_corners=False)
phase = phase * torch.pi * 2
slm_plane = slm_plane_fft * torch.exp(1j * phase)
# sensor_plane = Diffraction_propagation(slm_plane, d0, dx, lambda_, device=device)
# sensor_abe = get_amplitude(sensor_plane)  # 1,8,1080,1920
sensor_plane = torch.fft.fft2(torch.fft.fftshift(slm_plane))
sensor_abe = get_amplitude(sensor_plane)
sensor_abe = sensor_abe / sensor_abe.amax(dim=(2, 3), keepdim=True)
plt.imsave('1.png', sensor_abe[0][0].cpu().numpy(), cmap='gray')
plt.imsave('2.png', sensor_abe[0][1].cpu().numpy(), cmap='gray')
plt.imsave('3.png', sensor_abe[0][2].cpu().numpy(), cmap='gray')

img1 = cv2.imread('1.png', cv2.IMREAD_GRAYSCALE) / 255.0
img2 = cv2.imread('2.png', cv2.IMREAD_GRAYSCALE) / 255.0
img3 = cv2.imread('3.png', cv2.IMREAD_GRAYSCALE) / 255.0

stacked_images = np.stack([img1, img2, img3], axis=0)
sensor_abe1 = torch.tensor(stacked_images, device=device).unsqueeze(0)
# plt.imshow(sensor_abe[0][0].cpu(), cmap='gray')
# plt.show()

noise_level = 0.1  # Adjustable parameter for noise level
noise = torch.rand(sensor_abe.shape, dtype=torch.float64, device=device) * noise_level
sensor_abe_noisy = sensor_abe + noise

#  Get sensor intensity and random phase, then recovery intensity and phase of slm plane field
time1 = time.time()
recovery_slm_field = random_phase_recovery(sensor_abe1, phase, d0, dx, lambda_, 20, 'FFT', device)
final_slm_field = second_iterate(img, recovery_slm_field, sensor_abe1, phase, 30, 'FFT', device)

est_abe_pha = get_phase(final_slm_field / test)
time2 = time.time()
plt.imshow(est_abe_pha[0].cpu(), cmap='gray')
plt.title('our')
plt.show()
# imageio.imsave('cnn_iter100.png', phasemap_8bit(est_abe_pha, inverted=False))

# Pure iteration
time3 = time.time()
recovery_slm_field2 = random_phase_recovery(sensor_abe1, phase, d0, dx, lambda_, 200, 'FFT', device)
est_abe_pha2 = get_phase(recovery_slm_field2 / test)
time4 = time.time()
print(time2 - time1)
print(time4 - time3)

plt.imshow(est_abe_pha2[0].cpu(), cmap='gray')
plt.title('pure_gs')
plt.show()
# imageio.imsave('iter100.png', phasemap_8bit(est_abe_pha2, inverted=False))

# gt_slm_field = sensor_p = torch.fft.fftshift(torch.fft.fft2(img)*torch.exp(1j*zernike_pha))
#
#
#
# recovery_sensor_field = torch.fft.fft2(gt_slm_field) * torch.exp(-1j * est_abe_pha)
# sensor_intensity1 = get_amplitude(recovery_sensor_field)
# plt.imsave('re_cnn100.png', sensor_intensity1[0].cpu().numpy(), cmap='gray')
#
# recovery_sensor_field = torch.fft.fft2(gt_slm_field) * torch.exp(-1j * est_abe_pha2)
# sensor_intensity2 = get_amplitude(recovery_sensor_field)
# plt.imsave('re_pure100.png', sensor_intensity2[0].cpu().numpy(), cmap='gray')
# delta_cnn = torch.angle(torch.exp(1j*zernike_pha) / torch.exp(1j*est_abe_pha))
# delta_pure = torch.angle(torch.exp(1j*zernike_pha) / torch.exp(1j*est_abe_pha2))
# delta_cnn = torch.sqrt(torch.mean(delta_cnn**2)).item()
# delta_pure = torch.sqrt(torch.mean(delta_pure**2)).item()
# print(delta_cnn, delta_pure)
# plt.imshow(sensor_intensity.cpu(), cmap='gray')
# plt.title('recovery sensor plane intensity')
# plt.show()
# imageio.imsave('est_abe_pha.png', phasemap_8bit(est_abe_pha, inverted=False))


# without_cor_sensor = torch.fft.fft2(recovery_slm_field)
# sensor_abe = get_amplitude(without_cor_sensor[0])
# plt.imshow(sensor_abe.cpu(), cmap='gray')
# plt.title('abe sensor plane intensity')
# plt.show()
# est_pha = get_phase(recovery_field[0])
# plt.imshow(est_pha.cpu(), cmap='gray')
# plt.show()

# imageio.imsave('est_pha.png', phasemap_8bit(ori_pha, inverted=True))
