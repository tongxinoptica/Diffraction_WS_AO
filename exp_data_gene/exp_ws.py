import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from Diffraction_H import get_0_2pi, get_amplitude, random_phase_recovery, get_phase, second_iterate
from Zernike import generate_zer_poly
from unit import pad_array, phasemap_8bit, pad_tensor
import imageio
import cv2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dx = 8e-6  # m
d0 = 0.03
lambda_ = 532e-9

obj = cv2.imread('tx/1.tif', cv2.IMREAD_GRAYSCALE) / 255
# obj = obj[:, 420:1500]
# obj = cv2.imread('tx/1.tif', cv2.IMREAD_GRAYSCALE) / 255
obj = cv2.resize(obj, (1080, 1080), interpolation=cv2.INTER_CUBIC)
obj = obj/np.max(obj)
# obj = pad_array(obj, 1080, 1920)
obj = torch.tensor(obj, dtype=torch.float64, device=device)
# plt.imshow(obj.cpu(), cmap='gray')
# plt.show()
# slmp1 = cv2.imread('slm_p/1_1.png', cv2.IMREAD_GRAYSCALE) / 255
#
# slmp1 = torch.tensor(slmp1, dtype=torch.float64, device=device) * 2 * torch.pi
# slm_plane = torch.fft.fftshift(torch.fft.fft2(obj)) * torch.exp(1j * slmp1)
# sensor_plane = torch.fft.fft2(torch.fft.fftshift(slm_plane))
# sensor_abe = get_amplitude(sensor_plane)
# plt.imshow(sensor_abe.cpu(), cmap='gray')
# plt.show()

abe = cv2.imread('abe/1.png', cv2.IMREAD_GRAYSCALE) / 255
abe = torch.tensor(abe[:, 420:1500], dtype=torch.float64, device=device)
abe = abe * torch.pi * 2

image_paths = [f'tx/1.{i}.tif' for i in range(1, 11)]
images = []
for path in image_paths:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) / 255.0
    img = cv2.resize(img, (1080, 1080), interpolation=cv2.INTER_CUBIC)
    img / img.max()
    # img = pad_array(img, 1080, 1920)
    images.append(img)
intensity = np.stack(images, axis=0)
intensity = torch.tensor(intensity, device=device, dtype=torch.float64)  # 1,10,1080,1080

image_paths = [f'rand_p/1_{i}.png' for i in range(1, 11)]
b = []
for path in image_paths:
    # 读取图像
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) / 255.0
    img = img[:, 420:1500]
    # plt.imshow(img, cmap='gray')
    # plt.show()
    # img = cv2.resize(img, (1080, 1080), interpolation=cv2.INTER_CUBIC)
    b.append(img * np.pi * 2)
rand_p = np.stack(b, axis=0)
rand_p = torch.tensor(rand_p, device=device, dtype=torch.float64)  # 1,10,1080,1080

recovery_slm_field = random_phase_recovery(intensity.unsqueeze(0), rand_p, d0, dx, lambda_, 100, 'FFT', device=device)
final_slm_field = second_iterate(obj, recovery_slm_field, intensity, rand_p, 50, 'FFT', device=device)
est_abe_pha = get_phase(recovery_slm_field / torch.fft.fftshift(torch.fft.fft2(obj)))
plt.imshow(est_abe_pha[0].cpu(), cmap='gray')
plt.show()
sensor = get_amplitude(torch.fft.ifft2(recovery_slm_field * torch.exp(-1j * est_abe_pha)))
plt.imshow(sensor[0].cpu(), cmap='gray')
plt.show()

new_p = get_0_2pi(abe-est_abe_pha[0])
plt.imshow(new_p.cpu(), cmap='gray')
plt.show()
# imageio.imsave('final_slm.png', phasemap_8bit(new_p))
