import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from Diffraction_H import get_amplitude
from unit import pad_array, phasemap_8bit, rotate_array

obj = cv2.imread('asm/cabli500.tif', cv2.IMREAD_GRAYSCALE) / 255
rotated = rotate_array(obj, -1.5)
plt.imshow(obj, cmap='gray')
plt.show()
plt.imshow(rotated, cmap='gray')
plt.show()
# obj = pad_array(obj, 1080,1920)
# obj = obj[:,420:1500]
# obj = torch.tensor(obj[0], dtype=torch.float64)
# for i in range(1,11):
#     path = f'H:/exp_data_gene/flip_ro_rand/1_{i}.png'
#     img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) / 255
#     img = img[420:1500, :]
#     img = pad_array(img, 1080,1920)
    # img[520:560, 940:980] = 0
    # plt.imsave(f'H:/exp_data_gene/1_{i}.png', img[0], cmap='gray')
    # slmp = torch.tensor(img, dtype=torch.float64) * 2 * torch.pi
    # slm_plane = torch.fft.fftshift(torch.fft.fft2(obj)) * torch.exp(1j * slmp)
    # sensor_plane = torch.fft.fft2(torch.fft.fftshift(slm_plane))
    # sensor_abe = get_amplitude(sensor_plane)
    # plt.imshow(sensor_abe.cpu(), cmap='gray')
    # plt.title('{}'.format(i))
    # plt.show()

# for j in range(1, 11):
#     phase = torch.rand(1, 100, 100, dtype=torch.float64)
#     phase = F.interpolate(phase.unsqueeze(0), size=(1080, 1080), mode='bicubic', align_corners=False)
#     phase = phase * torch.pi * 2
#     imageio.imsave('asm_p/1_{}.png'.format(j), phasemap_8bit(phase, inverted=False))
#
