import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from Diffraction_H import get_amplitude
from unit import pad_array
obj = cv2.imread('gt/1.png', cv2.IMREAD_GRAYSCALE) / 255
obj = pad_array(obj, 1080,1920)
# obj = obj[:,420:1500]
obj = torch.tensor(obj[0], dtype=torch.float64)
for i in range(1,11):
    path = f'slm_hole/1_{i}.png'
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) / 255
    # img[520:560, 940:980] = 0
    # plt.imsave(f'H:/exp_data_gene/flip2/1_{i}.png', img, cmap='gray')
    slmp = torch.tensor(img, dtype=torch.float64) * 2 * torch.pi
    slm_plane = torch.fft.fftshift(torch.fft.fft2(obj)) * torch.exp(1j * slmp)
    sensor_plane = torch.fft.fft2(torch.fft.fftshift(slm_plane))
    sensor_abe = get_amplitude(sensor_plane)
    plt.imshow(sensor_abe.cpu(), cmap='gray')
    plt.title('{}'.format(i))
    plt.show()

