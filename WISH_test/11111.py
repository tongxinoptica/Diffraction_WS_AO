import os
import shutil

import cv2
import imageio
import numpy as np
import torch
from matplotlib import pyplot as plt
from Diffraction_H import get_0_2pi, get_amplitude
from unit import phasemap_8bit, pad_array

n_max = 15
zer_radius = 500
zer_path = '../parameter/zernike_stack_{}_{}.pth'.format(n_max, zer_radius)

path = './data/img'
image_files = sorted([f for f in os.listdir(path) if f.endswith(('jpg', 'jpeg', 'png', 'bmp', 'tiff'))],
                     key=lambda x: int(os.path.splitext(x)[0]))
for image_file in image_files:
    image_path = os.path.join(path, image_file)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(image_path, image)


# count = 1
# n = 1
# for img in image_files:
#     img_path = os.path.join(path, img)
#     count = n
#     for i in range(99):
#         count += 10
#         new_img_name = f"{count}.png"
#         new_img_path = os.path.join('./data/gt', new_img_name)
#         shutil.copy(img_path, new_img_path)
#     n = n+1


def crop_center(img, cropx, cropy):
    y, x = img.shape
    startx = x // 2 - cropx // 2
    starty = y // 2 - cropy // 2
    return img[starty:starty + cropy, startx:startx + cropx]


zernike_stack = torch.load(zer_path)  # zur_num,1,1000,1000
zer_num = zernike_stack.shape[0]

# for i in range(1, 101):
#     zernike_pha = get_0_2pi(
#         (torch.rand(zer_num, 1, 1, 1, dtype=torch.float64, device='cpu') * 5 * zernike_stack).sum(dim=0))
#     imageio.imsave('./data/abe/{}.png'.format(i), phasemap_8bit(zernike_pha, inverted=False))
# n = 1
# for image_file in image_files:
#     if n % 10 == 1:
#         zernike_pha = get_0_2pi(
#             (torch.rand(zer_num, 1, 1, 1, dtype=torch.float64, device='cpu') * 5 * zernike_stack).sum(dim=0))
#         imageio.imsave('./data/abe/{}.png'.format(n), phasemap_8bit(zernike_pha, inverted=False))
#     image_path = os.path.join(path, image_file)
#     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) / 255
#     image = torch.tensor(image, dtype=torch.float64)
#     slm_plane_fft = torch.fft.fftshift(torch.fft.fft2(image)) * torch.exp(1j * zernike_pha)
#     sensor_plane_fft = torch.abs(torch.fft.ifft2(torch.fft.ifftshift(slm_plane_fft)))
#     plt.imsave('./data/img/{}.png'.format(n), sensor_plane_fft.squeeze(0).data.numpy(), cmap='gray')
#     n = n + 1
