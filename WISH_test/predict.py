import time

import cv2
import torch.nn.functional as F
import torch
from PIL import Image
from matplotlib import pyplot as plt
from torch import nn
from tqdm import tqdm
from Diffraction_H import get_0_2pi, get_amplitude, random_phase_recovery, second_iterate, get_phase
from unet_model import Unet
from model import RDR_model
import numpy as np
from torch.utils.data import DataLoader
from train_dataloader import train_data

from unit import creat_obj

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
lambda_ = 532e-9  # mm
pi = torch.tensor(np.pi, dtype=torch.float64)
k = (2 * pi / lambda_)
dx = 8e-6  # m
d0 = 0.03
size = (1000, 1000)
n_max = 15
zer_radius = 400
train_batch = 1

gt = './data/2.png'
test = './data/img/2.png'
img = cv2.imread(gt, cv2.IMREAD_GRAYSCALE) / 255
img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_CUBIC)
# abe = cv2.resize(abe,(512,512), interpolation=cv2.INTER_CUBIC)
img = torch.tensor(img, dtype=torch.float32, device=device)

test = cv2.imread(test, cv2.IMREAD_GRAYSCALE) / 255
test = torch.tensor(test, dtype=torch.float32, device=device)
# zer_path = '../parameter/zernike_stack_{}_{}.pth'.format(n_max, zer_radius)
# zernike_stack = torch.load(zer_path)  # zur_num,1,1000,1000
# zer_num = zernike_stack.shape[0]

phase = torch.rand(2, 50, 50, dtype=torch.float64, device=device)
phase = F.interpolate(phase.unsqueeze(0), size=(512, 512), mode='bicubic', align_corners=False)
phase = phase * torch.pi * 2
model = RDR_model()
model.to(device)
model.load_state_dict(torch.load('./data/bi_n15_i5_900.pth', map_location=device))
print('Weight Loaded')

# train_loader = DataLoader(train_data(img_dir, gt_dir), batch_size=train_batch, shuffle=False)
loss_function = nn.MSELoss()

model.eval()
with torch.no_grad():
    for i in range(1, 2):
        abe = './data/abe/{}.png'.format(i)
        abe = cv2.imread(abe, cv2.IMREAD_GRAYSCALE) / 255
        abe = torch.tensor(abe, dtype=torch.float32, device=device) * 2 * torch.pi
        slm_plane_fft = torch.fft.fftshift(torch.fft.fft2(img)) * torch.exp(1j * abe)
        sensor_plane_fft = torch.abs(torch.fft.ifft2(torch.fft.ifftshift(slm_plane_fft)))
        # plt.imshow(sensor_plane_fft.cpu(), cmap='gray')
        # plt.show()
        test = test / torch.max(test)

        slm_plane = slm_plane_fft * torch.exp(1j * phase)
        sensor_plane = torch.fft.fft2(torch.fft.fftshift(slm_plane))
        sensor_abe = get_amplitude(sensor_plane)
        sensor_abe = sensor_abe / sensor_abe.amax(dim=(2, 3), keepdim=True)
        noise_level = 0  # Adjustable parameter  for noise level
        noise = torch.randn(sensor_abe.shape, dtype=torch.float64, device=device) * noise_level
        sensor_abe_noisy = sensor_abe + noise
        time1 = time.time()
        recovery_slm_field = random_phase_recovery(sensor_abe_noisy, phase, d0, dx, lambda_, 20, 'FFT', device)

        imgcnn = model(test.unsqueeze(0).unsqueeze(0))

        # imgcnn = torch.clamp(imgcnn, min=0)
        # imgcnn = imgcnn / torch.max(imgcnn)
        # imgcnn = (imgcnn >= 0.5).float()
        final_slm_field = second_iterate(imgcnn.squeeze(0).squeeze(0), recovery_slm_field, sensor_abe_noisy, phase, 30,
                                         'FFT', device)
        est_abe_pha = get_phase(final_slm_field / torch.fft.fftshift(torch.fft.fft2(imgcnn[0][0])))
        time2 = time.time()
        sensor = get_amplitude(torch.fft.ifft2(final_slm_field * torch.exp(-1j * abe)))
        print(time2 - time1)
        print(loss_function(imgcnn, img))
        plt.imshow(imgcnn[0][0].cpu(), cmap='gray')
        plt.show()
        plt.imshow(est_abe_pha[0].cpu(), cmap='gray')
        plt.title('gs_cnn_gs')
        plt.show()
        # recovery_slm_field2= random_phase_recovery(sensor_abe_noisy, phase, d0, dx, lambda_, 1000, 'FFT', device)
        # est_abe_pha2 = get_phase(recovery_slm_field2 / torch.fft.fftshift(torch.fft.fft2(imgcnn[0][0])))
        # plt.imshow(est_abe_pha2[0].cpu(), cmap='gray')
        # plt.title('puregs_cnn')
        # plt.show()
# with torch.no_grad():
#     img = creat_obj('castle.png', size, radius=500, binaty_inv=2, if_obj=True,
#                     device=device)
#     zernike_pha = get_0_2pi(
#         (torch.rand(zer_num, 1, 1, 1, dtype=torch.float64, device=device) * 2 * zernike_stack).sum(dim=0))
#     zernike_pha = zernike_pha[0][116:884, 116:884]
#     slm_plane_fft = torch.fft.fftshift(torch.fft.fft2(img)) * torch.exp(1j * zernike_pha)
#     sensor_plane_fft = torch.fft.ifft2(torch.fft.ifftshift(slm_plane_fft))
#     ori_abe = get_amplitude(sensor_plane_fft)
#     output = model(ori_abe.float())
