import cv2
import torch.nn.functional as F
import torch
from PIL import Image
from matplotlib import pyplot as plt
from torch import nn
from tqdm import tqdm
from Diffraction_H import get_0_2pi, get_amplitude, random_phase_recovery, second_iterate
# from unet_model import Unet
from model import RDR_model
import numpy as np
from torch.utils.data import DataLoader
from train_dataloader import train_data
from unit import creat_obj, sobel_grad

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
lambda_ = 532e-9  # mm
pi = torch.tensor(np.pi, dtype=torch.float64)
k = (2 * pi / lambda_)
dx = 8e-6  # m
size = (768, 768)
n_max = 15
zer_radius = 400
train_batch = 4
total_epoch = 1500
learning_rate = 0.001
img_dir = './data/img'
gt_dir = './data/gt'
zer_path = '../parameter/zernike_stack_{}_{}.pth'.format(n_max, zer_radius)
zernike_stack = torch.load(zer_path)  # zur_num,1,1000,1000
zer_num = zernike_stack.shape[0]

phase = torch.rand(3, 100, 100, dtype=torch.float64, device=device)
phase = F.interpolate(phase.unsqueeze(0), size=size, mode='bicubic', align_corners=False)

noise_level = 0.3  # Adjustable parameter for noise level
noise = torch.randn(size, dtype=torch.float64, device=device) * noise_level

model = RDR_model()
model.to(device)
train_loader = DataLoader(train_data(img_dir, gt_dir), batch_size=train_batch, shuffle=False)
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
for i in range(total_epoch):
    current_lr = optimizer.param_groups[0]['lr']
    if i % 50 == 0:
        optimizer.param_groups[0]['lr'] = current_lr * 0.8
        print('lr update: {}'.format(optimizer.param_groups[0]['lr']))

    model.train()
    train_loader = tqdm(train_loader)
    for batch_id, train_data in enumerate(train_loader):
        img, gt = train_data
        img = F.interpolate(img.to(device), size=(512, 512), mode='bicubic', align_corners=False)
        gt = F.interpolate(gt.to(device), size=(512, 512), mode='bicubic', align_corners=False)
        # label = label.to(device)
        # zernike_pha = get_0_2pi(
        #     (torch.rand(zer_num, 1, 1, 1, dtype=torch.float64, device='cpu') * 2 * zernike_stack).sum(dim=0))
        # slm_plane_fft = torch.fft.fftshift(torch.fft.fft2(img)) * torch.exp(1j * zernike_pha)
        # sensor_plane_fft = get_amplitude(torch.fft.ifft2(torch.fft.ifftshift(slm_plane_fft))).float()
        # sensor_plane_fft = F.interpolate(sensor_plane_fft, size=(512, 512), mode='bicubic', align_corners=False)
        # sensor_plane_fft = sensor_plane_fft / torch.max(sensor_plane_fft)
        # plt.imshow(sensor_plane_fft[0][0], cmap='gray')
        # plt.show()
        # plt.imshow(img[0][0], cmap='gray')
        # plt.show()
        # img = F.interpolate(img, size=(512, 512), mode='bicubic', align_corners=False)
        output = model(img.to(device))
        # out_gx, out_gy = sobel_grad(output, device)
        # img_gx, img_gy = sobel_grad(img, device)
        loss = loss_function(output, gt)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loader.desc = "[train epoch {}] loss: {:.6f}".format(i + 1, loss)
    if i % 100 == 0:
        torch.save(model.state_dict(), './unet+/n15_i2_{}.pth'.format(i))

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
