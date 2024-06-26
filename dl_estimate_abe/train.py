import os
from torch import nn
import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from Diffraction_H import get_0_2pi, Diffraction_propagation, get_amplitude
from Zernike import generate_zer_poly
from resnet50 import ResNet50
from generate_data import train_data

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
lambda_ = 532e-9  # mm
pi = torch.tensor(np.pi, dtype=torch.float64)
k = (2 * pi / lambda_)
dx = 8e-6  # m
d0 = 0.10  # m
size = 1000
mask_size = (size, size)
zer_radius = 500
pupil_radium = 400
n_max = 15
w = 3e-3
learning_rate = 0.04
batch = 4
epoch = 250000
cut = True
zer_path = '../parameter/zernike_stack_{}_{}.pth'.format(n_max, zer_radius)
holo_path = 'gray_grid/test_10cm.png'
writer = SummaryWriter('runs/gray')

# Define zernike aberration
if os.path.exists(zer_path):
    zernike_stack = torch.load(zer_path).to(device)  # zur_num,1,1000,1000
    zer_num = zernike_stack.shape[0]
    if cut:
        x = torch.linspace(-size / 2, size / 2, size, dtype=torch.float64) * dx
        y = torch.linspace(size / 2, -size / 2, size, dtype=torch.float64) * dx
        X, Y = torch.meshgrid(x, y, indexing='xy')
        rho = torch.sqrt(X ** 2 + (Y ** 2))
        mask = rho <= dx * 400
        mask = mask.unsqueeze(0).unsqueeze(0)
        zernike_stack = zernike_stack * mask
    print('zer_num = {}, zer_radium = {}'.format(zer_num, zer_radius))
else:
    print('Generate Zernike polynomial')
    zernike_stack, zer_num = generate_zer_poly(size=size, dx=dx, n_max=n_max, radius=zer_radius, device=device)

#  Load resnet50
loss_mse = nn.MSELoss()
loss_l1 = nn.L1Loss()
model = ResNet50(num_classes=zer_num).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
load_weght = False
if load_weght:
    model.load_state_dict(torch.load('gray_grid/u50_20000.pth'))
    print('Weight Loaded')
model.train()

pbar_train = tqdm(range(1, epoch))
for i in pbar_train:
    input, coeff, obj_field, ref, _ = train_data(batch, zernike_stack, holo_path, d0, dx, lambda_, device)
    output = model(input.float().to(device))
    out_coeff = output.unsqueeze(4)  # size=(batch,zer_num,1,1,1)
    est_zer_phase = get_0_2pi((out_coeff * zernike_stack).sum(dim=1))
    out_ref_com = Diffraction_propagation(obj_field * torch.exp(-1j * est_zer_phase), d0, dx, lambda_, device=device)
    out_ref = get_amplitude(out_ref_com)
    loss_ref = loss_mse(out_ref.to(torch.float32), ref.to(torch.float32))
    loss_coeff = loss_mse(out_coeff, coeff.to(torch.float32))
    loss = loss_coeff
    current_lr = optimizer.param_groups[0]['lr']
    if i % 10000 == 0:
        optimizer.param_groups[0]['lr'] = current_lr * 0.8
        print('lr update: {}'.format(optimizer.param_groups[0]['lr']))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    pbar_train.desc = "[train epoch {}] loss_coeff: {:.6f} loss_ref: {:.6f}".format(i + 1,
                                                                                    loss_coeff.item(), loss_ref.item())
    if i % 2000 == 0:
        writer.add_images('Input', out_ref, i)
        writer.add_scalar('Training Loss', loss.item(), i)
        torch.save(model.state_dict(), 'gray_grid/u50_{}.pth'.format(i))
writer.close()
# model.eval()
# pbar_eval = tqdm(range(100))
# for i in pbar_eval:
#     with torch.no_grad():
#         input, coeff, obj_field, ref = train_data(batch, zernike_stack, holo_path, d0, dx, lambda_, device)
#         output = model(input.float().to(device))
#         out_coeff = output.unsqueeze(4)  # size=(batch,zer_num,1,1,1)
#         est_zer_phase = get_0_2pi((out_coeff * zernike_stack).sum(dim=1))
#         out_ref_com = Diffraction_propagation(obj_field*torch.exp(-1j*est_zer_phase), d0, dx, lambda_, device=device)
#         out_ref = get_amplitude(out_ref_com)
#         loss = loss_mse(out_ref, ref) + loss_mse(out_coeff, coeff)
