import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from Diffraction_H import Diffraction_propagation, get_amplitude, get_phase, get_0_2pi, lens_phase, get_hologram
from Zernike import zernike_phase, zernike, generate_zer_poly
import torchvision.transforms as transforms
import torch.nn.functional as F
from unit import creat_obj
from skimage import draw

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
lambda_ = 532e-9  # mm
pi = torch.tensor(torch.pi, dtype=torch.float64)
k = (2 * pi / lambda_)
dx = 8e-6  # m
d0 = 0.1  # m
size = 1000
mask_size = (size, size)
Zer_radius = 400
pupil_radium = 400
n_max = 15
f1 = 0.1
f2 = 0.1

# Input pupil
in_pupil = torch.ones(mask_size, dtype=torch.float64)
in_pupil_mirror = torch.ones(mask_size, dtype=torch.float64)
# rr, cc = draw.disk((size / 2, size / 2), radius=pupil_radium)
# in_pupil[rr, cc] = 1.0

x = torch.linspace(-size / 2, size / 2, size, dtype=torch.float64) * dx
y = torch.linspace(size / 2, -size / 2, size, dtype=torch.float64) * dx
X, Y = torch.meshgrid(x, y, indexing='xy')

pupil_shift = False
if pupil_shift:
    x_shift = dx*100
    y_shift = 0
else:
    x_shift = 0
    y_shift = 0

rho = torch.sqrt((X-x_shift) ** 2 + ((Y-y_shift) ** 2))
mask = rho > dx * pupil_radium
in_pupil[mask] = 0.0
rho_mirror = torch.sqrt((X+x_shift) ** 2 + ((Y+y_shift) ** 2))
in_pupil_mirror[rho_mirror > dx * pupil_radium] = 0.0
Oblique = False
if Oblique:
    incident_angle = torch.tensor(1.0, dtype=torch.float64)
    phase_gradient = k * torch.sin(torch.deg2rad(incident_angle)) * Y
    in_pupil = in_pupil * torch.exp(1j * phase_gradient)

#  Start propagation
obj_field = in_pupil.unsqueeze(0).unsqueeze(0).to(device)
free_d0 = Diffraction_propagation(obj_field, f1, dx, lambda_).to(device)
free_d0_amp = get_amplitude(free_d0)
free_d0_ph = get_phase(free_d0)
# plt.imshow(obj_field.squeeze(0).squeeze(0).cpu().data.numpy(), cmap='gray')
# plt.show()
# plt.imshow(free_d0_ph.squeeze(0).squeeze(0).cpu().data.numpy(), cmap='gray')
# plt.show()
len1_phs = lens_phase(X, Y, k, f1).to(device)  # lens1 phase
new_ph = get_0_2pi(free_d0_ph - len1_phs)
free_d1_field = get_hologram(free_d0_amp, new_ph)

free_d1 = Diffraction_propagation(free_d1_field, f1+1*f2, dx, lambda_)
free_d1_amp = get_amplitude(free_d1)
free_d1_ph = get_phase(free_d1)
# plt.imshow(free_d1_amp.squeeze(0).squeeze(0).cpu().data.numpy(), cmap='gray')
# plt.show()

len2_phs = lens_phase(X, Y, k, f2).to(device)  # lens2 phase
new_ph = get_0_2pi(free_d1_ph - len2_phs)
free_d2_field = get_hologram(free_d1_amp, new_ph)
free_d2 = Diffraction_propagation(free_d2_field, f2, dx, lambda_)
free_d2_amp = get_amplitude(free_d2)
free_d2_ph = get_phase(free_d2)  # 0-2pi
Gaussian_blur = False
Gaussian_noise = False
noise_mask = mask.unsqueeze(0).unsqueeze(0).to(device)
if Gaussian_blur:
    gaussian_blur = transforms.GaussianBlur(45, sigma=[25, 25])
    free_d2_amp = gaussian_blur(free_d2_amp)

if Gaussian_noise:
    noise_std = 0.5
    noise = torch.randn(free_d2_amp.shape).to(device) * noise_std
    gaussian_blur = transforms.GaussianBlur(5, sigma=[1, 1])
    noise = gaussian_blur(noise)
    noise[noise_mask] = 0
    free_d2_amp = free_d2_amp + noise
    free_d2 = get_hologram(free_d2_amp, free_d2_ph)
# plt.imshow(free_d2_amp.squeeze(0).squeeze(0).cpu().data.numpy(), cmap='gray')
# plt.show()
# plt.imshow(free_d2_ph.squeeze(0).squeeze(0).cpu().data.numpy(), cmap='gray')
# plt.title('4f phase')
# plt.show()


#  Shift 4f
free_d1_ = Diffraction_propagation(free_d1_field, f1+1.1*f2, dx, lambda_)
free_d1_amp_ = get_amplitude(free_d1_)
free_d1_ph_ = get_phase(free_d1_)
new_ph = get_0_2pi(free_d1_ph_ - len2_phs)
free_d2_field_ = get_hologram(free_d1_amp_, new_ph)
free_d2_ = Diffraction_propagation(free_d2_field_, f2, dx, lambda_)
free_d2_amp_ = get_amplitude(free_d2_)
free_d2_ph_ = get_phase(free_d2_)  # 0-2pi
# plt.imshow(free_d2_amp_.squeeze(0).squeeze(0).cpu().data.numpy(), cmap='gray')
# plt.show()
# plt.imshow(free_d2_ph_.squeeze(0).squeeze(0).cpu().data.numpy(), cmap='gray')
# plt.title('shift 4f phase')
# plt.show()

#  GT correction phase
correct_gt_phase = get_0_2pi(free_d2_ph_ - free_d2_ph)
# plt.imshow((correct_gt_phase.squeeze(0).squeeze(0).cpu()*in_pupil_mirror).data.numpy(), cmap='gray')
# plt.title('correction gt phase')
# plt.show()
record_loss = False

# 4f rear focus plane adds mask
amp_type = 'grid10'  # rand_mask no_mask
if amp_type == 'grid10':
    amp = cv2.imread('test_img/grid_10.png', cv2.IMREAD_GRAYSCALE) / 255
elif amp_type == 'rand':
    amp = np.random.choice([0, 1], size=mask_size, p=[0.2, 0.8])
elif amp_type == 'no':
    amp = np.ones(mask_size)


# Create zernike aberration
zer_path = 'parameter/zernike_stack_{}_{}.pth'.format(n_max,Zer_radius)
if os.path.exists(zer_path):
    zernike_stack = torch.load(zer_path).to(device)  # zur_num,1,1000,1000
    zer_num = zernike_stack.shape[0]
else:
    print('Generate Zernike polynomial')
    zernike_stack, zer_num = generate_zer_poly(size=size, dx=dx, n_max=n_max, radius=Zer_radius)

coeff = torch.rand(zer_num, 1, 1, 1, device=device, dtype=torch.float64)
gt_pha = get_0_2pi((coeff * zernike_stack).sum(dim=0)).unsqueeze(0)
plt.imshow(gt_pha.squeeze(0).squeeze(0).cpu().data.numpy(), cmap='gray')
plt.title('Aberration')
plt.show()

a = torch.tensor(amp, dtype=torch.float64).unsqueeze(0).unsqueeze(0).to(device)
obj0 = torch.sqrt(a+1e-9)*free_d2
plt.imshow(get_amplitude(obj0).squeeze(0).squeeze(0).cpu().data.numpy(), cmap='gray')
plt.title('Pupil amp')
plt.show()
# plt.imshow(get_phase(obj0).squeeze(0).squeeze(0).cpu().data.numpy(), cmap='gray')
# plt.title('Pupil pha')
# plt.show()
free_d3 = Diffraction_propagation(obj0, d0, dx, lambda_)
free_d3_amp = get_amplitude(free_d3)
free_d3_amp[noise_mask] = 0
plt.imshow(free_d3_amp.squeeze(0).squeeze(0).cpu().data.numpy(), cmap='gray')
plt.title('4f No abe diffraction z = {}'.format(d0))
plt.show()
free_d4 = Diffraction_propagation(obj0*torch.exp(1j*gt_pha), d0, dx, lambda_)
free_d4_amp = get_amplitude(free_d4)
free_d4_amp[noise_mask] = 0
plt.imshow(free_d4_amp.squeeze(0).squeeze(0).cpu().data.numpy(), cmap='gray')
plt.title('4f abe diffraction z = {}'.format(d0))
plt.show()

# Start iteration
num_iters = 1000
params = torch.nn.Parameter(torch.zeros_like(coeff))
optimizer = torch.optim.Adam([params], lr=0.04)
initial_lr = optimizer.param_groups[0]['lr']
loss_fn = torch.nn.MSELoss()
best_loss = 10.0
amploss_path = "loss/loss_{}_mask_{}.txt".format(amp_type, int(d0))
if not os.path.isfile(amploss_path):
    print("File does not exist. Creating file.")
    mode = 'w+'
else:
    print("File exists. Appending to file.")
    mode = 'w'
with open(amploss_path, mode) as f:
    pbar = tqdm(range(num_iters))

    for i in pbar:
        optimizer.zero_grad()
        phase = get_0_2pi((params * zernike_stack).sum(dim=0)).unsqueeze(0).to(device)
        obj1 = obj0 * torch.exp(1j * (gt_pha - phase))  # Zernike
        # obj1 = torch.sqrt(a+1e-9) * free_d2_*torch.exp(-1j*phase)  # shift 4f
        free_d1 = Diffraction_propagation(obj1, d0, dx, lambda_)
        free_d1_amp = get_amplitude(free_d1)
        free_d1_amp[noise_mask] = 0
        with torch.no_grad():
            s = (free_d1_amp * free_d3_amp).mean() / \
                (free_d1_amp ** 2).mean()
        loss_val = loss_fn(s * free_d1_amp, free_d3_amp)
        # coeff_loss = torch.sum(torch.abs(params - coeff))
        loss_val.backward()
        optimizer.step()

        current_lr = optimizer.param_groups[0]['lr']
        if i % 400 == 0:
            optimizer.param_groups[0]['lr'] = current_lr * 0.8
        if record_loss:
            f.write(f"{loss_val.item()}\n")
            # f.write(f"{coeff_loss.item()}\n")
        with torch.no_grad():
            if loss_val < best_loss:
                best_para = params.clone().detach()
                best_phase = get_0_2pi((params * zernike_stack).sum(dim=0)).squeeze(0).squeeze(0)
                best_loss = loss_val.item()
                best_amp = s * free_d1_amp
        pbar.set_postfix(loss=f'{loss_val:.6f}', refresh=True)
    print(best_loss)
    # print(coeff_loss)
# free_d0_ph = free_d0_ph / torch.max(free_d0_ph)
plt.imshow(best_amp.squeeze(0).squeeze(0).cpu().data.numpy(), cmap='gray')
plt.show()
plt.imshow((best_phase.cpu()*in_pupil_mirror).data.numpy(), cmap='gray')
plt.show()


