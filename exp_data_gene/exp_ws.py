import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from Diffraction_H import get_0_2pi, get_amplitude, random_phase_recovery, get_phase, second_iterate, \
    Diffraction_propagation
from Zernike import generate_zer_poly
from unit import pad_array, phasemap_8bit, pad_tensor
import imageio
import cv2

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
# device = 'cpu'
dx = 8e-6  # m
d0 = 0.08
lambda_ = 532e-9


# obj = cv2.imread('obj2/obj1.png', cv2.IMREAD_GRAYSCALE) / 255
# obj = obj[:, 420:1500]
# obj = cv2.imread('tx/1.tif', cv2.IMREAD_GRAYSCALE) / 255
# obj = cv2.resize(obj, (1080, 1080), interpolation=cv2.INTER_CUBIC)
# obj = obj/np.max(obj)
# obj = pad_array(obj, 1080, 1920)
# obj = torch.tensor(obj, dtype=torch.float64, device=device)
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

# abe = cv2.imread('abe/1.png', cv2.IMREAD_GRAYSCALE) / 255
# abe = torch.tensor(abe[:, 420:1500], dtype=torch.float64, device=device)
# abe = abe * torch.pi * 2

def calculate_slm_input(n, dx, d0, lambda_, device):
    image_paths = [f'asm/intensity/{i}.bmp' for i in range(1, n)]
    intensity = []
    for path in image_paths:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) / 255.0
        img = img[310:2710, 765:3165]  # 1510,1965
        img = cv2.resize(img, (1035, 1035), interpolation=cv2.INTER_CUBIC)
        # img / img.max()
        # img = pad_array(img, 1080, 1920)
        # img = np.rot90(img, 2)  # Rotate 180
        img = np.flip(img, axis=0)
        # plt.imshow(img, cmap='gray')
        # plt.show()
        intensity.append(img)
    plt.imsave('asm/asm_p/gt_sensor.bmp', intensity[0], cmap='gray')
    intensity = np.stack(intensity, axis=0)
    intensity = torch.tensor(intensity, device=device, dtype=torch.float64).unsqueeze(0)  # 1,10,1080,1080

    image_paths = [f'asm/asm_p/ori/1_{i}.bmp' for i in range(1, n)]
    rand_p = []
    for path in image_paths:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) / 255.0
        # img = img[:, 420:1500]
        # img = cv2.resize(img, (1600, 1600), interpolation=cv2.INTER_NEAREST)
        img = pad_array(img, 1035, 1035, 0)
        rand_p.append(img * np.pi * 2)
    rand_p = np.stack(rand_p, axis=1)
    rand_p = torch.tensor(rand_p, device=device, dtype=torch.float64)  # 1,10,1080,1080

    recovery_slm_field = random_phase_recovery(intensity, rand_p, d0, dx, lambda_, 1000, 'ASM', device=device)
    torch.save(recovery_slm_field, 'asm/asm_p/recovery_slm_field.pth')
    # final_slm_field = second_iterate(obj, recovery_slm_field, intensity, rand_p, 50, 'FFT', device=device)
    est_abe_pha = get_phase(recovery_slm_field)
    plt.imshow(est_abe_pha[0].cpu(), cmap='gray')
    plt.show()
    slm_intensity = get_amplitude(recovery_slm_field)
    plt.imshow(slm_intensity[0].cpu(), cmap='gray')
    plt.show()
    plt.imsave('asm/asm_p/slm_intensity.bmp', slm_intensity[0].cpu().numpy(), cmap='gray')
    plt.imsave('asm/asm_p/slm_phase.bmp', est_abe_pha[0].cpu().numpy(), cmap='gray')
    ori = Diffraction_propagation(recovery_slm_field * torch.exp(1j * rand_p[0][0]), d0, dx, lambda_, device=device)
    sensor_intensity = get_amplitude(ori[0])
    plt.imshow(sensor_intensity.cpu(), cmap='gray')
    plt.show()
    plt.imsave('asm/asm_p/est_sensor.bmp', sensor_intensity.cpu().numpy(), cmap='gray')
    return recovery_slm_field


recovery_slm_field = calculate_slm_input(n=33, dx=dx, d0=d0, lambda_=lambda_, device=device)
# recovery_slm_field = torch.load('asm/asm_p/recovery_slm_field.pth').to(device)

# Calculate hologram
phase = torch.zeros(1, 1, 1035, 1035).to(device)
phase.requires_grad = True
phase = torch.nn.Parameter(phase)
optimizer = torch.optim.Adam([phase], lr=0.01)
initial_lr = optimizer.param_groups[0]['lr']
loss_fn = torch.nn.MSELoss()
best_loss = 10.0
gt_sensor = cv2.imread('asm/usaf_square.png', cv2.IMREAD_GRAYSCALE) / 255
gt_sensor = cv2.resize(gt_sensor, (690, 690), interpolation=cv2.INTER_CUBIC)
gt_sensor = pad_array(gt_sensor, 1035, 1035, 0)
gt_sensor = torch.tensor(gt_sensor, device=device, dtype=torch.float64)
pbar = tqdm(range(5000))
for i in pbar:
    optimizer.zero_grad()
    # phase_ex = F.interpolate(phase, size=(1600, 1600), mode='nearest')
    # phase_ex = pad_tensor(phase_ex, 2000, 2000, 0)
    slm_phase = get_phase(recovery_slm_field)
    slm_field = torch.exp(1j * phase) * torch.exp(1j*slm_phase)
    sensor = Diffraction_propagation(slm_field, d0, dx, lambda_, transfer_fun='Angular Spectrum', device=device)
    sensor_amp = get_amplitude(sensor)
    # with torch.no_grad():
    #     s = (sensor_amp * gt_sensor).mean() / \
    #         (sensor_amp ** 2).mean()
    loss_val = loss_fn(1 * sensor_amp, gt_sensor)
    loss_val.backward()
    optimizer.step()
    current_lr = optimizer.param_groups[0]['lr']
    if i % 500 == 0:
        optimizer.param_groups[0]['lr'] = current_lr * 1
    with torch.no_grad():
        if loss_val < best_loss:
            best_phase = phase
            best_loss = loss_val.item()
            best_amp = 1 * sensor_amp
    pbar.set_postfix(loss=f'{loss_val:.6f}', lr=current_lr, refresh=True)
plt.imshow(best_amp[0][0].cpu().data.numpy(), cmap='gray')
plt.show()
plt.imshow(best_phase[0][0].cpu().data.numpy(), cmap='gray')
plt.show()
# best_phase = pad_tensor(best_phase, 1200, 1920, 0)
output_phase = get_0_2pi(best_phase)/(2*torch.pi)
output_phase = torch.flip(output_phase, [3])
plt.imsave('asm/asm_p/usaf_p1.png', output_phase[0][0].cpu().data.numpy(), cmap='gray')
# imageio.imsave('asm/asm_p/load_slm_p.png', phasemap_8bit(best_phase, inverted=False))
# nodifus = get_phase(slm_field)
# imageio.imsave('asm/asm_p/nodfus_slm.png', phasemap_8bit(nodifus, inverted=False))

# load_slm_p = cv2.imread('asm/asm_p/load_slm_p.png', cv2.IMREAD_GRAYSCALE) / 255
# load_slm_p = torch.tensor(load_slm_p*2*torch.pi, dtype=torch.float64, device=device)
# field = Diffraction_propagation(torch.exp(1j*load_slm_p), d0, dx, lambda_, transfer_fun='Angular Spectrum', device=device)
# inten = get_amplitude(field)
# plt.imshow(inten.cpu().data.numpy(), cmap='gray')
# plt.show()