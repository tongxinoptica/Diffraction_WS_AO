import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage import draw
from tqdm import tqdm

from Diffraction_H import lens_phase, Diffraction_propagation, get_phase, get_amplitude, get_hologram, get_0_2pi
from Zernike import zernike_phase
from unit import creat_obj, phasemap_8bit
import imageio

lambda_ = 532e-9  #
k = 2 * np.pi / lambda_
f1 = 0.1  # focal length
f2 = 0.1
f3 = 0.1
f4 = 0.15
d0 = 1 * f1  # dis between obj and lens1
d1 = 1 * f1  # dis between lens1 and abe
d2 = 1 * f2  # dis between abe and lens2
d3 = f2 + f3  # dis between lens2 and observe

radius = 400
size = 1000  #
dx = 8e-6  #
sample_path = 'test_img/grid_10.png'
if_obj = False
Add_zernike = False
slm = False
phase_correct = False
SGD_correct = False
# if phase_correct or SGD_correct:  # dis between lens2 and observe/correction
#     d3 = 0.5 * f
#     d4 = 0.5 * f
# else:
#     d3 = 1 * f

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

# Create object and binary
obj = creat_obj(sample_path, size, radius=radius, binaty_inv=0, if_obj=if_obj)  # 0: Inverse; 1: NoInverse; 2: None
# plt.imshow(obj.data.numpy(), cmap='gray')
# plt.show()


x = torch.linspace(-size / 2, size / 2, size, dtype=torch.float64) * dx
y = torch.linspace(size / 2, -size / 2, size, dtype=torch.float64) * dx
X, Y = torch.meshgrid(x, y, indexing='xy')

# plt.imshow(zer.data.cpu().numpy(), cmap='gray')
# plt.show()
# imageio.imwrite('./test_img/zer_1.png', phasemap_8bit(zer))
# plt.imsave('./test_img/zer_1.png', zer.data.numpy(), cmap='gray')

# Propagate to lens1
obj_field = obj.unsqueeze(0).unsqueeze(0).to(device)
free_d0 = Diffraction_propagation(obj_field, d0, dx, lambda_).to(device)
free_d0_amp = get_amplitude(free_d0)
free_d0_ph = get_phase(free_d0)
len1_phs = lens_phase(X, Y, k, f1).to(device)  # lens1 phase
new_ph = get_0_2pi(free_d0_ph - len1_phs)
free_d1_field = get_hologram(free_d0_amp, new_ph)

# Propagate lens1 to abe
free_d1 = Diffraction_propagation(free_d1_field, d1, dx, lambda_)
free_d1_amp = get_amplitude(free_d1)
free_d1_ph = get_phase(free_d1)
# plt.imshow(free_d1_amp.squeeze(0).squeeze(0).cpu().data.numpy(), cmap='gray')
# plt.show()


if Add_zernike:
    # Propagate abe to lens2
    zer = zernike_phase(size, dx, n_max=15, radius=radius, intensity=9).unsqueeze(0)  # 0-2pi
    plt.imshow(zer.squeeze(0).squeeze(0).cpu().data.numpy(), cmap='gray')
    plt.title('Aberration')
    plt.show()
    imageio.imwrite('./test_img/zer.png', phasemap_8bit(zer))
    zer_phase = get_0_2pi(zer + free_d1_ph)

    zer_field = free_d1_amp * torch.exp(1j * zer_phase)
    free_d2 = Diffraction_propagation(zer_field, 1*d2, dx, lambda_)
    free_d2_amp = get_amplitude(free_d2)
    free_d2_ph = get_phase(free_d2)
    # plt.imshow(free_d0_amp.squeeze(0).squeeze(0), cmap='gray')
    # plt.show()
else:
    # propagate to lens2
    free_d2 = Diffraction_propagation(free_d1, 1*d2, dx, lambda_)
    free_d2_amp = get_amplitude(free_d2)
    free_d2_ph = get_phase(free_d2)

# Propagate lens2 to lens3
len2_phs = lens_phase(X, Y, k, f2).to(device)  # lens2 phase
new_ph = get_0_2pi(free_d2_ph - len2_phs)
free_d2_field = get_hologram(free_d2_amp, new_ph)
free_d3 = Diffraction_propagation(free_d2_field, 0.1, dx, lambda_)
free_d3_amp = get_amplitude(free_d3)
free_d3_ph = get_phase(free_d3)  # 0-2pi
free_d3_amp = free_d3_amp / torch.max(free_d3_amp)

plt.imshow(free_d3_amp.squeeze(0).squeeze(0).cpu().data.numpy(), cmap='gray')
plt.show()
plt.imshow(free_d3_ph.squeeze(0).squeeze(0).cpu().data.numpy(), cmap='gray')
plt.show()
# free_d3 = Diffraction_propagation(free_d3, f, dx, lambda_)
# free_d3_amp = get_amplitude(free_d3)
# free_d3_ph = get_phase(free_d3)  # 0-2pi

len3_phs = lens_phase(X, Y, k, f3).to(device)  # lens3 phase
new_ph = get_0_2pi(free_d3_ph - len3_phs)
free_d3_field = get_hologram(free_d3_amp, new_ph)
if slm:
    # Propagate lens3 to slm
    d4 = 1 * f3
    d5 = 1 * f4
    free_d4 = Diffraction_propagation(free_d3_field, d4, dx, lambda_)
    free_d4_amp = get_amplitude(free_d4)
    free_d4_ph = get_phase(free_d4)  # 0-2pi
    # plt.imshow(free_d4_amp.squeeze(0).squeeze(0).cpu().data.numpy(), cmap='gray')
    # plt.show()
    # plt.imshow(free_d4_ph.squeeze(0).squeeze(0).cpu().data.numpy(), cmap='gray')
    # plt.show()

    slm_cor = torch.rot90(zer, 2, [2, 3])
    cor_pha = get_0_2pi(free_d4_ph - slm_cor)
    free_slm = get_hologram(free_d4_amp, cor_pha)

    free_d5 = Diffraction_propagation(free_slm, d5, dx, lambda_)
    free_d5_amp = get_amplitude(free_d5)
    free_d5_ph = get_phase(free_d5)  # 0-2pi
    len4_phs = lens_phase(X, Y, k, f4).to(device)  # lens4 phase
    new_ph = get_0_2pi(free_d5_ph - len4_phs)
    free_d5_field = get_hologram(free_d5_amp, new_ph)
    free_d6 = Diffraction_propagation(free_d5_field, d5, dx, lambda_)
    free_d6_amp = get_amplitude(free_d6)
    free_d6_ph = get_phase(free_d6)  # 0-2pi
    plt.imshow(free_d6_amp.squeeze(0).squeeze(0).cpu().data.numpy(), cmap='gray')
    plt.title('Image Correction')
    plt.show()
    # plt.imsave('./test_img/castle_corr.png', free_d6_amp.squeeze(0).squeeze(0).cpu().data.numpy(), cmap='gray')

#  Image No Correction
d4 = f3 + f4
d5 = 1 * f4
free_d4 = Diffraction_propagation(free_d3_field, d4, dx, lambda_)
free_d4_amp = get_amplitude(free_d4)
free_d4_ph = get_phase(free_d4)  # 0-2pi

len4_phs = lens_phase(X, Y, k, f4).to(device)  # lens4 phase
new_ph = get_0_2pi(free_d4_ph - len4_phs)
free_d4_field = get_hologram(free_d4_amp, new_ph)
free_d5 = Diffraction_propagation(free_d4_field, d5, dx, lambda_)
free_d5_amp = get_amplitude(free_d5)
free_d5_ph = get_phase(free_d5)  # 0-2pi
plt.imshow(free_d5_amp.squeeze(0).squeeze(0).cpu().data.numpy(), cmap='gray')
plt.title('Image No Correction')
plt.show()
# plt.imsave('./test_img/castle_zer.png', free_d5_amp.squeeze(0).squeeze(0).cpu().data.numpy(), cmap='gray')

# plt.imshow(free_d5_ph.squeeze(0).squeeze(0).cpu().data.numpy(), cmap='gray')
# plt.show()
# plt.imsave('./test_img/g4_f_zer_amp.png', free_d3_amp.squeeze(0).squeeze(0).cpu().data.numpy(),cmap='gray')
#  Add slm phase to correct abe


# imageio.imwrite('./test_img/g8_0.5f_nozer_ph.png', phasemap_8bit(free_d3_ph, False))
# plt.imsave('./test_img/obj_0.5f_zer_ph.png', free_d3_ph.squeeze(0).squeeze(0).numpy(), cmap='gray')

if phase_correct:
    ph_zer = cv2.imread('test_img/g4_0.5f_zer_ph.png', cv2.IMREAD_GRAYSCALE)
    ph_nozer = cv2.imread('test_img/g4_0.5f_nozer_ph.png', cv2.IMREAD_GRAYSCALE)
    ph_zer = (torch.tensor(ph_zer) / 255) * 2 * torch.pi
    ph_nozer = (torch.tensor(ph_nozer) / 255) * 2 * torch.pi
    del_phase = get_0_2pi(ph_nozer - ph_zer).unsqueeze(0).unsqueeze(0)  # 0-2pi
    plt.imshow(del_phase.squeeze(0).squeeze(0).data.numpy(), cmap='gray')
    plt.show()
    corr_ph = get_0_2pi(free_d3_ph + ph_nozer - ph_zer)
    free_d4_field = free_d3_amp * torch.exp(1j * corr_ph)
    free_d4 = Diffraction_propagation(free_d4_field, d4, dx, lambda_)
    free_d4_amp = get_amplitude(free_d4)
    free_d4_ph = get_phase(free_d4)  # 0-2pi
    plt.imshow(free_d4_amp.squeeze(0).squeeze(0).data.numpy(), cmap='gray')
    plt.show()

    # plt.imsave('./test_img/g4_f_com_amp.png', free_d4_amp.squeeze(0).squeeze(0).data.numpy(),cmap='gray')

if SGD_correct:
    d4 = 0.5 * f4
    lr = 0.01
    num_iters = 15000
    init_phase = free_d3_ph.requires_grad_(True)
    optvars = [{'params': init_phase}]
    optimizer = torch.optim.Adam(optvars, lr=lr)
    loss_fn = torch.nn.MSELoss()
    loss_val = []
    best_loss = 10.
    # Generate target_amp
    target = cv2.imread('test_img/obj_zer/g10_f_nozer_amp.png', cv2.IMREAD_GRAYSCALE)
    target_amp = (torch.tensor(target) / 255).unsqueeze(0).unsqueeze(0).to(device)
    plt.imshow(target_amp.squeeze(0).squeeze(0).cpu().data.numpy(), cmap='gray')
    plt.axis('off')  # 关闭坐标轴
    plt.title('Target')
    plt.show()

    pbar = tqdm(range(num_iters))
    for i in pbar:
        optimizer.zero_grad()
        slm_field = free_d3_amp * torch.exp(1j * init_phase).to(device)
        ob_field = Diffraction_propagation(slm_field, d4, dx, lambda_)
        ob_amp = get_amplitude(ob_field)
        with torch.no_grad():
            s = (ob_amp * target_amp).mean() / \
                (ob_amp ** 2).mean()
        loss_val = loss_fn(s * ob_amp, target_amp)
        loss_val.backward()
        optimizer.step()

        if i == 0:
            plt.imshow(ob_amp.squeeze(0).squeeze(0).cpu().data.numpy(), cmap='gray')
            plt.axis('off')  # 关闭坐标轴
            plt.title('Aberration')
            plt.show()

        with torch.no_grad():
            if loss_val < best_loss:
                best_phase = init_phase
                best_loss = loss_val.item()
                best_amp = s * ob_amp
                best_iter = i + 1
        pbar.set_postfix(loss=f'{loss_val:.6f}', refresh=True)
    print(f' -- optimization is done, best loss: {best_loss}')

    plt.imshow(best_amp.squeeze(0).squeeze(0).cpu().data.numpy(), cmap='gray')
    plt.axis('off')
    plt.title('Correction')
    plt.show()

    imageio.imwrite('./test_img/g10_correct_phase.png', phasemap_8bit(best_phase, False))
    plt.imsave('./test_img/g10_correct_amp.png', best_amp.squeeze(0).squeeze(0).cpu().data.numpy(), cmap='gray')
