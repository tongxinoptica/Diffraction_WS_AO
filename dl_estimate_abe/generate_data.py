import cv2
import matplotlib.pyplot as plt
import torch

from Diffraction_H import get_0_2pi, Diffraction_propagation, get_amplitude
from unit import sobel_grad


def train_data(batch, zernike_stack, holo_path, d0, dx, lambda_, device):
    size = 1000
    zer_num = zernike_stack.shape[0]
    coeff = torch.rand(batch, zer_num, 1, 1, 1, dtype=torch.float64, device=device)
    zernike_stack = zernike_stack.unsqueeze(0)
    zer_pha = get_0_2pi((coeff * zernike_stack).sum(dim=1))  # size=(batch,1,1000,1000)
    # plt.imshow(zer_pha[0].squeeze(0), cmap='gray')
    # plt.show()
    #  Load holo
    in_phase = cv2.imread(holo_path, cv2.IMREAD_GRAYSCALE) / 255
    in_phase = torch.tensor(in_phase[40:1040, 460:1460], dtype=torch.float64, device=device) * 2 * torch.pi
    in_phase = get_0_2pi(in_phase.unsqueeze(0).unsqueeze(0))
    obj_field = torch.exp(1j * in_phase) * torch.exp(1j * zer_pha)
    abe_complex = Diffraction_propagation(obj_field, d0, dx, lambda_, device=device)
    abe = get_amplitude(abe_complex)  # Capture image with abe
    ref_complex = Diffraction_propagation(torch.exp(1j * in_phase), d0, dx, lambda_, device=device)
    ref = get_amplitude(ref_complex)  # Capture image without abe
    abe_gx, abe_gy = sobel_grad(abe, device)
    ref_gx, ref_gy = sobel_grad(ref, device)
    delta_intensity = (abe - ref)**2
    abe_mag = torch.sqrt(abe_gx**2 + abe_gy**2)
    abe_ang = get_0_2pi(torch.atan2(abe_gy, abe_gx))
    ref_mag = torch.sqrt(ref_gx**2 + ref_gy**2)
    ref_ang = get_0_2pi(torch.atan2(ref_gy, ref_gx))
    delta_mag = torch.abs(abe_mag - ref_mag)
    delta_ang = torch.abs(abe_ang - ref_ang)  # size=(batch,1,1000,1000)
    input = torch.cat((delta_intensity, delta_mag, delta_ang), dim=1)
    return input, coeff, obj_field, ref, abe


def random_abe(pha_path, size, holo_path, d0, dx, lambda_, device):
    pha = center_crop(pha_path, size, size)
    pha = (torch.tensor(pha) / 255) * 2 * torch.pi
    zer_pha = get_0_2pi(pha)
    x = torch.linspace(-size / 2, size / 2, size, dtype=torch.float64) * dx
    y = torch.linspace(size / 2, -size / 2, size, dtype=torch.float64) * dx
    X, Y = torch.meshgrid(x, y, indexing='xy')
    rho = torch.sqrt(X ** 2 + (Y ** 2))
    mask = rho > dx * 400
    zer_pha[mask] = 0
    plt.imshow(zer_pha, cmap='gray')
    plt.show()
    in_phase = cv2.imread(holo_path, cv2.IMREAD_GRAYSCALE) / 255
    in_phase = torch.tensor(in_phase[40:1040, 460:1460], dtype=torch.float64, device=device) * 2 * torch.pi
    in_phase = get_0_2pi(in_phase.unsqueeze(0).unsqueeze(0))
    obj_field = torch.exp(1j * in_phase) * torch.exp(1j * zer_pha)
    abe_complex = Diffraction_propagation(obj_field, d0, dx, lambda_, device=device)
    abe = get_amplitude(abe_complex)  # Capture image with abe
    ref_complex = Diffraction_propagation(torch.exp(1j * in_phase), d0, dx, lambda_, device=device)
    ref = get_amplitude(ref_complex)  # Capture image without abe
    abe_gx, abe_gy = sobel_grad(abe, device)
    ref_gx, ref_gy = sobel_grad(ref, device)
    delta_intensity = abe - ref
    delta_gx = abe_gx - ref_gx
    delta_gy = abe_gy - ref_gy  # size=(batch,1,1000,1000)
    input = torch.cat((delta_intensity, delta_gx, delta_gy), dim=1)
    return input, obj_field, ref, abe


def center_crop(img_path, new_width, new_height):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    height, width = img.shape
    left = int((width - new_width) / 2)
    top = int((height - new_height) / 2)

    img_cropped = img[top:top + new_height, left:left + new_width]
    return img_cropped
