import cv2
import torch

from Diffraction_H import get_0_2pi, Diffraction_propagation, get_amplitude
from unit import sobel_grad


def train_data(batch, zernike_stack, holo_path, d0, dx, lambda_, device):
    zer_num = zernike_stack.shape[0]
    coeff = torch.rand(batch, zer_num, 1, 1, 1, dtype=torch.float64, device=device)
    zernike_stack = zernike_stack.unsqueeze(0)
    zer_pha = get_0_2pi((coeff * zernike_stack).sum(dim=1))  # size=(batch,1,1000,1000)
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
    delta_intensity = abe - ref
    delta_gx = abe_gx - ref_gx
    delta_gy = abe_gy - ref_gy  # size=(batch,1,1000,1000)
    input = torch.cat((delta_intensity, delta_gx, delta_gy), dim=1)
    return input, coeff, obj_field, ref, abe
