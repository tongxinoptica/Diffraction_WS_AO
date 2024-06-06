import scipy
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

from unit import to_mseloss, to_ssim, to_pearson, to_psnr
import torch.nn.functional as F


def Diffraction_propagation(field, distance, dx, wavelength, transfer_fun='Angular Spectrum', device='cpu'):
    H = get_transfer_fun(
        field.shape[-2],
        field.shape[-1],
        dx=dx,
        wavelength=wavelength,
        distance=distance,
        transfer_fun=transfer_fun,
        device=device)
    U1 = torch.fft.fftshift(torch.fft.fftn(field, dim=(-2, -1), norm='ortho'), (-2, -1))
    U2 = U1 * H
    result = torch.fft.ifftn(torch.fft.ifftshift(U2, (-2, -1)), dim=(-2, -1), norm='ortho')
    return result


def get_transfer_fun(nu, nv, dx, wavelength, distance, transfer_fun, device):
    distance = torch.tensor([distance], dtype=torch.float64).to(device)
    fy = torch.linspace(-1 / (2 * dx), 1 / (2 * dx) - 1 / (nv * dx), nu, dtype=torch.float64).to(device)
    fx = torch.linspace(-1 / (2 * dx), 1 / (2 * dx) - 1 / (nv * dx), nv, dtype=torch.float64).to(device)
    FX, FY = torch.meshgrid(fx, fy, indexing='ij')
    FX = torch.transpose(FX, 0, 1)
    FY = torch.transpose(FY, 0, 1)
    k = 2 * torch.pi / wavelength
    if transfer_fun == 'Angular Spectrum':
        t = distance * k * torch.sqrt(1. - (wavelength * FX) ** 2 - (wavelength * FY) ** 2)
        # t = t.numpy()
        H = torch.exp(1j * t)
        # complex_data = H1['H']
        # complex_data_np = np.array(complex_data)
        # H1 = torch.tensor(complex_data_np)
        H_filter = (torch.abs(FX ** 2 + FY ** 2) <= (1 ** 2) * torch.abs(FX ** 2 + FY ** 2).max()).type(
            torch.FloatTensor).to(device)
        # df = pd.DataFrame(H.data.numpy())
        # df.to_csv('H.csv', index=False)
        return H * H_filter
    if transfer_fun == 'Fresnel':
        k = 2 * np.pi * (1 / wavelength)
        H = torch.exp(1j * k * distance * (1 - 0.5 * ((FX * wavelength) ** 2 + (FY * wavelength) ** 2)))
        H = H.to(device)
        return H


def get_amplitude(field):
    Amplitude = torch.abs(field)
    return Amplitude


def get_phase(field):  # --> 0-2pi
    Phase = torch.angle(field)
    Phase = (Phase + 2 * torch.pi) % (2 * torch.pi)
    return Phase


def get_hologram(amplitude, phase):
    hologram = amplitude * torch.cos(phase) + 1j * amplitude * torch.sin(phase)
    return hologram


def get_0_2pi(phase):
    return (phase + 2 * torch.pi) % (2 * torch.pi)


def ONN_Propagation(img, dis_first, dis_onn, dis_after, dx, wavelength, p1, p2, p3, p4):
    output1 = Diffraction_propagation(img, dis_first, dx, wavelength, transfer_fun='Angular Spectrum')
    am1 = get_amplitude(output1)
    ph1 = get_phase(output1) + p1
    ph1 = get_0_2pi(ph1)
    hologram1 = get_hologram(am1, ph1)

    # Propagation 2
    output2 = Diffraction_propagation(hologram1, dis_onn, dx, wavelength, transfer_fun='Angular Spectrum')
    am2 = get_amplitude(output2)
    ph2 = get_phase(output2) + p2
    ph2 = get_0_2pi(ph2)
    hologram2 = get_hologram(am2, ph2)

    # Propagation 3
    output3 = Diffraction_propagation(hologram2, dis_onn, dx, wavelength, transfer_fun='Angular Spectrum')
    am3 = get_amplitude(output3)
    ph3 = get_phase(output3) + p3
    ph3 = get_0_2pi(ph3)
    hologram3 = get_hologram(am3, ph3)

    # Propagation 4
    output4 = Diffraction_propagation(hologram3, dis_onn, dx, wavelength, transfer_fun='Angular Spectrum')
    am4 = get_amplitude(output4)
    ph4 = get_phase(output2) + p4
    ph4 = get_0_2pi(ph4)
    hologram4 = get_hologram(am4, ph4)

    # Propagation last
    output5 = Diffraction_propagation(hologram3, dis_after, dx, wavelength, transfer_fun='Angular Spectrum')
    return output5


def SLM_Propagation(img, dis, dx, wavelength, p):
    img = get_hologram(img, p)
    output = Diffraction_propagation(img, dis, dx, wavelength)
    return output


def lens_phase(X, Y, k, f):  # X and Y is space coordinate
    len_p = (k * (X ** 2 + Y ** 2) / (2 * f)) % (2 * torch.pi) + k * 1.4 * 0.003
    return len_p  # (0 , 2pi)


def random_phase_recovery(sensor_abe, random_phase, d0, dx, lambda_, iter_num, method, device='cpu'):
    if method == 'ASM':
        init_u = Diffraction_propagation(sensor_abe, -d0, dx, lambda_, device=device)  # Back prop
        init_u = torch.mean(init_u, dim=1)  # 1,1080,1920
        pbar = tqdm(range(iter_num))
        for i in pbar:
            sensor_p = Diffraction_propagation(init_u.unsqueeze(0) * torch.exp(1j * random_phase), d0, dx, lambda_,
                                               device=device)
            sensor_angle = get_phase(sensor_p)
            # new_sensor = sensor_abe * torch.exp(1j * sensor_angle)
            new_sensor = ((sensor_abe - get_amplitude(sensor_p)) * torch.rand(1, 4, 1, 1, device=device) + sensor_abe) * torch.exp(1j * sensor_angle)
            new_slm = Diffraction_propagation(new_sensor, -d0, dx, lambda_, device=device)  # Back prop
            init_u = torch.mean(new_slm * torch.exp(-1j * random_phase), dim=1)
        return init_u
    if method == 'FFT':
        init_u = torch.fft.ifftshift(torch.fft.ifft2(sensor_abe))
        init_u = torch.mean(init_u, dim=1)  # 1,1080,1920
        pbar = tqdm(range(iter_num))
        for i in pbar:
            sensor_p = torch.fft.fft2(torch.fft.fftshift(init_u.unsqueeze(0) * torch.exp(1j * random_phase)))

            sensor_angle = get_phase(sensor_p)
            # new_sensor = sensor_abe * torch.exp(1j * sensor_angle)
            new_sensor = ((sensor_abe - get_amplitude(sensor_p)) * torch.rand(1, 10, 1, 1, device=device) + sensor_abe) * torch.exp(1j * sensor_angle)
            new_slm = torch.fft.ifftshift(torch.fft.ifft2(new_sensor))  # Back prop
            init_u = torch.mean(new_slm * torch.exp(-1j * random_phase), dim=1)
        return init_u

def second_iterate(re_obj, init_u, sensor_abe, random_phase, iter_num, method, device='cpu'):
    if method == 'FFT':
        est_abe_pha = get_phase(init_u / torch.fft.fftshift(torch.fft.fft2(re_obj)))
        init_u = torch.fft.fftshift(torch.fft.fft2(re_obj)) * torch.exp(1j*est_abe_pha)
        pbar = tqdm(range(iter_num))
        for i in pbar:
            sensor_p = torch.fft.fft2(torch.fft.fftshift(init_u.unsqueeze(0) * torch.exp(1j * random_phase)))
            sensor_angle = get_phase(sensor_p)
            new_sensor = sensor_abe * torch.exp(1j * sensor_angle)
            # new_sensor = ((sensor_abe - get_amplitude(sensor_p)) * torch.rand(1, 3, 1, 1, device=device) + sensor_abe) * torch.exp(1j * sensor_angle)
            new_slm = torch.fft.ifftshift(torch.fft.ifft2(new_sensor))  # Back prop
            init_u = torch.mean(new_slm * torch.exp(-1j * random_phase), dim=1)
            pha = get_phase(init_u / torch.fft.fftshift(torch.fft.fft2(re_obj)))
            init_u = torch.fft.fftshift(torch.fft.fft2(re_obj)) * torch.exp(1j * pha)

        return init_u