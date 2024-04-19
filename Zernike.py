import imageio
import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.special import factorial
import os
from unit import phasemap_8bit
from tqdm import tqdm

def zernike_radial(n, m, rho):
    """
    Calculate the radial component of Zernike polynomial (R_n^m).
    """
    R = torch.zeros_like(rho)
    for k in range((n - m) // 2 + 1):
        R += ((-1) ** k * factorial(n - k) /
              (factorial(k) * factorial((n + m) // 2 - k) * factorial((n - m) // 2 - k))
              ) * rho ** (n - 2 * k)
    return R


def zernike(n, m, rho, theta):
    """
    Generate Zernike polynomial Z_n^m.
    """
    if m >= 0:
        zer = zernike_radial(n, m, rho) * torch.cos(m * theta)
    else:
        zer = zernike_radial(n, -m, rho) * torch.sin(-m * theta)
    zer = (torch.pi * zer / torch.max(zer) + torch.pi) % (2 * torch.pi)
    return zer


def generate_zer_poly(size, dx, n_max, radius):
    zer_path = 'parameter/zernike_stack_{}.pth'.format(n_max)
    if os.path.exists(zer_path):
        zernike_stack = torch.load(zer_path)
        zer_num = zernike_stack.shape[0]
        print('Zernike coefficient number: {}'.format(zer_num))
    else:
        x = torch.linspace(-size / 2, size / 2, size) * dx
        y = torch.linspace(size / 2, -size / 2, size) * dx
        X, Y = torch.meshgrid(x, y, indexing='xy')
        rho = torch.sqrt(X ** 2 + Y ** 2)
        theta = torch.arctan2(Y, X)
        mask = rho > dx * radius
        rho[mask] = 0.0
        rho = rho / torch.max(rho)
        zernike_list = []
        for n in range(1, n_max + 1):
            for m in range(-n, n + 1):
                if (n - abs(m)) % 2 == 0:
                    zer = zernike(n, m, rho, theta)
                    zer = (zer % (2 * torch.pi))  # 0-2pi
                    zer[mask] = 0
                    zernike_list.append(zer.unsqueeze(0))
        zer_num = len(zernike_list)
        print('Zernike coefficient number: {}'.format(zer_num))
        zernike_stack = torch.stack(zernike_list, dim=0)  # Stack along the new axis
        torch.save(zernike_stack, './parameter/zernike_stack_{}.pth'.format(n_max))

    return zernike_stack, zer_num


def zernike_phase(size, dx, n_max, radius, intensity):
    zernike_stack, zer_num = generate_zer_poly(size=size, dx=dx, n_max=n_max, radius=radius)
    random_coeffs = torch.rand(zer_num, 1, 1, 1, dtype=torch.float64) * intensity + 1
    zernike_phase = (random_coeffs * zernike_stack).sum(dim=0)
    zernike_phase = zernike_phase % (2 * torch.pi)
    return zernike_phase

#
radius = 500
size = 1000
dx = 8e-6
generate_zer_poly(size,dx,15,radius)
# zer_path = 'parameter/zernike_stack_15.pth'
# zernike_stack = torch.load(zer_path)
# zer_num = zernike_stack.shape[0]
# for i in range(1, 10):
#     zer_out_path = 'E:/Data/Zernike_phase/{}_{}'.format(i, i + 1)
#     os.makedirs(zer_out_path, exist_ok=True)
#     for j in tqdm(range(1, 2001)):
#         random_coeffs = torch.rand(zer_num, 1, 1, 1, dtype=torch.float64) * i + 1
#         zernike_phase = (random_coeffs * zernike_stack).sum(dim=0)
#         zernike_phase = zernike_phase % (2 * torch.pi)
#         imageio.imwrite(f'{zer_out_path}/{j}.png', phasemap_8bit(zernike_phase))
