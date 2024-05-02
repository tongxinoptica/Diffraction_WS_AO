import os

import numpy as np
import torch
from matplotlib import pyplot as plt

from Diffraction_H import get_0_2pi
from Zernike import generate_zer_poly

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
lambda_ = 532e-9  # mm
pi = torch.tensor(np.pi, dtype=torch.float64)
k = (2 * pi / lambda_)
dx = 8e-6  # m
d0 = 0.10  # m
size = 1000
mask_size = (size, size)
Zer_radius = 400
pupil_radium = 400
n_max = 15
w = 3e-3
zer_path = 'parameter/zernike_stack_{}_{}.pth'.format(n_max, Zer_radius)
if os.path.exists(zer_path):
    zernike_stack = torch.load(zer_path).to(device)  # zur_num,1,1000,1000
    zer_num = zernike_stack.shape[0]
else:
    print('Generate Zernike polynomial')
    zernike_stack, zer_num = generate_zer_poly(size=size, dx=dx, n_max=n_max, radius=Zer_radius)

coeff = torch.rand(zer_num, 1, 1, 1, device=device, dtype=torch.float64)
zer_pha = get_0_2pi((coeff * zernike_stack).sum(dim=0)).squeeze(0)
plt.imshow(zer_pha.cpu(), cmap='gray')
plt.show()
coeff = coeff.expand(-1, -1, 1000, 1000)
zer_pha = get_0_2pi((coeff * zernike_stack).sum(dim=0)).squeeze(0)
plt.imshow(zer_pha.cpu(), cmap='gray')
plt.show()
