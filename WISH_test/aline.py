from unit import pad_array, pad_tensor
import numpy as np
import matplotlib.pyplot as plt
import torch
from Diffraction_H import Diffraction_propagation, get_phase

lambda_ = 532e-9  # mm
pi = torch.tensor(np.pi, dtype=torch.float64)
k = (2 * pi / lambda_)
dx = 8e-6  # m
d0 = 0.1

a = torch.ones(500, 500, dtype=torch.float64)
a = pad_tensor(a, 1000, 1000)
size = 1000
l = 1
x = torch.linspace(-size / 2, size / 2, size, dtype=torch.float64) * dx
y = torch.linspace(size / 2, -size / 2, size, dtype=torch.float64) * dx
X, Y = torch.meshgrid(x, y, indexing='xy')
phi = torch.arctan2(Y, X)
vortex_phase = torch.exp(1j * l * phi)

b = torch.fft.ifftshift(torch.fft.fft2(torch.fft.fftshift(a)))
angle = get_phase(b)
plt.imshow(angle[0], cmap='gray')
plt.show()
b1 = b * vortex_phase

c = Diffraction_propagation(b1, d0, dx, lambda_)
c = torch.abs(c)
plt.imshow(c[0], cmap='gray')
plt.show()
