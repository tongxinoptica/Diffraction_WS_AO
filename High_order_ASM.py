import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from scipy.fft import fftshift, ifftshift, fft2, ifft2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Target image
image = cv2.imread('exp_data_gene/asm/usaf_square.png', cv2.IMREAD_GRAYSCALE) / 255.0
image = torch.tensor(image, device=device)
H, W = image.shape

# Parameter
wavelength = 532e-9
k = 2 * np.pi / wavelength
pixel_size = 8e-6
x_size = W * pixel_size
y_size = H * pixel_size
alpha = 3
p = pixel_size
distance = 0.1

# 定义坐标网格
x = np.linspace(-x_size / 2, x_size / 2, W)
y = np.linspace(-y_size / 2, y_size / 2, H)
X, Y = np.meshgrid(x, y)
R = np.sqrt(X ** 2 + Y ** 2)

# 初始化相位图
phi = torch.rand((H, W), dtype=torch.float32, device=device, requires_grad=True)

# 角谱法中的傅里叶变换和逆傅里叶变换函数
def fourier_transform(E):
    return torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(E)))


def inverse_fourier_transform(E):
    return torch.fft.ifftshift(torch.fft.ifft2(torch.fft.fftshift(E)))


# 计算调制函数 U(fx, fy; phi)
def modulation_function(phi, alpha, p, fx, fy):
    U = torch.zeros_like(fx, dtype=torch.complex64)
    for i in range(-int((alpha - 1) / 2), int((alpha - 1) / 2) + 1):
        for j in range(-int((alpha - 1) / 2), int((alpha - 1) / 2) + 1):
            shifted_phi = torch.exp(1j * phi)
            shifted_phi_ft = fourier_transform(shifted_phi)
            U += shifted_phi_ft * torch.exp(1j * 2 * np.pi * (fx * i / p + fy * j / p))
    return U


# 滤波函数 A(fx, fy)
def filter_function(fx, fy, p):
    return torch.sinc(fx * p * torch.pi) * torch.sinc(fy * p * torch.pi)

def transfer_function(fx, fy, wavelength, distance):
    fx2_fy2 = fx**2 + fy**2
    mask = fx2_fy2 < (1 / wavelength)**2
    H = torch.zeros_like(fx, dtype=torch.complex64)
    sqrt_term = torch.sqrt(1 - (wavelength * fx)**2 - (wavelength * fy)**2)
    H[mask] = torch.exp(1j * 2 * np.pi / wavelength * sqrt_term[mask] * distance)
    return H

# 优化器
optimizer = torch.optim.Adam([phi], lr=0.01)
iterations = 1000

# 计算频域坐标
fx = torch.fft.fftfreq(W, d=pixel_size).to(device)
fy = torch.fft.fftfreq(H, d=pixel_size).to(device)
FX, FY = torch.meshgrid(fx, fy)

for iter in range(iterations):
    optimizer.zero_grad()

    # 计算调制函数和滤波函数
    U = modulation_function(phi, alpha, p, FX, FY)
    A = filter_function(FX, FY, p)
    H = transfer_function(FX, FY, wavelength, distance)

    # 计算光场
    E = inverse_fourier_transform(U * A * H)

    # 计算误差
    amplitude = torch.abs(E)
    error = amplitude - image

    # 计算损失
    loss = torch.mean(error ** 2)

    # 反向传播
    loss.backward()
    optimizer.step()

    if iter % 10 == 0:
        print(f"Iteration {iter}/{iterations}, Loss: {loss.item()}")

# 最终相位图和重建图像的绘制
phi_np = phi.detach().cpu().numpy()
reconstructed_intensity = amplitude.detach().cpu().numpy()

plt.figure(figsize=(12, 6))

# 绘制相位图
plt.subplot(1, 2, 1)
plt.imshow(phi_np, cmap='gray')
plt.colorbar()
plt.title('Generated Phase Map with Reduced Artifacts')

# 绘制重建的图像
plt.subplot(1, 2, 2)
plt.imshow(reconstructed_intensity, cmap='gray')
plt.colorbar()
plt.title('Reconstructed Intensity Image')

plt.show()