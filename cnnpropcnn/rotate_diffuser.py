import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fftshift, ifftshift, fft2, ifft2
from scipy.ndimage import rotate

# 定义参数
wavelength = 532e-9  # 波长，单位：米
k = 2 * np.pi / wavelength
pixel_size = 8e-6  # 像素大小，单位：米
grid_size = 1080  # 网格大小
physical_size = grid_size * pixel_size  # 物理尺寸
spot_radius = 1e-3  # 光斑半径，单位：米

# 定义坐标网格
x = np.linspace(-physical_size / 2, physical_size / 2, grid_size)
y = np.linspace(-physical_size / 2, physical_size / 2, grid_size)
X, Y = np.meshgrid(x, y)
R = np.sqrt(X ** 2 + Y ** 2)

# 窗口函数
window = np.hanning(grid_size)[:, None] * np.hanning(grid_size)[None, :]

# 初始光斑的随机振幅和相位
np.random.seed(0)  # 固定随机种子
frames = 100
intensity_sum = np.zeros((grid_size, grid_size))
phase_sum = np.zeros((grid_size, grid_size))


# 透镜的传递函数
def lens_transfer_function(shape, f, wavelength):
    N = shape[0]
    L = N * pixel_size
    fx = np.fft.fftfreq(N, d=L / N)
    fy = np.fft.fftfreq(N, d=L / N)
    FX, FY = np.meshgrid(fx, fy)
    H = np.exp(-1j * (np.pi * wavelength * f) * (FX ** 2 + FY ** 2))
    return fftshift(H)
def lens_phase(X, Y, k, f):  # X and Y is space coordinate
    len_p = (k * (X ** 2 + Y ** 2) / (2 * f)) % (2 * np.pi) + k * 1.4 * 0.003
    return len_p  # (0 , 2pi)

# 角谱传播函数
def angular_spectrum_propagation(E, z, wavelength, dx):
    k = 2 * np.pi / wavelength
    N = E.shape[0]
    L = N * dx
    fx = np.fft.fftfreq(N, d=L / N)
    fy = np.fft.fftfreq(N, d=L / N)
    FX, FY = np.meshgrid(fx, fy)
    H = np.exp(1j * z * np.sqrt(k ** 2 - (2 * np.pi * FX) ** 2 - (2 * np.pi * FY) ** 2))
    E_f = fftshift(fft2(ifftshift(E)))
    E_fz = E_f * H
    E_z = fftshift(ifft2(ifftshift(E_fz)))
    return E_z


# 计算透镜的传递函数
f_lens = 0.1  # 透镜焦距，单位：米
H_lens = lens_transfer_function((grid_size, grid_size), f=f_lens, wavelength=wavelength)

# 定义传播距离
propagation_distance = 0.01  # 传播距离，单位：米
dx = pixel_size  # 计算空间分辨率

# 初始高斯光束
A_0 = 1  # 光束中心的振幅
gaussian_beam = A_0 * np.exp(-R**2 / spot_radius**2)

# 生成旋转散射片的初始相位分布
scatter_phase = np.random.rand(grid_size, grid_size) * 2 * np.pi
scatter_amp = np.random.rand(grid_size, grid_size)
scatter = 1 * np.exp(1j*scatter_phase)
for t in range(frames):
    amplitude = np.random.rand(grid_size, grid_size) * 1
    phase = np.random.rand(grid_size, grid_size) * 2 * np.pi * 0.5
    E0 = gaussian_beam * amplitude * np.exp(1j*phase)


    # 定义傅里叶变换和逆傅里叶变换函数
    def fourier_transform(E):
        return fftshift(fft2(ifftshift(E)))


    def inverse_fourier_transform(E):
        return ifftshift(ifft2(fftshift(E)))


    # 透镜聚焦
    E1 = inverse_fourier_transform(fourier_transform(E0) * H_lens)

    # angle = t * (360 / frames)  # 计算旋转角度
    # rotated_phase = rotate(scatter, angle, reshape=False, order=1)

    E2 = E1 * scatter  # 经过旋转散射片后的电场分布
    top_rows = scatter[:4, :]
    scatter[:-4, :] = scatter[4:, :]
    scatter[-4:, :] = top_rows

    # 自由空间传播（角谱方法）
    E3 = angular_spectrum_propagation(E2, propagation_distance, wavelength, dx)

    # 针孔滤波
    pinhole_radius = 100e-6  # 针孔半径，单位：米
    pinhole_mask = R <= pinhole_radius
    E4 = E3 * pinhole_mask  # 经过针孔后的电场分布

    # 透镜准直（傅里叶逆变换）
    Ef = inverse_fourier_transform(fourier_transform(E4) * H_lens)

    intensity_sum += np.abs(Ef) ** 2
    phase_sum += np.angle(Ef)

# 时间平均后的光强分布
intensity_avg = intensity_sum / frames

# 时间平均后的相位分布
phase_avg = phase_sum / frames

# 应用窗口函数
intensity_avg *= window

# 绘制强度结果
plt.figure(figsize=(8, 8))
plt.imshow(intensity_avg, extent=(-physical_size / 2, physical_size / 2, -physical_size / 2, physical_size / 2),
           cmap='hot')
plt.colorbar()
plt.title('Time-averaged Far-field Intensity after Pin-hole and Collimating Lens')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.show()

# 绘制相位结果
plt.figure(figsize=(8, 8))
plt.imshow(phase_avg, extent=(-physical_size / 2, physical_size / 2, -physical_size / 2, physical_size / 2), cmap='hsv')
plt.colorbar()
plt.title('Time-averaged Phase after Pin-hole and Collimating Lens')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.show()

# 计算空间相干性（互相干函数）
coherence_length = 10e-6  # 互相干函数的计算范围
coherence_points = 400  # 计算点数


def compute_coherence(intensity_avg):
    # 选择中心区域用于计算互相干函数
    intensity_avg_center = intensity_avg[grid_size // 2 - coherence_points // 2:grid_size // 2 + coherence_points // 2,
                           grid_size // 2 - coherence_points // 2:grid_size // 2 + coherence_points // 2]

    # 计算傅里叶变换
    intensity_avg_center_ft = np.fft.fft2(intensity_avg_center)

    # 计算共轭乘积的逆傅里叶变换，得到互相干函数
    coherence_function = np.fft.ifft2(intensity_avg_center_ft * np.conj(intensity_avg_center_ft))

    # 傅里叶变换的结果需要经过中心化处理
    coherence_function = np.fft.fftshift(coherence_function)

    return coherence_function


coherence = np.abs(compute_coherence(intensity_avg))

# 绘制空间相干性
plt.figure(figsize=(8, 8))
plt.imshow(coherence / np.max(coherence),cmap='hot')
# plt.colorbar()
# plt.title('Spatial Coherence after Rotating Scattering Plate and Pin-hole')
# plt.xlabel('x (m)')
# plt.ylabel('y (m)')
plt.show()
plt.imsave(f'R={pinhole_radius}.png', coherence, cmap='hot')
