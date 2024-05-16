import torch
import torch.nn.functional as F
from math import log10
import torchvision.utils as utils
from PIL import Image
from torchvision import transforms
from skimage.metrics import structural_similarity as ssim
import numpy as np
import cv2
from skimage import draw


# def to_psnr(img1, img2):
#     mse = F.mse_loss(img1, img2, reduction='none')
#     mse_split = torch.split(mse, 1)
#     mse_list = [torch.mean(torch.squeeze(mse_split[ind])).item() for ind in range(len(mse_split))]
#
#     intensity_max = 1.0
#     psnr_list = [10.0 * log10(intensity_max / mse) for mse in mse_list]
#     return psnr_list

def to_psnr(img1, img2):
    batch_img, channel_img, H_img, W_img = img1.shape[0], img1.shape[1], img1.shape[2], img1.shape[3]
    mse_list = []
    for b in range(batch_img):
        mse = torch.sum((img1[b] - img2[b]) ** 2) / (H_img * W_img)
        mse_list.append(mse)
    mse = sum(mse_list) / len(mse_list)
    psnr = 10.0 * log10(1 / mse)
    return psnr


def to_ssim_skimage(img1, img2):
    img1_list = torch.split(img1, 1)
    img2_list = torch.split(img2, 1)

    img1_list_np = [img1_list[ind].permute(0, 2, 3, 1).data.cpu().numpy().squeeze() for ind in range(len(img1_list))]
    img2_list_np = [img2_list[ind].permute(0, 2, 3, 1).data.cpu().numpy().squeeze() for ind in range(len(img2_list))]
    ssim_list = [ssim(img1_list_np[ind], img2_list_np[ind], data_range=1, multichannel=True) for ind in
                 range(len(img1_list))]
    return ssim_list


def to_ssim(img1, img2):
    batch_img, channel_img, H_img, W_img = img1.shape[0], img1.shape[1], img1.shape[2], img1.shape[3]
    ssim_list = []
    for b in range(batch_img):
        img1_mean = torch.mean(img1[b])
        img2_mean = torch.mean(img2[b])
        img1_2 = torch.sum((img1[b] - img1_mean) ** 2) / (H_img * W_img - 1)
        img2_2 = torch.sum((img2[b] - img2_mean) ** 2) / (H_img * W_img - 1)
        img1_img2 = torch.sum((img1[b] - img1_mean) * (img2[b] - img2_mean)) / (H_img * W_img - 1)
        ssim_list.append((2 * img1_mean * img2_mean + 0.0001) * (2 * img1_img2 + 0.0009) / (
                (img1_mean ** 2 + img2_mean ** 2 + 0.0001) * (img1_2 + img2_2 + 0.0009)))
    ssim_loss = sum(ssim_list) / len(ssim_list)
    return ssim_loss


def to_pearson(img1, img2):
    batch_img, channel_img, H_img, W_img = img1.shape[0], img1.shape[1], img1.shape[2], img1.shape[3]

    PCC_list = []
    for b in range(batch_img):
        img1_mean = torch.mean(img1[b])
        img2_mean = torch.mean(img2[b])
        y = img1[b] - img1_mean
        g = img2[b] - img2_mean
        yg = torch.sum(y * g)
        y2 = torch.sqrt(torch.sum(y * y))
        g2 = torch.sqrt(torch.sum(g * g))
        PCC_list.append(yg / (y2 * g2))
    pcc = -1.0 * sum(PCC_list) / len(PCC_list)
    return pcc


def to_mseloss(img1, img2):
    batch_img, channel_img, H_img, W_img = img1.shape[0], img1.shape[1], img1.shape[2], img1.shape[3]
    mse_list = []
    for b in range(batch_img):
        mse = torch.sum((img1[b] - img2[b]) ** 2) / (H_img * W_img)
        mse_list.append(mse)
    mse = sum(mse_list) / len(mse_list)
    return mse


def creat_obj(smple_path, light_size, radius, binaty_inv, if_obj, device):
    sample = cv2.imread(smple_path, cv2.IMREAD_GRAYSCALE) / 255.0
    # sample = sample[40: 1040, 460: 1460]
    re_size = 500
    sample_resized = cv2.resize(sample, (re_size, re_size))
    pad_size = int((light_size - re_size) / 2)
    sample_padded = np.pad(sample_resized, ((pad_size, pad_size), (pad_size, pad_size)), 'constant',
                           constant_values=(0, 0))
    if binaty_inv == 0:
        _, test = cv2.threshold(sample_padded, 100, 255, cv2.THRESH_BINARY_INV)
    elif binaty_inv == 1:
        _, test = cv2.threshold(sample_padded, 100, 255, cv2.THRESH_BINARY)
    else:
        test = sample_padded

    obj = torch.zeros((light_size, light_size), dtype=torch.float64)
    # rr, cc = draw.rectangle(start=(400, 400), end=(600, 600))
    rr, cc = draw.disk((light_size / 2, light_size / 2), radius=radius)
    obj[rr, cc] = 1.0
    if if_obj:
        obj = obj * test
    return obj.to(device)


def phasemap_8bit(phasemap, inverted=False):
    output_phase = ((phasemap + 2 * np.pi) % (2 * np.pi)) / (2 * np.pi)
    if inverted:
        phase_out_8bit = ((1 - output_phase) * 255).round().cpu().detach().squeeze().numpy().astype(
            np.uint8)  # quantized to 8 bits
    else:
        phase_out_8bit = ((output_phase) * 255).round().cpu().detach().squeeze().numpy().astype(
            np.uint8)  # quantized to 8 bits
    return phase_out_8bit




def gaussian_window(size, sigma):
    """生成一个2D高斯窗口"""
    coords = torch.arange(size, dtype=torch.float64)
    coords -= size // 2
    g = coords ** 2
    g = (-g / (2.0 * sigma ** 2)).exp()
    g /= g.sum()
    return g.outer(g)


def twossim(img1, img2, window_size=11, sigma=1.5, size_average=True):
    """计算两个图像之间的SSIM指数"""
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    window = gaussian_window(window_size, sigma).to(img1.device)
    window = window.expand(1, 1, window_size, window_size)

    mu1 = F.conv2d(img1.unsqueeze(0).unsqueeze(0), window, padding=window_size // 2, groups=1)
    mu2 = F.conv2d(img2.unsqueeze(0).unsqueeze(0), window, padding=window_size // 2, groups=1)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1.unsqueeze(0).unsqueeze(0) * img1.unsqueeze(0).unsqueeze(0), window,
                         padding=window_size // 2, groups=1) - mu1_sq
    sigma2_sq = F.conv2d(img2.unsqueeze(0).unsqueeze(0) * img2.unsqueeze(0).unsqueeze(0), window,
                         padding=window_size // 2, groups=1) - mu2_sq
    sigma12 = F.conv2d(img1.unsqueeze(0).unsqueeze(0) * img2.unsqueeze(0).unsqueeze(0), window,
                       padding=window_size // 2, groups=1) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean() if size_average else ssim_map


def cal_grad(img):
    H, W = img.shape
    grad_x = img[:, 1:] - img[:, :-1]
    grad_y = img[1:, :] - img[:-1, :]
    grad_x = np.pad(grad_x, ((0, 0), (0, 1)), mode='constant', constant_values=0)
    grad_y = np.pad(grad_y, ((0, 1), (0, 0)), mode='constant', constant_values=0)
    return grad_x, grad_y


def sobel_grad(img, device):  # Tensor (batch, channel, high, weight)
    sobel_x_kernel = torch.tensor([[-1., 0., 1.],
                                   [-2., 0., 2.],
                                   [-1., 0., 1.]], dtype=torch.float64, device=device).view(1, 1, 3, 3)
    sobel_y_kernel = torch.tensor([[-1., -2., -1.],
                                   [0., 0., 0.],
                                   [1., 2., 1.]], dtype=torch.float64, device=device).view(1, 1, 3, 3)
    sobel_x = F.conv2d(img, sobel_x_kernel, padding=1)
    sobel_y = F.conv2d(img, sobel_y_kernel, padding=1)
    gradient_magnitude_torch = torch.sqrt(sobel_x ** 2 + sobel_y ** 2)
    return sobel_x, sobel_y


def pad_tensor(input_tensor, target_height, target_width, pad_value=0):

    input_shape = input_tensor.shape
    if len(input_shape) == 2:  # (H, W)
        C = 1
        H, W = input_shape
        input_tensor = input_tensor.unsqueeze(0)
    elif len(input_shape) == 3:  # (C, H, W)
        C, H, W = input_shape
    else:
        raise ValueError("Shape of Input must be 2d or 3d")

    pad_left = (target_width - W) // 2
    pad_right = target_width - W - pad_left
    pad_top = (target_height - H) // 2
    pad_bottom = target_height - H - pad_top

    padded_tensor = F.pad(input_tensor, (pad_left, pad_right, pad_top, pad_bottom), value=pad_value)

    return padded_tensor

def pad_array(input_array, target_height, target_width, pad_value=0):

    input_shape = input_array.shape
    if len(input_shape) == 2:  # (H, W)
        H, W = input_shape
        input_array = np.expand_dims(input_array, axis=0)
        C = 1
    elif len(input_shape) == 3:  # (C, H, W)
        C, H, W = input_shape
    else:
        raise ValueError("Shape of Input must be 2d or 3d")

    # 计算每边的padding大小
    pad_left = (target_width - W) // 2
    pad_right = target_width - W - pad_left
    pad_top = (target_height - H) // 2
    pad_bottom = target_height - H - pad_top

    # 使用指定值进行padding
    padding = ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right))
    padded_array = np.pad(input_array, padding, mode='constant', constant_values=pad_value)

    return padded_array


