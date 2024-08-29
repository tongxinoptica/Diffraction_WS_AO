## A simple capture

import os
import sys
import numpy as np
import skimage.io
import cv2
import time
import imageio
import serial
import utils.utils as utils
import scipy.io as sio
import torch
import skimage.io
from utils.physical_prop_module import PhysicalProp
import math


import configargparse

# Command line argument processing
p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

p.add_argument('--channel', type=int, default=1, help='red:0, green:1, blue:2, rgb:3')
p.add_argument('--num_frames_avg', type=int, default=100, help='number of pixels to average')
p.add_argument('--num_adj_pixels', type=int, default=1, help='number of adjacent pixels to change')
p.add_argument('--phase_path', type=str, default=None, help='phase path')

# parse arguments
opt = p.parse_args()
print(f'Number of frames: {opt.num_frames_avg}')
chan_str = ('red', 'green', 'blue')[opt.channel]
use_warper = True

roi_res = (960, 1680)
range_row = (0, 1200)
range_col = (0, 1920)
homography_path = 'G:\\Shared drives\\Stanford Computational Imaging\\Projects\\holography_highorder\\data\\phase_patterns\\0721_calibrated_dist_homographies'
camera_prop = PhysicalProp(opt.channel, laser_arduino=False, roi_res=(roi_res[1], roi_res[0]),
                            slm_settle_time=2*0.3+0.12, range_row=range_row, range_col=range_col,
                            patterns_path=homography_path, show_preview=False)

device = torch.device('cuda')
phi = torch.zeros(1, 1, 1080, 1920).to(device)
if opt.phase_path is not None:
    slm_phase = skimage.io.imread(opt.phase_path) / 256.0
    phi = torch.tensor((1 - slm_phase) * 2 * np.pi - np.pi,
                             dtype=torch.float32).reshape(1, 1, 1080, 1920).to(torch.device('cuda'))

# Change phi
center = (540, 960)
dphi = torch.zeros_like(phi)
dphi[:,:, center[0] - round((opt.num_adj_pixels-1)/2):center[0] + round((opt.num_adj_pixels+1)/2),
          center[1] - round((opt.num_adj_pixels-1)/2):center[1] + round((opt.num_adj_pixels+1)/2)] = math.pi

phi2 = phi.clone() + dphi

# result_path = f'D:/210212/gradient_captured_{opt.plane_idx}thplane_red/'
result_path = f'C://Users//EE267//Desktop//holography_highorder//code'
utils.cond_mkdir(result_path)

diff = torch.zeros(1, 1, *roi_res).to(device)


# Average over multiple frames
for i in range(opt.num_frames_avg):
    if (i % 10 == 0):
        print(i)

    amp1 = camera_prop(phi)  # original phase
    amp2 = camera_prop(phi2)  # original phase
    if (i % 10 == 0):
        print('----------------------------')
        print(amp1.max())
        print(amp1.mean())
        print(amp2.max())
        print(amp2.mean())
    diff += amp2 - amp1

diff /= opt.num_frames_avg

sio.savemat(os.path.join(result_path, f'captured_grad_{opt.num_adj_pixels}pixels_{opt.num_frames_avg}frames_{chan_str}.mat'),
            {f'grad': diff.squeeze().cpu().detach().numpy(),
             'amp1':amp1.squeeze().cpu().detach().numpy(),
             'amp2':amp2.squeeze().cpu().detach().numpy()})
