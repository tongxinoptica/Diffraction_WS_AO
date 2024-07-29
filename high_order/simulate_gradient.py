## A simple capture

import os
import sys
import numpy as np
import skimage.io
import cv2
import time
import imageio
import utils.utils as utils
import scipy.io as sio
import torch
import skimage.io
from propagation_ASM import Propagator
import math


import configargparse

# Command line argument processing
p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

p.add_argument('--channel', type=int, default=1, help='red:0, green:1, blue:2, rgb:3')
p.add_argument('--num_adj_pixels', type=int, default=3, help='number of adjacent pixels to change')
p.add_argument('--phase_path', type=str, default=None, help='phase path')

result_path = f'0731_gradients'
utils.cond_mkdir(result_path)

# parse arguments
opt = p.parse_args()
chan_str = ('red', 'green', 'blue')[opt.channel]
use_warper = True
cm, mm, um, nm = 1e-2, 1e-3, 1e-6, 1e-9
roi_res = (960, 1680)
slm_res = (1080, 1920)
wavelengths = (636.4 * nm, 517.7 * nm, 440.8 * nm)  # Laser
pitch = 6.4 * um
prop_dist = [34.7*mm, 35.0*mm, 35.0*mm]
device = torch.device('cuda')

phi = torch.zeros(1, 1, *slm_res).to(device)
if opt.phase_path is not None:
    slm_phase = skimage.io.imread(opt.phase_path) / 256.0
    phi = torch.tensor((1 - slm_phase) * 2 * np.pi - np.pi,
                             dtype=torch.float32).reshape(1, 1, 1080, 1920).to(device)

# Change phi
center = (540, 960)
dphi = torch.zeros_like(phi)
dphi[:,:, center[0] - round((opt.num_adj_pixels-1)/2):center[0] + round((opt.num_adj_pixels+1)/2),
          center[1] - round((opt.num_adj_pixels-1)/2):center[1] + round((opt.num_adj_pixels+1)/2)] = math.pi
phi2 = phi.clone() + dphi

slm_field = utils.polar_to_rect(torch.ones_like(phi), phi)
slm_field2 = utils.polar_to_rect(torch.ones_like(phi2), phi2)


propagator = Propagator([prop_dist[opt.channel]], pitch, wavelengths[opt.channel],
    pad_multiplier=-1, slm_res=slm_res,
    target_res=roi_res, order_num=3,
    filter=3.0, device=device)

amp1 = utils.full_bandwidth_amplitude(propagator(slm_field))  # original phase
amp1 = utils.intensity_blur(amp1, 6)
amp1 = amp1[...,3::6,3::6]
amp2 = utils.full_bandwidth_amplitude(propagator(slm_field2))  # original phase
amp2 = utils.intensity_blur(amp2, 6)
amp2 = amp2[...,3::6,3::6]

diff = amp2 - amp1

sio.savemat(os.path.join(result_path, f'HOGD_simulated_grad_{opt.num_adj_pixels}pixels_{chan_str}.mat'),
            {f'grad': diff.squeeze().cpu().detach().numpy(),
             'amp1':amp1.squeeze().cpu().detach().numpy(),
             'amp2':amp2.squeeze().cpu().detach().numpy()})



propagator = Propagator([prop_dist[opt.channel]], pitch, wavelengths[opt.channel],
    pad_multiplier=-1, slm_res=slm_res,
    target_res=roi_res, order_num=1,
    filter=1.0, device=device)

amp1 = propagator(slm_field).abs()  # original phase
amp2 = propagator(slm_field2).abs()  # original phase
diff = amp2 - amp1

sio.savemat(os.path.join(result_path, f'SGD_simulated_grad_{opt.num_adj_pixels}pixels_{chan_str}.mat'),
            {f'grad': diff.squeeze().cpu().detach().numpy(),
             'amp1':amp1.squeeze().cpu().detach().numpy(),
             'amp2':amp2.squeeze().cpu().detach().numpy()})
