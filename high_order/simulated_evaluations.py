"""
Holography_Highorder:
This script generates phase patterns for a handful of 2D
target examples using various methods.
-----
Example Usage:
$ python main.py --method=DPAC
"""

import sys

sys.path.append('../../code')
import os, imageio, torch, math
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import configargparse
import scipy.io as sio
from tensorboardX import SummaryWriter
from torchvision.transforms.functional import resize

import utils.utils as utils
from utils.image_loader import ImageLoader
from modules import DPAC, SGD, HOGD
from propagation_ASM import Propagator
from utils.physical_prop_module import PhysicalProp

# Commandline arguments
p = configargparse.ArgumentParser()
p.add_argument('--experiment', type=str, default=None, 
    help='Name of the experiment')
p.add_argument('--data_path', type=str, 
    default='C:\\Users\\EE267\\Desktop\\neural-holography\\data_final_2d',
    help='data path for target images')
p.add_argument('--homography_path', type=str, 
    default='G:\\Shared drives\\Stanford Computational Imaging\\Projects\\'+
    'holography_highorder\\data\\phase_patterns\\0721_distance_calibration\\'+
    'homography_phases', help='path for homography phase patterns')
p.add_argument('--output_path', type=str, default='0722_CITL',
    help='Output path for tensorboard results and phases')
p.add_argument('--tb_visualizations', type=utils.str2bool, default=False,
    help='If true, send visualizations for the output phase to tensorboard')
p.add_argument('--method', type=str, default='DPAC',
    help='Method for generating phases (DPAC, SGD, or HOGD)')
p.add_argument('--CITL', type=utils.str2bool, default=False,
    help='Use of Camera-in-the-loop optimization with SGD or HOGD')
p.add_argument('--citl_y_offset', type=int, default=3,
    help='Y offset for Camera-in-the-loop sampling with HOGD in [0, 6)')
p.add_argument('--citl_x_offset', type=int, default=3,
    help='X offset for Camera-in-the-loop sampling with HOGD in [0, 6)')
p.add_argument('--preblur_sigma', type=float, default=0.0,
    help='Blur to apply at SLM before performing DPAC')
p.add_argument('--lr', type=float, default=0.03,
    help='learning rate for phase with SGD, HOGD')
p.add_argument('--lr_s', type=float, default=0.0,
    help='learning rate for scale factor with SGD, HOGD, \
        set to 0.0 to use optimal scaling each iteration')
p.add_argument('--s0', type=float, default=1.0,
    help='initialization for scale factor with SGD, HOGD')
p.add_argument('--pad_multiplier', type=float, default=-1,
    help='Amount of padding as multiple of FOV kernel')
p.add_argument('--init_range', type=float, default=2.5,
    help='multiplier to init phase for SGD, HOGD')
p.add_argument('--num_iters', type=int, default=2500,
    help='Number of iterations for SGD, HOGD')
p.add_argument('--channel', type=int, default=None,
    help='if set, generate for single plane at that channel')
p.add_argument('--img_idx', type=int, default=None,
    help='If set, index of image to generate phase pattern for')
p.add_argument('--prop_dist', type=float, default=None,
    help='distance from SLM to target plane in mm')
p.add_argument('--pixel_pitch', type=float, default=None,
    help='pixel pitch of SLM in micrometers')
p.add_argument('--supervision_box_blur', type=int, default=6,
    help='Blur of target supervision. Default is SLM res.')
p.add_argument('--downsampled_loss', type=utils.str2bool, default=False,
    help='If true, downsample blurred target and recon before computing loss')
p.add_argument('--compact_vr', type=utils.str2bool, default=False,
    help='If true, use settings for compact vr')
p.add_argument('--full_bandwidth', type=utils.str2bool, default=False,
    help='If true, use full bandwidth amplitude for SGD')
p.add_argument('--periphery_box_blur', type=int, default=None,
    help='If set, blur of periphery supervision.')
p.add_argument('--foveation_center_x', type=float, default=0.75,
    help='Normalized x coord of foveation center')
p.add_argument('--foveation_center_y', type=float, default=0.5,
    help='Normalized y coord of foveation center')
p.add_argument('--foveation_radius', type=float, default=0.2,
    help='Normalized radius high res foveation center')
opt = p.parse_args()

# Set simulation parameters
cm, mm, um, nm = 1e-2, 1e-3, 1e-6, 1e-9
if opt.method.upper() == 'HOGD':
    fourier_filter_aperture = 3.0
    order_num = 3
else:
    fourier_filter_aperture = 1.0
    order_num = 1
if opt.pixel_pitch is None:
    pitch = 6.4 * um
else:
    pitch = opt.pixel_pitch * um
if opt.compact_vr:
    pitch = 8.0 * um
wavelengths = (636.4 * nm, 517.7 * nm, 440.8 * nm)  # Laser
if opt.compact_vr:
    opt.channel = 1
    wavelengths = (532 * nm, 532 * nm, 532 * nm)
slm_res = (1080, 1920)  # 1080p slm
roi_res = (960, 1680)
if (opt.img_idx is not None) and (opt.img_idx == 0):
    # Use full ROI for homography pattern
    roi_res = (1080, 1920)
    print(f'Adjusted ROI {roi_res} for Homography Pattern')
chan_str = ('red', 'green', 'blue')
channels = [opt.channel] if opt.channel is not None else range(3)
device = torch.device('cuda')
if opt.prop_dist is not None:
    prop_dist = [opt.prop_dist*mm, opt.prop_dist*mm, opt.prop_dist*mm]
else:
    prop_dist = [34.7*mm, 35.0*mm, 35.0*mm]

# Initialize tensorboard writer
sum_id = f'{opt.method}'
if opt.full_bandwidth:
    sum_id += f'_FullBandwidth'
if opt.experiment is not None:
    sum_id += f'_{opt.experiment}'
if opt.method.upper() in ['SGD', 'HOGD']:
    sum_id += f'_lr{opt.lr}_Init{opt.init_range}'
    sum_id += f'_lrs{opt.lr_s}_s0{opt.s0}'
    sum_id += f'_Iters{opt.num_iters}'
if opt.method.upper() == 'HOGD':
    sum_id += f'_supervision{opt.supervision_box_blur}'
    if opt.downsampled_loss:
        sum_id += f'_downsampled_loss'
    if opt.periphery_box_blur is not None:
        sum_id += f'_periphery{opt.periphery_box_blur}'
        sum_id += f'_X{opt.foveation_center_x}_Y{opt.foveation_center_y}'
        sum_id += f'_R{opt.foveation_radius}'
if opt.method.upper() == 'DPAC':
    sum_id += f'_Preblur{opt.preblur_sigma}'
if opt.img_idx is not None:
    sum_id += f'_Img{opt.img_idx}'
if opt.channel is not None:
    sum_id += f'_{chan_str[opt.channel]}'
if opt.pixel_pitch is not None:
    sum_id += f'_pitch{opt.pixel_pitch}'
if opt.compact_vr:
    sum_id += f'_compactVR'
if opt.CITL:
    sum_id += f'_CITL'
sum_id += f'_Dist{opt.prop_dist}_pad{opt.pad_multiplier}'
summaries_dir = os.path.join(opt.output_path, sum_id)
utils.cond_mkdir(summaries_dir)
writer = SummaryWriter(f'{summaries_dir}')

# Initialize data loader
if opt.img_idx is not None:
    idx_subset = [opt.img_idx]
else:
    idx_subset = None
loader = ImageLoader(opt.data_path, channel=None, image_res=roi_res,
    idx_subset=idx_subset)

# Create Masks for High and Low Res regions for Foveation
if opt.periphery_box_blur is not None:
    foveation_mask = utils.foveation_masking(roi_res,
        opt.foveation_center_x, opt.foveation_center_y,
        opt.foveation_radius).to(device)
    writer.add_image(f'Foveation Mask', foveation_mask[0,...], 0)
else:
    foveation_mask = None

# Loop over images
for sample in tqdm(loader):
    # Load scene data and send to tensorboard
    scene_amp, idx = sample
    scene_amp = (scene_amp/scene_amp.max()).to(device)
    final_outfolder = os.path.join(summaries_dir, f'final_phase_{idx}')

    # Run algorithm for each color
    slm_phases = []
    for c in channels:
        # Load phase pattern for specific distance
        slm_phase = imageio.imread(os.path.join(final_outfolder, f'{chan_str[c]}.png'))
        slm_phase = (1 - slm_phase / 256.0) * 2 * np.pi - np.pi
        slm_phase = torch.tensor(slm_phase, dtype=torch.float32).reshape(1, 1, *slm_phase.shape)
        slm_phases.append(slm_phase.to(device))

    # Write reconstructed visualizations to tensorboard
    with torch.no_grad():
        writer.add_image(f'SLM_phases', 
            torch.cat(slm_phases, 1)[0,...]/(2*math.pi)+0.5, idx)

        # Evaluation parameters for different filtering levels
        order_nums = [1, 1, 3, 3, 3]
        filters = [0.5, 1.0, 0.5, 1.0, 3.0]

        mse_list = utils.evaluate_phase(slm_phases, scene_amp, prop_dist,
            pitch, wavelengths, roi_res, slm_res, order_nums, filters,
            masks_c=None, loss=nn.MSELoss(),
            full_bandwidth=opt.full_bandwidth,
            pad_multiplier=opt.pad_multiplier,
            supervision_box_blur=opt.supervision_box_blur,
            writer=writer, tb_idx=idx, prop_visualizations=False)

        for k in range(len(mse_list)):
            sio.savemat(os.path.join(opt.output_path,
                f'{sum_id}_filter{filters[k]}_order{order_nums[k]}_Img{idx}.mat'),
                {f'mse': mse_list[k]})

        if (9 in idx): 
            # Generate propagation visualizations for 3x3 propagation             
            _ = utils.evaluate_phase(slm_phases, scene_amp, prop_dist,
                pitch, wavelengths, roi_res, slm_res, [3], [3.0],
                masks_c=None, loss=nn.MSELoss(),
                full_bandwidth=opt.full_bandwidth,
                pad_multiplier=opt.pad_multiplier,
                supervision_box_blur=opt.supervision_box_blur,
                writer=writer, tb_idx=idx, prop_visualizations=True)

        torch.cuda.empty_cache()
