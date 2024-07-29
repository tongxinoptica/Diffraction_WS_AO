"""
Holography_Highorder:
This script generates phase patterns for a handful of 2D
target examples using various methods.
-----
Example Usage:
$ python main.py --method=DPAC
"""

import sys

sys.path.append('../code')
import os, imageio, torch, math
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import configargparse
from tensorboardX import SummaryWriter
from torchvision.transforms.functional import resize
import scipy.io as sio

from utils.image_loader import ImageLoader
from modules import DPAC, SGD, HOGD
from propagation_ASM import Propagator
from utils.physical_prop_module import PhysicalProp
import utils.utils as utils

# Commandline arguments
p = configargparse.ArgumentParser()
p.add_argument('--experiment', type=str, default=None, 
    help='Name of the experiment')
p.add_argument('--data_path', type=str, 
    default='image',
    help='data path for target images')
p.add_argument('--homography_path', type=str, 
    default='hologram', help='path for homography phase patterns')
p.add_argument('--output_path', type=str, default='hologram',
    help='Output path for tensorboard results and phases')
p.add_argument('--tb_visualizations', type=utils.str2bool, default=False,
    help='If true, send visualizations for the output phase to tensorboard')
p.add_argument('--method', type=str, default='HOGD',
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
p.add_argument('--prop_dist', type=float, default=0.1,
    help='distance from SLM to target plane in mm')
p.add_argument('--pixel_pitch', type=float, default=None,
    help='pixel pitch of SLM in micrometers')
p.add_argument('--supervision_box_blur', type=int, default=6,
    help='Blur of target supervision. Default is SLM res.')
p.add_argument('--downsampled_loss', type=utils.str2bool, default=False,
    help='If true, downsample blurred target and recon before computing loss')
p.add_argument('--compact_vr', type=utils.str2bool, default=True,
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
p.add_argument('--pupil', type=float, default=None,
    help='radius of pupil in millimeters')
p.add_argument('--eyepiece', type=float, default=50,
    help='eyepiece focal length in millimeters')
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
device = torch.device('cuda:3')
exposure_list = [0.3, 0.5, 1.4]
if opt.prop_dist is not None:
    prop_dist = [opt.prop_dist, opt.prop_dist, opt.prop_dist]
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
if opt.pupil is not None:
    sum_id += f'_pupil{opt.pupil}_E{opt.eyepiece}'
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
total_loss = 0.0
for sample in tqdm(loader):
    # Load scene data and send to tensorboard
    scene_amp, idx = sample
    scene_amp = (scene_amp/scene_amp.max()).to(device)
    writer.add_image(f'Image', scene_amp[0,...], idx)
    # if opt.periphery_box_blur is not None:
    #     writer.add_image(f'Foveated Target', (foveation_mask*scene_amp)[0,...], idx)

    # Run algorithm for each color
    slm_phases = []
    final_amps = []
    for c in channels:
        # Initialize propagator for one color at a time to limit memory usage
        propagator = Propagator([prop_dist[c]], pitch, wavelengths[c],
            pad_multiplier=opt.pad_multiplier, slm_res=slm_res,
            target_res=roi_res, order_num=order_num,
            filter=fourier_filter_aperture, 
            pupil=(opt.pupil*mm if ((opt.pupil is not None) and ("HOGD" == opt.method.upper())) else None),
            eyepiece=opt.eyepiece*mm, device=device)
        
        # Hardware setup for CITL
        if opt.CITL:
            if opt.channel is None:
                print(f'Set the LUT, laser, and exposure for {chan_str[c]}')
                input('Press enter to continue: ')

            camera_prop = PhysicalProp(c, laser_arduino=False,
                roi_res=(roi_res[1], roi_res[0]), slm_settle_time=2*exposure_list[c]+0.12,
                range_row=(0, 1200), range_col=(0, 1920),
                patterns_path=opt.homography_path, show_preview=False)
        else:
            camera_prop = None

        if ("DPAC" == opt.method.upper()): 
            algorithm = DPAC(torch.Tensor([prop_dist[c]]), wavelengths[c],
                preblur_sigma=opt.preblur_sigma, propagator=propagator,
                device=device)
            target_amp = scene_amp[:,c:c+1,...]
        elif ("SGD" == opt.method.upper()):
            algorithm = SGD(opt.num_iters, propagator, loss=nn.MSELoss(),
                lr=opt.lr, lr_s=opt.lr_s, s0=opt.s0,
                full_bandwidth=opt.full_bandwidth, camera_prop=camera_prop,
                phase_path=summaries_dir, writer=writer, device=device)
            target_amp = scene_amp[:,c:c+1,...]
        elif ("HOGD" == opt.method.upper()):
            algorithm = HOGD(opt.num_iters, propagator, loss=nn.MSELoss(),
                lr=opt.lr, lr_s=opt.lr_s, s0=opt.s0, camera_prop=camera_prop,
                citl_y_offset=opt.citl_y_offset,
                citl_x_offset=opt.citl_x_offset, phase_path=summaries_dir,
                writer=writer, device=device,
                supervision_box_blur=opt.supervision_box_blur,
                downsampled_loss=opt.downsampled_loss,
                periphery_box_blur=opt.periphery_box_blur)

            # Upsample Target Amp to Resolution of Higher Order Propagation
            # Not used for CITL since only lower res can be captured
            if not opt.CITL:
                target_amp = resize(scene_amp[:,c:c+1,...],
                    [6*i for i in roi_res])
            else:
                target_amp = scene_amp[:,c:c+1,...]

        # Run phase generation algorithm
        algorithm.phase_path = os.path.join(summaries_dir,
            f'phase_{chan_str[c]}_{idx}')
        algorithm.tb_prefix = f'{chan_str[c]}_{idx}'
        if opt.method.upper() in ['SGD', 'HOGD']:
            # iterative methods, initial phase: uniform random
            init_phase = opt.init_range * (-0.5 + 1.0 * torch.rand(1, 1,
                *slm_res)).to(device)
            if opt.method.upper() == 'HOGD':
                if opt.CITL:
                    loss_val, final_amp, slm_phase = algorithm(target_amp, init_phase,
                        foveation_mask=foveation_mask)
                    final_amps.append(final_amp)
                    total_loss += loss_val*(1.0/len(channels))
                else:
                    slm_phase = algorithm(target_amp, init_phase,
                        foveation_mask=foveation_mask)
            else:
                if opt.CITL:
                    loss_val, final_amp, slm_phase = algorithm(target_amp, init_phase)
                    final_amps.append(final_amp)
                    total_loss += loss_val*(1.0/len(channels))
                else:
                    slm_phase = algorithm(target_amp, init_phase)
                    
        elif opt.method.upper() == "DPAC":
            # direct methods
            slm_phase = algorithm(target_amp)
        propagator = None
        slm_phases.append(slm_phase)

        # Format and save phases for SLM
        # 8bit phase
        phase_out_8bit = utils.phasemap_8bit(slm_phase, inverted=True)
        final_outfolder = os.path.join(summaries_dir, f'final_phase_{idx}')
        utils.cond_mkdir(final_outfolder)
        imageio.imwrite(os.path.join(final_outfolder, f'{chan_str[c]}.png'),
            phase_out_8bit)
        
    if opt.CITL:
        final_amps = torch.cat(final_amps, 1).cpu().detach().numpy()[0,...]
        final_amps = np.moveaxis(final_amps, 0, -1)
        imageio.imwrite(os.path.join(final_outfolder, f'capture.png'),
            final_amps)
        sio.savemat(os.path.join(opt.output_path, f'loss.mat'), {f'mse': total_loss})
            
