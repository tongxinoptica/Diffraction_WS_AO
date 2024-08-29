
import os, sys
sys.path.append('../../code')
import numpy as np
import torch
import torch.nn as nn
from imageio import imread, imwrite
from utils.physical_prop_module import PhysicalProp
from utils.image_loader import ImageLoader


# Expected Files
# {phase_path[n]}/{red/green/blue}.png contains the  phase patterns for the nth target image
# {data_path}/img_{target_idxs[n]}.png is the nth target image
# {homography_path}/{red/green/blue}.png is the homography phase pattern for an estimated distance

# Parameters
target_idxs = [8, 10, 18, 8, 10, 18, 8, 10, 18, 8, 10, 18, 8, 10, 18]
phase_paths = ['G:\\Shared drives\\Stanford Computational Imaging\\Projects\\holography_highorder\\data\\phase_patterns\\0804_phases\\DPAC_Preblur0.0_Img8_DistNone_pad-1\\final_phase_[8]',
               'G:\\Shared drives\\Stanford Computational Imaging\\Projects\\holography_highorder\\data\\phase_patterns\\0804_phases\\DPAC_Preblur0.0_Img10_DistNone_pad-1\\final_phase_[10]',
               'G:\\Shared drives\\Stanford Computational Imaging\\Projects\\holography_highorder\\data\\phase_patterns\\0804_phases\\DPAC_Preblur0.0_Img18_DistNone_pad-1\\final_phase_[18]',
               'G:\\Shared drives\\Stanford Computational Imaging\\Projects\\holography_highorder\\data\\phase_patterns\\0804_phases\\HOGD_lr0.03_Init1.0_lrs0.0_s01.0_Iters2500_supervision6_Img8_DistNone_pad-1\\final_phase_[8]',
               'G:\\Shared drives\\Stanford Computational Imaging\\Projects\\holography_highorder\\data\\phase_patterns\\0804_phases\\HOGD_lr0.03_Init1.0_lrs0.0_s01.0_Iters2500_supervision6_Img10_DistNone_pad-1\\final_phase_[10]',
               'G:\\Shared drives\\Stanford Computational Imaging\\Projects\\holography_highorder\\data\\phase_patterns\\0804_phases\\HOGD_lr0.03_Init1.0_lrs0.0_s01.0_Iters2500_supervision6_Img18_DistNone_pad-1\\final_phase_[18]',
               'G:\\Shared drives\\Stanford Computational Imaging\\Projects\\holography_highorder\\data\\phase_patterns\\0804_phases\\SGD_lr0.03_Init1.0_lrs0.0_s01.0_Iters2500_Img8_DistNone_pad-1\\final_phase_[8]',
               'G:\\Shared drives\\Stanford Computational Imaging\\Projects\\holography_highorder\\data\\phase_patterns\\0804_phases\\SGD_lr0.03_Init1.0_lrs0.0_s01.0_Iters2500_Img10_DistNone_pad-1\\final_phase_[10]',
               'G:\\Shared drives\\Stanford Computational Imaging\\Projects\\holography_highorder\\data\\phase_patterns\\0804_phases\\SGD_lr0.03_Init1.0_lrs0.0_s01.0_Iters2500_Img18_DistNone_pad-1\\final_phase_[18]',
               'G:\\Shared drives\\Stanford Computational Imaging\\Projects\\holography_highorder\\data\\phase_patterns\\0804_phases\\HOGD_lr0.03_Init2.5_lrs0.0_s01.0_Iters2500_supervision6_Img8_DistNone_pad-1\\final_phase_[8]',
               'G:\\Shared drives\\Stanford Computational Imaging\\Projects\\holography_highorder\\data\\phase_patterns\\0804_phases\\HOGD_lr0.03_Init2.5_lrs0.0_s01.0_Iters2500_supervision6_Img10_DistNone_pad-1\\final_phase_[10]',
               'G:\\Shared drives\\Stanford Computational Imaging\\Projects\\holography_highorder\\data\\phase_patterns\\0804_phases\\HOGD_lr0.03_Init2.5_lrs0.0_s01.0_Iters2500_supervision6_Img18_DistNone_pad-1\\final_phase_[18]',
               'G:\\Shared drives\\Stanford Computational Imaging\\Projects\\holography_highorder\\data\\phase_patterns\\0804_phases\\SGD_lr0.03_Init2.5_lrs0.0_s01.0_Iters2500_Img8_DistNone_pad-1\\final_phase_[8]',
               'G:\\Shared drives\\Stanford Computational Imaging\\Projects\\holography_highorder\\data\\phase_patterns\\0804_phases\\SGD_lr0.03_Init2.5_lrs0.0_s01.0_Iters2500_Img10_DistNone_pad-1\\final_phase_[10]',
               'G:\\Shared drives\\Stanford Computational Imaging\\Projects\\holography_highorder\\data\\phase_patterns\\0804_phases\\SGD_lr0.03_Init2.5_lrs0.0_s01.0_Iters2500_Img18_DistNone_pad-1\\final_phase_[18]']
homography_path = 'G:\\Shared drives\\Stanford Computational Imaging\\Projects\\holography_highorder\\data\\phase_patterns\\0721_calibrated_dist_homographies'
data_path = 'G:\\Shared drives\\Stanford Computational Imaging\\Projects\\holography_highorder\\data\\data_final_2d'
chan_str = ('red', 'green', 'blue')
loss = nn.MSELoss()

# ROI at target plane
roi_res = (960, 1680)
# Optional fixed crop for image on captured sensor
range_row = (0, 1200)
range_col = (0, 1920)

full_losses = []
red_losses = []
green_losses = []
blue_losses = []
full_outputs = []
exposure_list = [0.3, 0.5, 1.4]
# Loop through colors
for channel in range(3):
    print(f'Set the LUT, laser, and exposure for {chan_str[channel]}')
    input('Press enter to continue: ')
    
    # Initialize camera capture with homography pattern
    camera_prop = PhysicalProp(channel, laser_arduino=False, roi_res=(roi_res[1], roi_res[0]),
                            slm_settle_time=2*exposure_list[channel]+0.12, range_row=range_row, range_col=range_col,
                            patterns_path=homography_path, show_preview=False)

    # Loop through phase patterns, capture images, compute losses
    for phase_idx in range(len(phase_paths)):
        target_idx = target_idxs[phase_idx]
        phase_path = phase_paths[phase_idx]

        # Load target image and extract color
        loader = ImageLoader(data_path, channel=None, image_res=roi_res,
            idx_subset=[target_idx])
        loader.__iter__()
        target_amp, _ = loader.__next__()

        # Load phase pattern for specific distance
        slm_phase = imread(os.path.join(phase_path, f'{chan_str[channel]}.png'))
        slm_phase = (1 - slm_phase / np.iinfo(np.uint8).max) * 2 * np.pi - np.pi
        slm_phase = torch.tensor(slm_phase, dtype=torch.float32).reshape(1, *slm_phase.shape)

        captured_amp = camera_prop(slm_phase).to(target_amp.device)

        print("Percent greater than 0.95 of max")
        print(torch.sum((captured_amp>0.95*torch.max(captured_amp)).float())/(roi_res[0]*roi_res[1]))

        s = (captured_amp*target_amp[:,channel:channel+1,...]).sum().detach()/ \
            (captured_amp**2).sum().detach()

        loss_val = loss(s*captured_amp, target_amp[:,channel:channel+1,...]).detach().item()

        if target_idx == 20:
            s = 0.5*s

        if channel == 0:
            full_outputs.append(torch.clamp(s*captured_amp, 0, 1).detach())
            red_losses.append(loss_val)
            full_losses.append((1.0/3.0)*loss_val)
        elif channel == 1:
            full_outputs[phase_idx] = torch.cat([full_outputs[phase_idx],
                torch.clamp(s*captured_amp, 0, 1).detach()], 1)
            green_losses.append(loss_val)
            full_losses[phase_idx] = full_losses[phase_idx]+(1.0/3.0)*loss_val
        elif channel == 2:
            full_outputs[phase_idx] = torch.cat([full_outputs[phase_idx],
                torch.clamp(s*captured_amp, 0, 1).detach()], 1)
            blue_losses.append(loss_val)
            full_losses[phase_idx] = full_losses[phase_idx]+(1.0/3.0)*loss_val

# Loop through phase patterns and write out outputs
for phase_idx in range(len(phase_paths)):
    phase_path = phase_paths[phase_idx]
    full_output = full_outputs[phase_idx]
    
    full_output = full_output.cpu().detach().numpy()[0,...]
    full_output = np.moveaxis(full_output, 0, -1)
    print(np.shape(full_output))
    imwrite(os.path.join(phase_path, f'capture.png'), full_output)
    print(phase_path)
    print(f'{-10*np.log10(full_losses[phase_idx])}: (R: {-10*np.log10(red_losses[phase_idx])}, G: {-10*np.log10(green_losses[phase_idx])}, B: {-10*np.log10(blue_losses[phase_idx])})')

