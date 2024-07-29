
import os, sys
sys.path.append('../code')
import numpy as np
import torch
import torch.nn as nn
from imageio import imread
from utils.physical_prop_module import PhysicalProp
from utils.image_loader import ImageLoader


# Expected Files
# {phase_path}/{red/green/blue}/distX.png contains the  phase patterns for the castle image for each distance
# {data_path}/img_{target_idx}.png is the target image
# {homography_path}/{red/green/blue}.png is the homography phase pattern for an estimated distance

# Parameters
channel = 2 # 0: red, 1: green, 2: blue
chan_str = ('red', 'green', 'blue')
loss = nn.MSELoss()
phase_path = 'G:\\Shared drives\\Stanford Computational Imaging\\Projects\\holography_highorder\\data\\phase_patterns\\0721_distance_calibration\\phases_per_distance'
phase_path = os.path.join(phase_path, chan_str[channel])
homography_path = 'G:\\Shared drives\\Stanford Computational Imaging\\Projects\\holography_highorder\\data\\phase_patterns\\0721_distance_calibration\\homography_phases'
dist_array = ['33.8', '34.0', '34.2', '34.4', '34.6', '34.8', '35.0', '35.2', '35.4', '35.6', '35.8', '36.0', '36.2']

# ROI at target plane
roi_res = (960, 1680)

# Optional fixed crop for image on captured sensor
range_row = (0, 1200)
range_col = (0, 1920)

# Load target image and extract color
target_idx = 10 # Castle Pattern
data_path = 'C:\\Users\\EE267\\Desktop\\neural-holography\\data_final_2d'
loader = ImageLoader(data_path, channel=channel, image_res=roi_res,
    idx_subset=[target_idx])
loader.__iter__()
target_amp, _ = loader.__next__()

# Initialize camera capture with homography pattern
camera_prop = PhysicalProp(channel, laser_arduino=False, roi_res=(roi_res[1], roi_res[0]),
                           slm_settle_time=0.12, range_row=range_row, range_col=range_col,
                           patterns_path=homography_path, show_preview=False)

# Loop through phase patterns to display target image at different depths
min_loss = 100.0
best_dist = ''
for dist in dist_array:
    # Load phase pattern for specific distance
    slm_phase = imread(os.path.join(phase_path, f'{dist}.png'))
    slm_phase = (1 - slm_phase / 256.0) * 2 * np.pi - np.pi
    slm_phase = torch.tensor(slm_phase, dtype=torch.float32).reshape(1, *slm_phase.shape)

    captured_amp = camera_prop(slm_phase).to(target_amp.device)

    s = (captured_amp*target_amp).sum().detach()/ \
        (captured_amp**2).sum().detach()

    loss_val = loss(s*captured_amp, target_amp)
    
    print(f'{dist}: {loss_val}')
    if loss_val < min_loss:
        min_loss = loss_val
        best_dist = dist

print(f'{best_dist}: {loss_val}')
