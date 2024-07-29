import math
import numpy as np
from typing import Tuple
import torch
import torch.nn as nn
import os
from torchvision.transforms.functional import resize


def burst_img_processor(img_burst_list):
    img_tensor = np.stack(img_burst_list, axis=0)
    img_avg = np.mean(img_tensor, axis=0)
    return im2float(img_avg)  # changed from int8 to float32

def str2bool(v):
    """ Simple query parser for configArgParse
    (which doesn't support native bool from cmd)
    Ref: https://stackoverflow.com/questions/15008758/parsing-
        boolean-values-with-argparse

    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError('Boolean value expected.')

# @torch.jit.script
def rect_to_polar(rect):
    """Converts a field from complex values to polar form
    Input
    -----
    :return rect: complex values of field
    Output
    -----
    :param mag: field magnitude
    :param ang: field angle
    """
    rect = torch.view_as_real(rect)
    mag = torch.pow(rect[..., 0] ** 2 + rect[..., 1] ** 2, 0.5)
    ang = torch.atan2(rect[..., 1], rect[..., 0])
    return mag, ang


# @torch.jit.script
def polar_to_rect(mag, ang):
    """Converts a field from polar form to complex values
    Input
    -----
    :param mag: field magnitude
    :param ang: field angle
    Output
    -----
    :return rect: complex values of field
    """
    real = mag * torch.cos(ang)
    imag = mag * torch.sin(ang)
    rect = torch.view_as_complex(torch.stack((real, imag), -1))
    return rect

# @torch.jit.script
def ifftshift(tensor):
    """Performs Matlab ifftshift along 2nd and 3rd dimensions
    Input
    -----
    :param tensor: tensor to be ifftshifted
    Output
    -----
    :return out: ifftshifted tensor
    """
    size = tensor.size()
    tensor_shifted = circshift(tensor, -math.floor(size[2] / 2.0), 2)
    out = circshift(tensor_shifted, -math.floor(size[3] / 2.0), 3)
    return out


# @torch.jit.script
def fftshift(tensor):
    """Performs Matlab fftshift along 2nd and 3rd dimensions
    Input
    -----
    :param tensor: tensor to be fftshifted
    Output
    -----
    :return out: fftshifted tensor
    """
    size = tensor.size()
    tensor_shifted = circshift(tensor, math.floor(size[2] / 2.0), 2)
    out = circshift(tensor_shifted, math.floor(size[3] / 2.0), 3)
    return out

# @torch.jit.script
def circshift(tensor, shift:int, axis:int):
    """Circle shifts a tensor by a given shift along a given dimensions
    Input
    -----
    :param tensor: tensor to be shifted
    :param shift: amount of shift
    :param axis: axis to be shifted
    Output
    -----
    :return out: circle shifted tensor
    """
    if shift == 0:
        return tensor

    if axis < 0:
        axis += tensor.dim()

    dim_size = tensor.size(axis)
    after_start = dim_size - shift
    if shift < 0:
        after_start = -shift
        shift = dim_size - abs(shift)

    before = tensor.narrow(axis, 0, dim_size - shift)
    after = tensor.narrow(axis, after_start, shift)
    out = torch.cat([after, before], axis)

    return out

# @torch.jit.script
def pad_image(field, target_shape:Tuple[int, int]):
    """Pads an input field to a desired target shape
    Input
    -----
    :param field: input field to be padded
    :param target_shape: target_shape for output
    Output
    -----
    :return out: padded output field
    """
    size_diff = torch.tensor(target_shape) - torch.tensor(field.shape[-2:])
    odd_dim = torch.tensor(field.shape[-2:]) % 2

    # pad the dimensions that need to increase in size
    if (size_diff > 0).any():
        pad_total = torch.where(size_diff > 0, size_diff, 0)
        pad_front = (pad_total + odd_dim) // 2
        pad_end = (pad_total + 1 - odd_dim) // 2

        # Fill list with desired padding levels for each axis
        pad_axes = [int(pad_front[1].item()), int(pad_end[1].item())]
        pad_axes += [int(pad_front[0].item()), int(pad_end[0].item())]

        out = nn.functional.pad(field, pad_axes, value=0.0)
    else:
        out = field

    return out

# @torch.jit.script
def crop_image(field, target_shape:Tuple[int, int]):
    """Crops an input field to a desired target shape
    Input
    -----
    :param field: input field to be cropped
    :param target_shape: target_shape for output
    Output
    -----
    :return out: cropped output field
    """
    size_diff = torch.tensor(field.shape[-2:]) - torch.tensor(target_shape)
    odd_dim = torch.tensor(field.shape[-2:]) % 2

    # crop dimensions that need to decrease in size
    if (size_diff > 0).any():
        crop_total = torch.where(size_diff > 0, size_diff, 0)
        crop_front = (crop_total + 1 - odd_dim) // 2
        crop_end = (crop_total + odd_dim) // 2

        # Amount of crop from front of dimensions
        f0 = int(crop_front[0].item())
        f1 = int(crop_front[1].item())

        # Amount of crop from end of dimensions
        e0 = int(crop_end[0].item())
        e1 = int(crop_end[1].item())

        # Crop first dimension
        if e0:
            out = field[..., f0:-e0, :]
        else:
            out = field[..., f0:, :]

        # Crop second dimension
        if e1:
            out = out[..., f1:-e1]
        else:
            out = out[..., f1:]
    else:
        out = field

    return out

def cond_mkdir(path):
    """Create a desired folder if it does not already exist
    Input
    -----
    :param patch: folder to be created if it does not exist
    """
    if not os.path.exists(path):
        os.makedirs(path)


def phasemap_8bit(phasemap, inverted=True):
    """Convert a phasemap tensor into a numpy 8bit phasemap for SLM
    Input
    -----
    :param phasemap: input phasemap tensor, in the range of [-pi, pi].
    :param inverted: flag for if the phasemap is inverted.
    Output
    ------
    :return phase_out_8bit: output phasemap, with uint8 dtype (in [0, 256))
    """

    out_phase = phasemap.cpu().detach().squeeze().numpy()
    out_phase = ((out_phase + np.pi) % (2 * np.pi)) / (2 * np.pi)
    if inverted:
        phase_out_8bit = ((1 - out_phase) * 256).round().astype(np.uint8) # change from 255 to 256
    else:
        phase_out_8bit = ((out_phase) * 256).round().astype(np.uint8)

    return phase_out_8bit


def im2float(im, dtype=np.float32):
    """Convert uint16 or uint8 image to float32, with range scaled to 0-1
    Input
    -----
    :param im: image
    :param dtype: default np.float32
    Output
    ------
    :return image as float with range of 0-1
    """
    if issubclass(im.dtype.type, np.floating):
        return im.astype(dtype)
    elif issubclass(im.dtype.type, np.integer):
        return im / dtype(np.iinfo(im.dtype).max)
    else:
        raise ValueError(f'Unsupported data type {im.dtype}')



def foveation_masking(roi_res, foveation_center_x, foveation_center_y, 
    foveation_radius):
    """Function to generate foveation mask for target
    Input
    -----
    :param roi_res: resolution of target
    :param foveation_center_x: Normalized x coordinate of foveation center
    :param foveation_center_y: Normalized y coordinate of foveation center
    :param foveation_radius: Normalized radius of foveation region
    Output
    ------
    :return foveation_mask: foveation mask for target
    """
    # Foveated region mask at resolution produced by higher order propagation
    y = torch.linspace(0.0, 1.0, 6*roi_res[0])
    x = torch.linspace(0.0, 1.0, 6*roi_res[1])
    X, Y = torch.meshgrid(x, y)
    X = torch.transpose(X, 0, 1)
    Y = torch.transpose(Y, 0, 1)
    foveation_mask = 1.0*(((X-foveation_center_x)*roi_res[1]/roi_res[0])**2+(Y-foveation_center_y)**2
        <= foveation_radius**2)
    foveation_mask = torch.reshape(foveation_mask, (1, 1,
        foveation_mask.size()[0], foveation_mask.size()[1]))
    
    return foveation_mask

# @torch.jit.script
def full_bandwidth_amplitude(field):
    """Function to perform sinc interpolation before taking the absolute value
    to avoid aliasing as discussed in Chen et al 2021.
    Input
    -----
    :param field: field to extract amplitude from

    Output
    ------
    :return upsampled_amp: amplitude without aliasing
    """
    frequency_domain = fftshift(torch.fft.fftn(ifftshift(field), dim=(-2, -1),
        norm='ortho'))
    frequency_domain = pad_image(frequency_domain, (2*field.shape[-2],
        2*field.shape[-1]))
    upsampled_amp = fftshift(torch.fft.ifftn(ifftshift(frequency_domain),
        dim=(-2, -1), norm='ortho')).abs()
    return upsampled_amp

# @torch.jit.script
def memory_efficient_loss(target, recon, loss, masks=None,
    reduction_factor=2, downsampled_loss=False):
    """Function to perform memory efficient loss by division into multiple
    computations.
    Input
    -----
    :param target: target for loss
    :param recon: recon for loss
    :param masks: masks for loss, use None for all-in-focus content
    :param reduction factor: factor to reduce outputs by for memory efficiency
    :param downsampled_loss: If true, downsample blurred target and recon
        before computing loss

    Output
    ------
    :return loss_val: computed loss value
    """
    loss_val = 0
    subpixels = [0] if downsampled_loss else range(reduction_factor)
    for i in subpixels:
        for j in subpixels:
            target_sub = target[...,i::reduction_factor,j::reduction_factor]
            recon_sub = recon[...,i::reduction_factor,j::reduction_factor]
            if masks is None:
                loss_val += loss(target_sub, recon_sub)
            else:
                # mask output to only get gradients at the in-focus content
                masks_sub = masks[...,i::reduction_factor,j::reduction_factor]
                nonzero = masks_sub > 0
                out_amp = torch.zeros_like(recon_sub)
                out_amp[nonzero] = recon_sub[nonzero] * masks_sub[nonzero]

                loss_val += loss(out_amp[nonzero],
                    (target_sub*masks_sub)[nonzero])
    loss_val *= (1.0/(float(len(subpixels))**2))

    return loss_val

# @torch.jit.script
def intensity_blur(amplitude, blur):
    """Blurs intensity of specified amplitude
    Input
    -----
    :param amplitude: amplitude 
    :param blur: blur level indicating desired loss in resolution 

    Output
    ------
    :return amplitude_out: amplitude of blurred intensity
    """
    intensity = amplitude**0.5

    # Construct Box Blur kernel (Other kernels can be explored)
    channel_count = amplitude.shape[1]
    kernel = torch.ones(1,1,blur, blur)
    kernel = kernel / kernel.sum()
    kernel = kernel.repeat(channel_count, 1, 1, 1)
    pad_count = int(blur//2)
    filter = nn.Conv2d(in_channels=channel_count,
        out_channels=channel_count, kernel_size=blur,
        padding=(pad_count, pad_count), padding_mode='reflect',
        groups=channel_count, bias=False).to(amplitude.device)
    filter.weight.data = kernel.to(amplitude.device)
    filter.weight.requires_grad = False

    intensity_out = filter(intensity)
    amplitude_out = intensity_out**2.0

    if blur%2 == 0:
        # Drop last row/col to get original image size
        amplitude_out = amplitude_out[...,:-1,:-1]

    return amplitude_out


def evaluate_phase(slm_phases, scene_amp, prop_dists, pitch, wavelengths,
    roi_res, slm_res, order_nums, filters, pupils=None, masks_c=None, loss=nn.MSELoss(),
    full_bandwidth=False, pad_multiplier=1.5, supervision_box_blur=6,
    writer=None, tb_idx=0, prop_visualizations=False, eyepiece=0.05):
    """Evaluates slm phase patterns with specified filtering and simulated
    orders, writes loss, PSNR, reconstructed amplitudes, blurred amplitudes,
    and sampled amplitudes to tensorboard
    Input
    -----
    :param slm_phases: list of SLM phase patterns for each color
    :param scene_amp: target scene
    :param prop_dists: prop dists between SLM and target planes, in meters
    :param pitch: the SLM pixel pitch, in meters
    :param wavelengths: list of wavelengths of interest, in meters
    :param slm_phases: list of slm_phases for different colors
    :param roi_res: Size of image reconstruction, in units of pitch size
    :param slm_res: resolution of SLM
    :param order_nums: list of orders to simulate
    :param filters: list of propagation filters to simulate
    :param masks_c: masks_c for in focus content if multiple target planes
    :param loss: loss to compute (Should be l2 for PSNR)
    :param pad_multiplier: Amount of padding as multiple of FOV kernel
    :param supervision_box_blur: level of blurring for supervision
    :param writer: writer to write propagation visualizations to
    :param tb_prefix: a string, for identifying visualizations in tensorboard
    :param tb_idx: image idx, for identifying visualizations in tensorboard
    :param prop_visualizations: Flag to just generate propagation
        visualizations
    """
    import propagation_ASM
    # Switch colors to dim 0 to use dim 1 for depth plane
    scene_amp = scene_amp[0,...].unsqueeze(1)
    mse_list = []
    for j in range(len(order_nums)):
        lossVal = 0
        # Determine level of blur for supervision based on orders simulated
        if order_nums[j] == 1:
            reduction_factor = 2 if full_bandwidth else 1
        elif order_nums[j] == 3:
            reduction_factor = 6
        elif order_nums[j] == 5:
            reduction_factor = 10
        for c in range(len(wavelengths)):
            # Set tensorboard identifier for visualizations
            chan_str = ('red', 'green', 'blue')[c]
            tb_prefix = f'{chan_str}_order{order_nums[j]}' \
                + f'_filtering{filters[j]}_pupil{(pupils[j] if pupils is not None else None)}'
            print(f'Processing visualizations for {tb_prefix}')

            # Compute target field
            torch.cuda.empty_cache()
            slm_phase = slm_phases[c]
            slm_field = polar_to_rect(torch.ones_like(slm_phase), slm_phase)
            propagator = propagation_ASM.Propagator([prop_dists[c]], pitch, wavelengths[c],
                pad_multiplier=pad_multiplier, slm_res=slm_res,
                target_res=roi_res, order_num=order_nums[j],
                filter=filters[j],
                pupil=(pupils[j] if pupils is not None else None),
                eyepiece=eyepiece, device=slm_phase.device)
            if prop_visualizations:
                recon_field = propagator(slm_field, writer=writer,
                    tb_prefix=tb_prefix, tb_idx=tb_idx)
            else: 
                recon_field = propagator(slm_field)
                propagator = None
                if (order_nums[j] == 1) and not full_bandwidth:
                    recon_amp = recon_field.abs()
                else:
                    recon_amp = full_bandwidth_amplitude(recon_field)

                # Blur and sample intensities at the desired supervision level
                blurred_amp = intensity_blur(recon_amp, reduction_factor)
                blurred_amp = blurred_amp[...,
                    int(reduction_factor//2)::reduction_factor,
                    int(reduction_factor//2)::reduction_factor]

                # Compute per-channel scale factor
                if masks_c is None:
                    masks = None
                    s = (blurred_amp*scene_amp[c:c+1,...]).sum().detach()/ \
                        (blurred_amp**2).sum().detach()
                else:
                    # Resize masks for color
                    s = ((blurred_amp*masks)*(scene_amp[c:c+1,...]*masks)
                        ).sum().detach()/((blurred_amp*masks)**2).sum().detach()
                blurred_amp = s*blurred_amp
                recon_amp = s*recon_amp
                
                # Store per-channel outputs
                if c == 0:
                    recon_amp_c = recon_amp.cpu().detach()
                    blurred_amp_c = blurred_amp.cpu().detach()
                else:
                    recon_amp_c = torch.cat([recon_amp_c,
                        recon_amp.cpu().detach()], 0)
                    blurred_amp_c = torch.cat([blurred_amp_c,
                        blurred_amp.cpu().detach()], 0)

                # Compute loss per channel
                lossVal += memory_efficient_loss(scene_amp[c:c+1,...], blurred_amp, loss,
                    masks=masks, reduction_factor=1,
                    downsampled_loss=True).item()*(1/float(len(wavelengths)))
        
        if not prop_visualizations:
            # Write to tensorboard
            writer.add_scalar(
                f'loss_simulation_order{order_nums[j]}_filtering{filters[j]}_pupil{(pupils[j] if pupils is not None else None)}',
                lossVal, tb_idx)
            mse_list.append(lossVal)
            
            writer.add_scalar(
                f'psnr_simulation_order{order_nums[j]}_filtering{filters[j]}_pupil{(pupils[j] if pupils is not None else None)}',
                -10.0*np.log10(lossVal), tb_idx)

            writer.add_image(
                f'recon_amp_{tb_prefix}',
                torch.clamp(recon_amp_c, 0, 1)[:,0,...], tb_idx)
            writer.add_image(
                f'sampled_amp_{tb_prefix}',
                torch.clamp(blurred_amp_c, 0, 1)[:,0,...], tb_idx)

    return mse_list

