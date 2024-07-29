
"""
This is the algorithm script used to compute phase patterns - DPAC, SGD, HOGD
"""
import sys
sys.path.append("../")
import torch, math
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torchvision.transforms import GaussianBlur
import cv2

import utils.utils as utils

# 1. DPAC
def double_phase_amplitude_coding(target_amps, target_phases, propagator,
    preblur_sigma=0.0):
    """ Use a single propagation and converts amplitude and phase to double
    phase coding
    Input
    -----
    :param target_amps: A tensor, (1,depths,*roi_res), the amplitude at the
        target planes.
    :param target_phases: The phases at the target image plane
    :param propagator: predefined instance of the Propagator module.
    :param preblur_sigma: preblur_sigma for AA-DPM from Shi et al 2021
    Output
    ------
    :return: a tensor, the optimized phase pattern at the SLM plane, in the 
        shape of (1,1,*slm_res).
    """
    # Create stack of wavefronts at target planes
    target_field = utils.polar_to_rect(target_amps, target_phases)

    # Propagate to and sum wavefronts at SLM plane
    slm_field = propagator(target_field, to_slm=True)

    # Perform blur for AA-DPM if needed
    if preblur_sigma != 0.0:
        blur = GaussianBlur(5, sigma=preblur_sigma)
        slm_field = torch.view_as_real(slm_field)
        slm_field[...,0] = blur(slm_field[...,0])
        slm_field[...,1] = blur(slm_field[...,1])
        slm_field = torch.view_as_complex(slm_field)

    # Convert to amplitude and phase representation
    amplitudes, phases = utils.rect_to_polar(slm_field)

    # normalize (with slight bounds to avoid nan with acos)
    amplitudes = torch.clip(amplitudes / (amplitudes.max()+1e-6), 1e-6,
        1.0-1e-6)

    # Perform double phase computation
    phases_a = phases - torch.acos(amplitudes)
    phases_b = phases + torch.acos(amplitudes)
    phases_out = phases_a
    phases_out[..., ::2, 1::2] = phases_b[..., ::2, 1::2]
    phases_out[..., 1::2, ::2] = phases_b[..., 1::2, ::2]

    # Center phase range around 0 before wrapping
    max_phase = 2 * math.pi
    phases_out = phases_out - 0.5*(phases_out.max() + phases_out.min())
    while phases_out.max() > max_phase / 2:
        phases_out = torch.where(phases_out > max_phase / 2,
            phases_out - 2.0 * math.pi, phases_out)
    while phases_out.min() < -max_phase / 2:
        phases_out = torch.where(phases_out < -max_phase / 2,
            phases_out + 2.0 * math.pi, phases_out)\

    return phases_out


# 2. SGD
def stochastic_gradient_descent(init_phase, target_amp, masks, num_iters,
    phase_path, propagator, tb_prefix='', loss=nn.MSELoss(), lr=0.01,
    lr_s=0.003, s0=1.0, full_bandwidth=False, camera_prop=None, writer=None):
    """ Given the initial guess, run the SGD algorithm to calculate the
    optimal phase pattern of spatial light modulator.
    Input
    ------
    :param init_phase: initial guess for the SLM phase.
    :param target_amp: the amplitude at the target planes.
    :param masks: the decomposition of target amp across target planes,
        None for all-in-focus content
    :param num_iters: the number of iterations
    :param phase_path: path to write out intermediate phases
    :param propagator: propagator instance (function / pytorch module)
    :param roi_res: region of interest to penalize the loss
    :param tb_prefix: a string, for identifying SGD run in tensorboard
    :param loss: loss function, default L2
    :param lr: learning rate for phase variables
    :param lr_s: learning rate for scale factor, set to 0.0 for optimal scale
    :param s0: initialization for scale factor
    :param camera_prop: optional physical propagation forward model for CITL
    :param writer: SummaryWriter instance for tensorboard
    Output
    ------
    :return: a tensor, the optimized phase pattern at the SLM plane, in the 
        shape of (1,1,*slm_res).
    """
    # phase at the slm plane
    slm_phase = init_phase.requires_grad_(True)

    # optimization variables and adam optimizer
    s = torch.tensor(s0, requires_grad=True, device=init_phase.device)
    optvars = [{'params': slm_phase}]
    if lr_s > 0.0:
        optvars += [{'params': s, 'lr': lr_s}]
    optimizer = optim.Adam(optvars, lr=lr)

    # run the iterative algorithm
    for k in range(num_iters):
        if camera_prop is not None:
            print(k)
            
        optimizer.zero_grad()

        # forward propagation from the SLM plane to the target plane
        slm_field = utils.polar_to_rect(torch.ones_like(slm_phase), slm_phase)
        recon_field = propagator(slm_field)
        if full_bandwidth:
            recon_amp = utils.full_bandwidth_amplitude(recon_field)
            recon_amp = utils.intensity_blur(recon_amp, 2)
            recon_amp = recon_amp[...,1::2,1::2]
        else:
            recon_amp = recon_field.abs()

        if camera_prop is not None:
            captured_amp = camera_prop(slm_phase)

            # use the gradient of proxy, replacing the amplitudes
            # captured_amp is assumed that its size already matches that of
            # recon_amp
            out_amp = recon_amp + (captured_amp - recon_amp).detach()
        else:
            out_amp = recon_amp

        if lr_s == 0.0:
            # Determine optimal scale factor
            with torch.no_grad():
                if masks is None:
                    s = (out_amp*target_amp).sum().detach()/ \
                        (out_amp**2).sum().detach()
                else:
                    s = ((out_amp*masks)*(target_amp*masks)).sum().detach()/ \
                        ((out_amp*masks)**2).sum().detach()
        out_amp = s*out_amp

        # calculate loss and backprop
        lossValue = utils.memory_efficient_loss(target_amp, out_amp, loss,
            masks=masks, reduction_factor=1)
        lossValue.backward()
        optimizer.step()

        # write to tensorboard / save intermediate phases
        with torch.no_grad():
            if writer is not None:
                if k % 50 == 0:
                    writer.add_scalar(f'loss/{tb_prefix}', lossValue, k)
                    writer.add_scalar(f'scalar/{tb_prefix}', s, k)
                if k % 250 == 0:
                    phase_out_8bit = utils.phasemap_8bit(slm_phase,
                        inverted=True)
                    cv2.imwrite(f'{phase_path}_{k + 1}.png', phase_out_8bit)

    if camera_prop is not None:
        return lossValue.item(), torch.clamp(out_amp, 0, 1), slm_phase
    return slm_phase

# 3. HOGD
def higher_order_gradient_descent(init_phase, target_amp, masks, num_iters,
    phase_path, propagator, tb_prefix='', loss=nn.MSELoss(), lr=0.01,
    lr_s=0.003, s0=1.0, camera_prop=None, citl_y_offset=0, citl_x_offset=0,
    writer=None, supervision_box_blur=6, downsampled_loss=False,
    periphery_box_blur=6, foveation_mask=None):
    """Given the initial guess, run the SGD algorithm to calculate the optimal
    phase pattern of spatial light modulator.
    Input
    ------
    :param init_phase: initial guess for the SLM phase.
    :param target_amp: the amplitude at the target planes.
    :param masks: the decomposition of target amp across target planes,
        None for all-in-focus content
    :param num_iters: the number of iterations
    :param phase_path: path to write out intermediate phases
    :param propagator: propagator instance (function / pytorch module)
    :param roi_res: region of interest to penalize the loss
    :param tb_prefix: a string, for identifying SGD run in tensorboard
    :param loss: loss function, default L2
    :param lr: learning rate for phase variables
    :param lr_s: learning rate for scale factor, set to 0.0 for optimal scale
    :param s0: initialization for scale factor
    :param camera_prop: optional physical propagation forward model for CITL
    :param citl_y_offset: y sampling offset for CITL in range of [0, 6)
    :param citl_x_offset: x sampling offset for CITL in range of [0, 6)
    :param writer: SummaryWriter instance for tensorboard
    :param supervision_box_blur: blur of target supervision
    :param downsampled_loss: If true, downsample blurred target and recon
        before computing loss
    :param periphery_box_blur: If set, blur of periphery supervision.
    :param foveation_mask: mask to apply foveated target resolution
    Output
    ------
    :return: a tensor, the optimized phase pattern at the SLM plane, in the 
        shape of (1,1,*slm_res).
    """
    # phase at the slm plane
    slm_phase = init_phase.requires_grad_(True)

    # optimization variables and adam optimizer
    s = torch.tensor(s0, requires_grad=True, device=init_phase.device)
    optvars = [{'params': slm_phase}]
    if lr_s > 0.0:
        optvars += [{'params': s, 'lr': lr_s}]
    optimizer = optim.Adam(optvars, lr=lr)

    # Setup foveated terms
    if foveation_mask is not None:
        # Blur target intensity to the desired periphery supervision level 
        periphery_target_amp = utils.intensity_blur(target_amp,
            periphery_box_blur)
        if masks is None:
            periphery_mask = (1-foveation_mask)
        else:
            periphery_mask = (1-foveation_mask)*masks
            foveation_mask = foveation_mask*masks

    if camera_prop is None:
        # Blur high res target intensity to the desired supervision level
        # Not needed for CITL where the target is lower res
        target_amp = utils.intensity_blur(target_amp, supervision_box_blur)

    NaN_found = False

    # run the iterative algorithm
    for k in range(num_iters):
        if camera_prop is not None:
            print(k)
            
        optimizer.zero_grad()

        # forward propagation from the SLM plane to the target plane
        slm_field = utils.polar_to_rect(torch.ones_like(slm_phase), slm_phase)
        recon_field = propagator(slm_field)
        recon_amp = utils.full_bandwidth_amplitude(recon_field)
        
        # Blur recon intensity to the desired supervision level
        blurred_recon = utils.intensity_blur(recon_amp, supervision_box_blur)

        if camera_prop is not None:
            captured_amp = camera_prop(slm_phase)

            # use the gradient of proxy, replacing the amplitudes
            # captured_amp is assumed that its size matches a downsampled
            # version of blurred_recon
            blurred_recon = blurred_recon[...,citl_y_offset::6,
                citl_x_offset::6]
            out_amp = blurred_recon + (captured_amp - blurred_recon).detach()
        else:
            out_amp = blurred_recon

        if lr_s == 0.0:
            # Determine optimal scale factor
            with torch.no_grad():
                if masks is None:
                    s = (out_amp*target_amp).sum().detach()/ \
                        (out_amp**2).sum().detach()
                else:
                    s = ((out_amp*masks)*(target_amp*masks)).sum().detach()/ \
                        ((out_amp*masks)**2).sum().detach()
        out_amp = s*out_amp

        # calculate loss and backprop
        if camera_prop is not None:
            lossValue = utils.memory_efficient_loss(target_amp, out_amp,
                loss, masks=masks, reduction_factor=1,
                downsampled_loss=downsampled_loss)
        elif foveation_mask is None:
            lossValue = utils.memory_efficient_loss(target_amp, out_amp,
                loss, masks=masks, reduction_factor=6,
                downsampled_loss=downsampled_loss)
        else:
            loss_foveated = utils.memory_efficient_loss(target_amp,
                out_amp, loss, masks=foveation_mask, reduction_factor=6,
                downsampled_loss=downsampled_loss)

            # Blur recon intensity to the desired periphery supervision level
            out_amp = utils.intensity_blur(recon_amp, periphery_box_blur)
            out_amp = s*out_amp
            loss_periphery = utils.memory_efficient_loss(periphery_target_amp,
                out_amp, loss, masks=periphery_mask, reduction_factor=6,
                downsampled_loss=downsampled_loss)
            
            lossValue = loss_foveated + loss_periphery

        lossValue.backward()
        optimizer.step()

        # write to tensorboard / save intermediate phases
        with torch.no_grad():
            if writer is not None:
                if k % 50 == 0:
                    writer.add_scalar(f'loss/{tb_prefix}', lossValue, k)
                    writer.add_scalar(f'scalar/{tb_prefix}', s, k)
                    if foveation_mask is not None:
                        writer.add_scalar(f'loss_foveated/{tb_prefix}',
                            loss_foveated, k)
                        writer.add_scalar(f'loss_periphery/{tb_prefix}',
                            loss_periphery, k)
                if k % 250 == 0:
                    phase_out_8bit = utils.phasemap_8bit(slm_phase,
                        inverted=True)
                    cv2.imwrite(f'{phase_path}_{k + 1}.png', phase_out_8bit)
                    
    if camera_prop is not None:
        print(f'loss {lossValue.item()}, PSNR {-10*np.log10(lossValue.item())}')
        return lossValue.item(), torch.clamp(out_amp, 0, 1), slm_phase
    return slm_phase
