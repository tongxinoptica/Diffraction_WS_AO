"""
This is the script that is used for the wave propagation using the angular
spectrum method (ASM). Refer to Goodman, Joseph W. Introduction to Fourier
optics. Roberts and Company Publishers, 2005, for principle details.

This code and data is released under the Creative Commons
Attribution-NonCommercial 4.0 International license (CC BY-NC.) In a nutshell:
    # The license is only for non-commercial use (commercial licenses can be
        obtained from Stanford).
    # The material is provided as-is, with no warranties whatsoever.
    # If you publish any code, data, or scientific work based on this, please
       cite our work.

"""

import math, torch
import torch.fft
import torch.nn as nn

import utils.utils as utils
        
class Propagator(nn.Module):
    """Wrapper to propagate field from SLM plane to multiple target planes
        Can also backpropagate the field from target planes to SLM

    Class initialization parameters
    -------------------------------
    :param prop_dists: propagation dists between SLM and target planes, meters
    :param pitch: the SLM pixel pitch, meters
    :param wavelength: the wavelength of interest, meters
    :param pad_multiplier: Amount of padding as multiple of max kernel
        FOV size (set to -1 for double padding)
    :param slm_res: resolution at the SLM plane (sampled at SLM pitch)
    :param target_res: resolution at the target planes (sampled at SLM pitch)
    :param order_num: number of orders to simulate (1 DPAC, 1 SGD, 3 HOGD)
    :param filter: fourier filtering level (0.5 DPAC, 1.0 SGD, 3.0 HOGD)
    :param device: torch.device
    """
    def __init__(self, prop_dists, pitch, wavelength, pad_multiplier=1.5,
        slm_res=(1080,1920), target_res=(1080,1920), order_num:int=3,
        filter=1.0, pupil=None, eyepiece=0.050, device=torch.device('cuda')):
        super(Propagator, self).__init__()

        # Store parameters
        self.pitch = pitch
        self.wavelength = wavelength
        self.dev = device
        self.slm_res = slm_res
        self.target_res = target_res
        self.prop_res = (max(slm_res[0],target_res[0]),
            max(slm_res[1],target_res[1]))
        self.filter = filter
        self.order_num = order_num
        self.prop_dists = prop_dists
        self.pupil = pupil
        self.eyepiece = eyepiece

        # Determine padding for linear conv
        self.pad_multiplier = pad_multiplier
        if self.pad_multiplier != -1:
            self.kernel_size = 2*int(math.ceil(pad_multiplier*max(map(abs,
                self.prop_dists))/self.pitch*math.tan(math.asin(
                    self.wavelength/(2.0*self.pitch)))))
        else:
            self.kernel_size = -1

        # Compute and store propagation kernels
        precomped_Hs = []
        for prop_dist in self.prop_dists:
            precomped_H = propagation_ASM(
                torch.empty(1, 1, *self.prop_res, device=self.dev),
                self.pitch, self.wavelength, prop_dist, return_H=True,
                kernel_size=self.kernel_size, filter=self.filter,
                order_num=self.order_num, pupil=self.pupil, eyepiece=0.050)
            precomped_H = precomped_H.detach()
            precomped_H.requires_grad = False
            precomped_Hs.append(precomped_H)
        # stack them in the channel dimension
        self.precomped_H = torch.cat(precomped_Hs, 1)
        
    def forward(self, input_field, plane_idxs=None, to_slm=False,
        writer=None, tb_prefix='', tb_idx=0):
        """Propagates the input field using the angular spectrum method 
        Input
        -----
        :param input_field: field at target if to_slm, else field at SLM
        :param plane_idxs: target plane indexes, if none use all planes
        :param to_slm: Flag to set propagation direction to SLM
        :param writer: writer to write visualizations to
        :param tb_prefix: a string, for identifying visualizations in
            tensorboard
        :param tb_idx: image idx, for identifying visualizations in
            tensorboard
        Output
        -----
        :return output_field: field at SLM if to_slm, else field at target
        """
        # Extract kernels to be used based on plane_idxs
        if plane_idxs is not None:
            precomped_H = self.precomped_H[:, plane_idxs, ...]
        else:
            precomped_H = self.precomped_H

        input_field = utils.pad_image(input_field, self.prop_res)
        if to_slm:
            # Backpropagate wavefronts to SLM and sum at SLM
            output_field = torch.conj(propagation_ASM(torch.conj(input_field),
                self.pitch, self.wavelength, self.prop_dists[0],
                precomped_H=precomped_H, kernel_size=self.kernel_size,
                summed_output=True, filter=self.filter,
                order_num=self.order_num, writer=writer, tb_prefix=tb_prefix,
                tb_idx=tb_idx))
            output_res = [self.order_num*i for i in self.slm_res]
        else:
            # Propagate wavefront to target plane(s)
            output_field = propagation_ASM(input_field,
                self.pitch, self.wavelength, self.prop_dists[0],
                precomped_H=precomped_H, kernel_size=self.kernel_size,
                summed_output=False, filter=self.filter,
                order_num=self.order_num, writer=writer, tb_prefix=tb_prefix,
                tb_idx=tb_idx)
            output_res = [self.order_num*i for i in self.target_res]
        
        output_field = utils.crop_image(output_field, output_res)
        return output_field

    def to(self, *args, **kwargs):
        slf = super().to(*args, **kwargs)
        slf.prop_dists = slf.prop_dists.to(*args, **kwargs)
        if slf.precomped_H is not None:
            slf.precomped_H = slf.precomped_H.to(*args, **kwargs)
        # try setting dev based on some parameter, default to cpu
        try:
            slf.dev = next(slf.parameters()).device
        except StopIteration:  # no parameters
            device_arg = torch._C._nn._parse_to(*args, **kwargs)[0]
            if device_arg is not None:
                slf.dev = device_arg
        return slf

# @torch.jit.script
def propagation_ASM(u_in, pitch:float, wavelength:float, z:float,
    return_H:bool=False, precomped_H=torch.Tensor([]), kernel_size:int=-1,
    summed_output:bool=False, filter:float=1.0, order_num:int=3, 
    pupil=None, eyepiece=0.050, writer=None, tb_prefix='', tb_idx=0):
    """Propagates the input field using the angular spectrum method 
    Input
    -----
    :param u_in: the input field
    :param pitch: the SLM pixel pitch, in meters
    :param wavelength: the wavelength of interest, in meters
    :param z: propagation dist between SLM and target plane, in meters
    :param return_H: flag to return pre-computed kernels
    :param precomped_H: pre-computed propagation kernels
    :param kernel_size: custom kernel size (Set to -1 for double padding)
    :param summed_output: flag to sum output channels in frequency domain
    :param filter: fourier filtering level (1 DPAC, 1 SGD, 3 HOGD)
    :param order_num: number of orders to simulate
    :param writer: writer to write visualizations to
    :param tb_prefix: a string, for identifying visualizations in tensorboard
    :param tb_idx: image idx, for identifying visualizations in tensorboard
    Output
    -----
    :return u_out: the output field
    """
    # Pad for linear conv.
    input_resolution = u_in.size()[-2:]
    if kernel_size == -1:
        conv_size = [4*input_resolution[0], 3*input_resolution[1]] # 4, 3 for CITL, 6, 4 for SGD
    else:
        conv_size = [i + kernel_size for i in input_resolution]
    u_in = utils.pad_image(u_in, conv_size)

    # Output resolution scaled by number of orders and
    output_resolution = [order_num*i for i in input_resolution]

    # Compute frequency representation and tile to produce orders
    U1 = torch.fft.fftn(utils.ifftshift(u_in), dim=(-2, -1), norm='ortho')
    U1 = utils.fftshift(U1)
    if writer is not None:
        writer.add_image(f'slm_freq_amp/{tb_prefix}',
            torch.clamp(U1.abs()[0,...], 0, 1), tb_idx)
        writer.add_image(f'slm_freq_phase/{tb_prefix}',
            U1.angle()[0,...]/(2*math.pi)+0.5, tb_idx)
    U1 = U1.repeat(1, 1, order_num, order_num)
    # if writer is not None:
    #     writer.add_image(f'tiled_slm_freq_amp/{tb_prefix}',
    #         torch.clamp(U1.abs()[0,...], 0, 1), tb_idx)
    #     writer.add_image(f'tiled_slm_freq_phase/{tb_prefix}',
    #         U1.angle()[0,...]/(2*math.pi)+0.5, tb_idx)
    U1 = utils.ifftshift(U1)
        
    if not precomped_H.numel():
        # resolution of frequency domain with all orders
        field_resolution = U1.size()

        # number of pixels
        num_y, num_x = field_resolution[-2], field_resolution[-1]

        # sampling interval size
        dy = pitch/float(order_num)
        dx = pitch/float(order_num)

        # size of the field
        y, x = (dy * float(num_y), dx * float(num_x))

        # frequency coordinates
        fy = torch.linspace(-1/(2*dy)+0.5/(2*y),1/(2*dy)-0.5/(2*y),
            num_y, device=u_in.device)
        fx = torch.linspace(-1/(2*dx)+0.5/(2*x),1/(2*dx)-0.5/(2*x),
            num_x, device=u_in.device)
        FX, FY = torch.meshgrid(fx, fy)
        FX = torch.transpose(FX, 0, 1)
        FY = torch.transpose(FY, 0, 1)

        # Phase Delay of ASM Propagation
        H_exp = 2*math.pi*torch.sqrt(1/wavelength**2-(FX**2+FY**2))
        H_exp = torch.reshape(H_exp, (1, 1, H_exp.size()[0], H_exp.size()[1]))
        H_exp = torch.mul(H_exp, z)

        # Limit for frequencies which would be aliased 
        # (band-limited ASM - Matsushima et al. (2009))
        fy_max = 1 / math.sqrt((2 * abs(z) * (1 / y))**2 + 1) / wavelength
        fx_max = 1 / math.sqrt((2 * abs(z) * (1 / x))**2 + 1) / wavelength
        
        # Zero out filtered frequencies and aliased frequencies
        H_amp = ((float(order_num)*torch.abs(FX)<=filter*torch.abs(FX).max())
            & (float(order_num)*torch.abs(FY)<=filter*torch.abs(FY).max())
            & (torch.abs(FX)<fx_max) & (torch.abs(FY)<fy_max))     

        if pupil is not None:
            H_pupil = (torch.sqrt(torch.abs(FX**2 + FY**2)) <= pupil/(wavelength*eyepiece))
            H_amp *= H_pupil

        # Apply Sinc Attenuation when higher orders are simulated         
        if order_num > 1:
            sincFX = torch.sin(math.pi*FX*pitch)/(math.pi*FX*pitch)
            sincFX[FX == 0.0] = 1
            H_amp = H_amp.float()*sincFX
            sincFY = torch.sin(math.pi*FY*pitch)/(math.pi*FY*pitch)
            sincFY[FY == 0.0] = 1
            H_amp *= sincFY

        # Convert to Pytorch Complex values
        H = utils.polar_to_rect(H_amp, H_exp)
        H = utils.ifftshift(H)
    else:
        H = precomped_H

    # return for use later as precomputed input
    if return_H:
        return H

    if writer is not None:
        H = utils.fftshift(H)
        writer.add_image(f'sinc_freq_filtered_amp/{tb_prefix}',
            torch.clamp(H.abs()[0,...], 0, 1), tb_idx)
        writer.add_image(f'sinc_freq_filtered_phase/{tb_prefix}',
            H.angle()[0,...]/(2*math.pi)+0.5, tb_idx)
        H = utils.ifftshift(H)

    # convolution performed in frequency domain
    U2 = H * U1

    # Optionally sum multiple wavefronts at output plane in frequency domain
    if summed_output:
        U2 = U2.sum(1, keepdim=True)

    # if writer is not None:
    #     U2 = utils.fftshift(U2)
    #     writer.add_image(f'propagated_freq_filtered_amp/{tb_prefix}',
    #         torch.clamp(U2.abs()[0,...], 0, 1), tb_idx)
    #     writer.add_image(f'propagated_freq_filtered_phase/{tb_prefix}',
    #         U2.angle()[0,...]/(2*math.pi)+0.5, tb_idx)
    #     U2 = utils.ifftshift(U2)

    # Fourier transform of the convolution to the observation plane
    u_out = utils.fftshift(torch.fft.ifftn(U2, dim=(-2, -1), norm='ortho'))
    u_out = utils.crop_image(u_out, output_resolution)
    return u_out

