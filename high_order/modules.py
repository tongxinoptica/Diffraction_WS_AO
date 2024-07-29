"""
Some modules to wrap the algorithms for computing phases
"""
import torch, math
import torch.nn as nn

from algorithms import stochastic_gradient_descent, \
    double_phase_amplitude_coding, higher_order_gradient_descent


class DPAC(nn.Module):
    """Double-phase Amplitude Coding
    Class initialization parameters
    -------------------------------
    :param prop_dists: propagation dists from SLM to target planes, in meters
    :param wavelength: the wavelength of interest, in meters
    :param propagator: propagator instance (function / pytorch module)
    :param linear_phase_compensation: Flag to use linear phase compensation
        from Maimone et al 2017
    :param preblur_sigma: preblur_sigma for AA-DPM from Shi et al 2021
    :param device: torch.device
    """
    def __init__(self, prop_dists, wavelength, propagator,
        linear_phase_compensation=True, preblur_sigma=0.0,
        device=torch.device('cuda')):
        super(DPAC, self).__init__()

        # propagation is from target to SLM plane (one step)
        self.prop_dists = prop_dists
        self.wavelength = wavelength
        self.prop = propagator
        self.dev = device
        self.linear_phase_compensation = linear_phase_compensation
        self.preblur_sigma = preblur_sigma
        self._tb_prefix = ''

    def forward(self, target_amps, target_phases=None):
        """Compute phase at SLM using double-phase amplitude coding
        Input
        -----
        :param target_amps: amplitude at the target planes, with dimensions
            [1, depths, *roi_res]
        :param target_phases: phase at the target planes, with dimensions
            [1, depths, *roi_res]

        Output
        ------
        :return phase: optimized phase-only representation at SLM plane,
            with dimensions [1, 1, *slm_res]
        """
        if target_phases is None:
            if self.linear_phase_compensation:
                target_phases = torch.zeros_like(target_amps)
                for k, prop_dist in enumerate(self.prop_dists):
                    target_phases[:,k, ...] = \
                        2*math.pi*prop_dist/self.wavelength
            else:
                target_phases = torch.zeros_like(target_amps)

        phase = double_phase_amplitude_coding(target_amps, target_phases, 
            self.prop, preblur_sigma=self.preblur_sigma)

        return phase

    @property
    def tb_prefix(self):
        return self._tb_prefix

    @tb_prefix.setter
    def tb_prefix(self, tb_prefix):
        self._tb_prefix = tb_prefix


class SGD(nn.Module):
    """Stochastic Gradient Descent Algorithm
    Class initialization parameters
    -------------------------------
    :param num_iters: the number of iteration
    :param propagator: propagator instance (function / pytorch module)
    :param loss: loss function, default L2
    :param lr: learning rate for phase variables
    :param lr_s: learning rate for scale factor, set to 0.0 for optimal scale
    :param s0: initialization for scale factor
    :param camera_prop: optional physical propagation forward model for CITL
    :param phase_path: path to write out intermediate phases
    :param writer: SummaryWriter instance for tensorboard
    :param device: torch.device
    """
    def __init__(self, num_iters, propagator, loss=nn.MSELoss(), lr=0.01,
        lr_s=0.03, s0=1.0, full_bandwidth=False, camera_prop=None,
        phase_path='./phases', writer=None, device=torch.device('cuda')):
        super(SGD, self).__init__()

        # Store parameters
        self.num_iters = num_iters
        self.prop = propagator
        self.loss = loss.to(device)
        self.lr = lr
        self.lr_s = lr_s
        self.s0 = s0
        self.full_bandwidth = full_bandwidth
        self.camera_prop = camera_prop
        self.phase_path = phase_path
        self.writer = writer
        self.dev = device
        self._tb_prefix = ''

    def forward(self, target_amp, init_phase, masks=None):
        """Compute phase at SLM RGBD scene using SGD
        Input
        -----
        :param target_amp: amplitude at the target plane, with dimensions
            [1, 1, *roi_res]
        :param init_phase: initial SLM phase estimate, with dimensions
            [1, 1, *slm_res]
        :param masks: decomposition of target amp across target planes, with
            dimensions [1, depths, *roi_res]

        Output
        ------
        :return final_phase: optimized phase-only representation at SLM plane,
            with dimensions  [1, 1, *slm_res]
        """

        final_phase = stochastic_gradient_descent(init_phase, target_amp,
            masks, self.num_iters, self.phase_path, self.prop,
            tb_prefix=self._tb_prefix, loss=self.loss, lr=self.lr,
            lr_s=self.lr_s, s0=self.s0, full_bandwidth=self.full_bandwidth,
            camera_prop=self.camera_prop, writer=self.writer)

        return final_phase

    @property
    def tb_prefix(self):
        return self._tb_prefix

    @tb_prefix.setter
    def tb_prefix(self, tb_prefix):
        self._tb_prefix = tb_prefix


class HOGD(nn.Module):
    """Higher Order Gradient Descent Algorithm
    Class initialization parameters
    -------------------------------
    :param num_iters: the number of iteration
    :param propagator: propagator instance (function / pytorch module)
    :param loss: loss function, default L2
    :param lr: learning rate for phase variables
    :param lr_s: learning rate for scale factor, set to 0.0 for optimal scale
    :param s0: initialization for scale factor
    :param camera_prop: optional physical propagation forward model for CITL
    :param citl_y_offset: y sampling offset for CITL in range of [0, 6)
    :param citl_x_offset: x sampling offset for CITL in range of [0, 6)
    :param phase_path: path to write out intermediate phases
    :param writer: SummaryWriter instance for tensorboard
    :param device: torch.device
    :param supervision_box_blur: blur of target supervision
    :param downsampled_loss: If true, downsample blurred target and recon
        before computing loss
    :param periphery_box_blur: If set, blur of periphery supervision.
    """
    def __init__(self, num_iters, propagator, loss=nn.MSELoss(), lr=0.01,
        lr_s=0.03, s0=1.0, camera_prop=None, citl_y_offset=0, citl_x_offset=0,
        phase_path='./phases', writer=None, device=torch.device('cuda'),
        supervision_box_blur=6, downsampled_loss=True, periphery_box_blur=6):
        super(HOGD, self).__init__()

        # Store parameters
        self.num_iters = num_iters
        self.prop = propagator
        self.loss = loss.to(device)
        self.lr = lr
        self.lr_s = lr_s
        self.s0 = s0
        self.camera_prop = camera_prop
        self.phase_path = phase_path
        self.writer = writer
        self.dev = device
        self._tb_prefix = ''
        self.supervision_box_blur = supervision_box_blur
        self.downsampled_loss = downsampled_loss
        self.periphery_box_blur = periphery_box_blur
        self.citl_y_offset = citl_y_offset
        self.citl_x_offset = citl_x_offset

    def forward(self, target_amp, init_phase, masks=None,
        foveation_mask=None):
        """Compute phase at SLM RGBD scene using SGD
        Input
        -----
        :param target_amp: amplitude at the target plane, with dimensions
            [1, 1, *roi_res]
        :param init_phase: initial SLM phase estimate, with dimensions
            [1, 1, *slm_res]
        :param masks: decomposition of target amp across target planes, with
            dimensions [1, depths, *roi_res]
        :param foveation_mask: mask to apply foveated target resolution, with
            dimensions [1, 1, *roi_res]

        Output
        ------
        :return final_phase: optimized phase-only representation at SLM plane,
            with dimensions  [1, 1, *slm_res]
        """

        final_phase = higher_order_gradient_descent(init_phase, target_amp,
            masks, self.num_iters, self.phase_path, self.prop,
            tb_prefix=self._tb_prefix, loss=self.loss, lr=self.lr,
            lr_s=self.lr_s, s0=self.s0, camera_prop=self.camera_prop,
            citl_y_offset=self.citl_y_offset,
            citl_x_offset=self.citl_x_offset, writer=self.writer,
            supervision_box_blur=self.supervision_box_blur,
            periphery_box_blur=self.periphery_box_blur,
            downsampled_loss=self.downsampled_loss,
            foveation_mask=foveation_mask)

        return final_phase

    @property
    def tb_prefix(self):
        return self._tb_prefix

    @tb_prefix.setter
    def tb_prefix(self, tb_prefix):
        self._tb_prefix = tb_prefix

        