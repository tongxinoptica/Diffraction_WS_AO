import torch
import torch.nn as nn

import os
import time
import skimage.io
import utils.utils as utils
import platform
my_os = platform.system()
if my_os == 'Windows':
    from utils.physical_utils.arduino_laser_control_module import ArduinoLaserControl
    from utils.physical_utils.camera_capture_module import CameraCapture
    from utils.physical_utils.calibration_module import Calibration
    from utils.physical_utils.slm_display_module import SLMDisplay

class PhysicalProp(nn.Module):
    """ A module for physical propagation,
    forward pass displays gets SLM pattern as an input and display the pattern on the physical setup,
    and capture the diffraction image at the target plane,
    and then return warped image using pre-calibrated homography from instantiation.

    Class initialization parameters
    -------------------------------
    :param channel:
    :param slm_settle_time:
    :param roi_res: *** Note that the order of x / y is reversed here ***
    :param num_circles:
    :param laser_arduino:
    :param com_port:
    :param arduino_port_num:
    :param range_row:
    :param range_col:
    :param patterns_path:
    :param calibration_preview:

    Usage
    -----
    Functions as a pytorch module:

    >>> camera_prop = PhysicalProp(...)
    >>> captured_amp = camera_prop(slm_phase)

    slm_phase: phase at the SLM plane, with dimensions [batch, 1, height, width]
    captured_amp: amplitude at the target plane, with dimensions [batch, 1, height, width]

    """
    def __init__(self, channel=1, slm_settle_time=0.1, roi_res=(1680, 960), num_circles=(22, 13),
                 laser_arduino=False, com_port='COM3', arduino_port_num=(6, 10, 11),
                 range_row=(0, 1200), range_col=(0, 1920),
                 patterns_path=f'F:/citl/calibration', show_preview=False):
        super(PhysicalProp, self).__init__()

        # 1. Connect Camera
        self.camera = CameraCapture()
        self.camera.connect(0)  # specify the camera to use, 0 for main cam, 1 for the second cam

        # 2. Connect SLM
        self.slm = SLMDisplay()
        self.slm.connect()
        self.slm_settle_time = slm_settle_time

        # 3. Connect to the Arduino that switches rgb color through the laser control box.
        if laser_arduino:
            self.alc = ArduinoLaserControl(com_port, arduino_port_num)
            self.alc.switch_control_box(channel)
        else:
            self.alc = None

        # 4. Calibrate hardwares using homography
        calib_ptrn_path = os.path.join(patterns_path, f'{("red", "green", "blue")[channel]}.png')
        space_btw_circs = [int(roi / (num_circs - 1)) for roi, num_circs in zip(roi_res, num_circles)]

        self.calibrate(calib_ptrn_path, num_circles, space_btw_circs,
                       range_row=range_row, range_col=range_col, show_preview=show_preview)

    def calibrate(self, calibration_pattern_path, num_circles, space_btw_circs,
                  range_row, range_col, show_preview=False, num_grab_images=10):
        """
        pre-calculate the homography between target plane and the camera captured plane

        :param calibration_pattern_path:
        :param num_circles:
        :param space_btw_circs: number of pixels between circles
        :param slm_settle_time:
        :param range_row:
        :param range_col:
        :param show_preview:
        :param num_grab_images:
        :return:
        """

        self.calibrator = Calibration(num_circles, space_btw_circs)

        # supposed to be a grid pattern image (21 x 12) for calibration
        calib_phase_img = skimage.io.imread(calibration_pattern_path)
        self.slm.show_data_from_array(calib_phase_img)

        # sleep for 0.1s
        time.sleep(self.slm_settle_time)

        # capture displayed grid pattern image
        captured_intensities = self.camera.grab_images(num_grab_images)  # capture 5-10 images for averaging
        captured_img = utils.burst_img_processor(captured_intensities)

        # masking out dot pattern region for homography
        captured_img_masked = captured_img[range_row[0]:range_row[1], range_col[0]:range_col[1], ...]
        calib_success = self.calibrator.calibrate(captured_img_masked, show_preview=show_preview)

        self.calibrator.start_row, self.calibrator.end_row = range_row
        self.calibrator.start_col, self.calibrator.end_col = range_col

        if calib_success:
            print('   - calibration success')
        else:
            raise ValueError('  - Calibration failed')

    def forward(self, slm_phase, num_grab_images=1):
        """
        this forward pass gets slm_phase to display and returns the amplitude image at the target plane.

        :param slm_phase:
        :param num_grab_images:
        :return: A pytorch tensor shape of (1, 1, H, W)
        """

        slm_phase_8bit = utils.phasemap_8bit(slm_phase, True)

        # display the pattern and capture linear intensity, after perspective transform
        captured_linear_np = self.capture_linear_intensity(slm_phase_8bit, num_grab_images=num_grab_images)

        # convert raw-16 linear intensity image into an amplitude tensor
        if len(captured_linear_np.shape) > 2:
            captured_linear = torch.tensor(captured_linear_np, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
            captured_linear = captured_linear.to(slm_phase.device)
            captured_linear = torch.sum(captured_linear, dim=1, keepdim=True)
        else:
            captured_linear = torch.tensor(captured_linear_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            captured_linear = captured_linear.to(slm_phase.device)

        # return amplitude
        return torch.sqrt(captured_linear)

    def capture_linear_intensity(self, slm_phase, num_grab_images):
        """

        :param slm_phase:
        :param num_grab_images:
        :return:
        """

        # display on SLM and sleep for 0.1s
        self.slm.show_data_from_array(slm_phase)
        time.sleep(self.slm_settle_time)

        # capture and take average
        grabbed_images = self.camera.grab_images(num_grab_images)
        captured_intensity_raw_avg = utils.burst_img_processor(grabbed_images)  # averaging

        # crop ROI as calibrated
        captured_intensity_raw_cropped = captured_intensity_raw_avg[
            self.calibrator.start_row:self.calibrator.end_row,
            self.calibrator.start_col:self.calibrator.end_col, ...]
        # apply homography
        return self.calibrator(captured_intensity_raw_cropped)

    def disconnect(self):
        self.camera.disconnect()
        self.slm.disconnect()
        if self.alc is not None:
            self.alc.turnOffAll()
