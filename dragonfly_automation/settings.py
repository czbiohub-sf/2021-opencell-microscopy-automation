
from types import SimpleNamespace
from collections import namedtuple


StackSettings = namedtuple('StackSettings', [
    
    # the name of the stage to use for stepping through the stack
    # (usually the Piezo stage called 'PiezoZ')
    'stage_label',

    # top and bottom of the stack, in um, relative to the AFC point
    'relative_top',
    'relative_bottom',

    # step size (in um)
    'step_size',
])


AutoexposureSettings = namedtuple('AutoexposureSettings', [

    # intensity used to define over-exposure in the autoexposure algorithm
    'max_intensity',

    # min/max/default exposure times 
    # (laser power is adjusted if exposure time falls below min_exposure_time)
    'min_exposure_time',
    'max_exposure_time',
    'default_exposure_time',

    # factor by which to decrease the exposure time or laser power
    # if a z-slice is found to be over-exposed 
    'relative_exposure_step',
])


class ChannelSettings(object):

    def __init__(
        self,
        config_group,
        config_name,
        camera_name,
        laser_line,
        laser_name,
        default_laser_power,
        default_camera_gain,
        default_exposure_time):

        self.config_group = config_group
        self.config_name = config_name
        self.camera_name = camera_name
        self.laser_line = laser_line
        self.laser_name = laser_name
        self.default_laser_power = default_laser_power
        self.default_camera_gain = default_camera_gain
        self.default_exposure_time = default_exposure_time

        self.reset()


    def reset(self):
        self.laser_power = self.default_laser_power
        self.camera_gain = self.default_camera_gain
        self.exposure_time = self.default_exposure_time

