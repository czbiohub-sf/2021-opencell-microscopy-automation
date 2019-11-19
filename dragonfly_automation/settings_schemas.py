
from types import SimpleNamespace
from collections import namedtuple


FOVSelectionSettings = namedtuple('FOVSelectionSettings', [

    # empirical minimum FOV score to define 'acceptable' FOVs
    'min_score',

    # the minimum number of positions to image in a well
    'min_num_positions',

    # the maximum number of positions to image in a well
    'max_num_positions',
])


StackSettings = namedtuple('StackSettings', [

    # the name of the stage to use for stepping through the stack
    # (this should usually be the Piezo stage named 'PiezoZ')
    'stage_label',

    # top and bottom of the stack, in um, relative to the AFC point
    'relative_top',
    'relative_bottom',

    # step size in um
    'step_size',
])


AutoexposureSettings = namedtuple('AutoexposureSettings', [

    # max intensity used to define over-exposure
    'max_intensity',

    # min intensity used to define under-exposure
    'min_intensity',

    # minimum exposure time used to decide when to lower the laser power
    'min_exposure_time',

    # max exposure time used during adjustment for under-exposure
    'max_exposure_time',

    # the initial exposure time used when the laser power is lowered
    'default_exposure_time',

    # the minimum laser power (used to define autoexposure failure)
    'min_laser_power',

    # factor by which to decrease the exposure time or laser power
    # if a z-slice is found to be over-exposed 
    'relative_exposure_step',

    # z-step size to use
    'z_step_size',
])


ChannelSettings = namedtuple('ChannelSettings', [

    # the name of the channel config group
    # (to which channel settings/properties are applied)
    'config_group',

    # 
    'config_name',

    # the name (or 'label') of the camera (which is a type of device)
    'camera_name',

    # the name of the laser line (another type of device)
    'laser_line',

    # the name of the laser itself (TODO: is this also a device?)
    'laser_name',

    # default values for exposure settings
    'default_laser_power',
    'default_camera_gain',
    'default_exposure_time'
])


class ChannelSettingsManager:

    def __init__(self, channel_settings):
        '''
        Manager for a ChannelSettings object
    
        Adds mutable attributes for exposure settings and a reset method
        to reset these settings to their immutable default values
        '''
        for key, value in dict(channel_settings._asdict()).items():
            setattr(self, key, value)
        self.reset()

    def reset(self):
        self.laser_power = self.default_laser_power # pylint: disable=no-member
        self.camera_gain = self.default_camera_gain # pylint: disable=no-member
        self.exposure_time = self.default_exposure_time # pylint: disable=no-member

