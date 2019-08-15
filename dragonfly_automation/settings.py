
from types import SimpleNamespace
from collections import namedtuple


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

