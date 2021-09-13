from dataclasses import dataclass


@dataclass
class FOVSelectionSettings:

    # empirical minimum FOV score to define 'acceptable' FOVs
    min_score: float

    # the minimum number of positions to image in a well
    min_num_positions: int

    # the maximum number of positions to image in a well
    max_num_positions: int

    # to save time, we call AFC at every nth position during FOV scoring
    # (this is usually possible because the positions within a well are close to one another)
    num_positions_between_afc_calls: int

    # intensity threshold used to determine whether any nuclei are present in the FOV,
    # in units of raw fluorescence intensity in a snapshot of the 405 channel
    # WARNING: this value depends on the Hoechst staining protocol and the exposure settings!
    absolute_intensity_threshold: int

    # the minimum number of nuclei that must be present in an FOV in order for it to be scored
    min_num_nuclei: int


@dataclass
class StackSettings:

    # the name of the stage to use for stepping through the stack
    # (this should usually be the Piezo stage, named 'PiezoZ')
    stage_label: str

    # the z-position of the top and bottom of the stack, relative to the AFC setpoint (in microns)
    relative_top: float
    relative_bottom: float

    # z-step size (in microns)
    step_size: float


@dataclass
class AutoexposureSettings:

    # max intensity used to define over-exposure
    max_intensity: int

    # min intensity used to define under-exposure
    min_intensity: int

    # minimum exposure time used to decide when to lower the laser power
    min_exposure_time: float

    # max exposure time used during adjustment for under-exposure
    max_exposure_time: float

    # the initial exposure time used when the laser power is lowered
    default_exposure_time: float

    # the minimum laser power (used to define autoexposure failure)
    min_laser_power: float

    # factor by which to decrease the exposure time or laser power
    # if a z-slice is found to be over-exposed
    relative_exposure_step: float

    # z-step size to use
    z_step_size: float


@dataclass
class ChannelSettings:

    # the name of the channel config group
    # (to which channel settings/properties are applied)
    config_group: str
    config_name: str

    # the name (or 'label') of the camera (which is a type of device)
    camera_name: str

    # the name of the laser line (this actually refers to a group of lasers)
    laser_line: str

    # the name of the laser itself (this is wavelength-specific)
    laser_name: str

    # default values for exposure settings
    default_laser_power: float
    default_camera_gain: float
    default_exposure_time: float

    def __post_init__(self):
        self.reset()

    def reset(self):
        self.laser_power = self.default_laser_power
        self.camera_gain = self.default_camera_gain
        self.exposure_time = self.default_exposure_time
