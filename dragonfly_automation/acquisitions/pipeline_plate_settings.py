
'''
Settings for the 'pipeline_plate' program

NOTE: exposure times and camera gain values must be floats
TODO: should/must laser powers also be floats? (they are ints in Nathan's script)
'''

from dragonfly_automation.settings_schemas import (
    StackSettings,
    ChannelSettings,
    AutoexposureSettings,
    FOVSelectionSettings,
)


fov_selection_settings = FOVSelectionSettings(

    # how often to call AFC during FOV scoring
    # empirically, calling AFC at every other position is fine,
    # but calling it at every fourth position yields some out-of-focus snaps,
    # so we compromise and call it at every third position
    num_positions_between_afc_calls=2,

    # the minimum number of positions at which to acquire z-stacks in each well
    # (ignoring the FOV scores)
    min_num_positions=2,

    # the max number of positions to acquire (again ignoring the FOV scores)
    max_num_positions=4,

    # the minimum score defines 'acceptable' FOVs
    # note that the value of -0.5 here was empirically determined
    min_score=-0.5,

    # intensity threshold used to determine whether any nuclei are present in the FOV,
    # in units of raw fluorescence intensity in a snapshot of the 405 channel
    # note: a value of 700 is used for standard OpenCell/pipeline imaging, 
    # based on the observation that the background intensity in raw FOVs is around 500
    # WARNING: this value depends on the Hoechst staining protocol and the exposure settings!
    absolute_intensity_threshold=700,

    # the minimum number of nuclei that must be present in an FOV in order for it to be scored
    min_num_nuclei=10
)


# z-stack settings for fluorescence channels
STAGE_LABEL = 'PiezoZ'
fluorescence_stack_settings = StackSettings(
    stage_label=STAGE_LABEL,
    relative_top=12.0,
    relative_bottom=-9.0,
    step_size=0.2
)

brightfield_stack_settings = StackSettings(
    stage_label=STAGE_LABEL,
    relative_top=20.0,
    relative_bottom=-20.0,
    step_size=1.0
)


# common names and settings shared between channels
CONFIG_GROUP = 'Channels-EMCCD'
LASER_LINE = 'Andor ILE-A'
CAMERA_NAME = 'Andor EMCCD'
DEFAULT_CAMERA_GAIN = 400.0

hoechst_channel_settings = ChannelSettings(
    config_group=CONFIG_GROUP,
    config_name='EMCCD_Confocal40_DAPI',
    camera_name=CAMERA_NAME,
    laser_line=LASER_LINE,
    laser_name='Laser 405-Power Setpoint',
    default_camera_gain=DEFAULT_CAMERA_GAIN,
    default_laser_power=10,
    default_exposure_time=100.0
)

gfp_channel_settings = ChannelSettings(
    config_group=CONFIG_GROUP,
    config_name='EMCCD_Confocal40_GFP',
    camera_name=CAMERA_NAME,
    laser_line=LASER_LINE,
    laser_name='Laser 488-Power Setpoint',
    default_camera_gain=DEFAULT_CAMERA_GAIN,
    default_laser_power=15,
    default_exposure_time=50.0
)

brightfield_channel_settings = ChannelSettings(
    config_group=CONFIG_GROUP,
    config_name='EMCCD_BF',
    camera_name=CAMERA_NAME,
    laser_line=None,
    laser_name=None,
    default_camera_gain=DEFAULT_CAMERA_GAIN,
    default_laser_power=None,
    default_exposure_time=100.0
)


autoexposure_settings = AutoexposureSettings(

    # min and max intensities that define under- and over-exposure
    # both are set to 2**15 to leave a (literal) bit of wiggle room
    min_intensity=2**15,
    max_intensity=2**15,

    # min and max exposure times 
    # (min of 40ms is to avoid artifacts from the spinning disk)
    min_exposure_time=50.0,
    max_exposure_time=500.0,
    default_exposure_time=50.0,

    # minimum laser power (in percent)
    min_laser_power=0.1,

    # exposure step (for over-exposure) is from Nathan
    relative_exposure_step=0.8,

    # a coarse z-step is sufficient for stepping through the stack during autoexposure
    z_step_size=1.0
)


