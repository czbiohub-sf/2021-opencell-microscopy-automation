
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


# -----------------------------------------------------------------------------
#
# FOV selection settings
#
# -----------------------------------------------------------------------------
fov_selection_settings = FOVSelectionSettings(
    
    # the minimum number of positions at which to acquire z-stacks in each well
    # (ignoring the FOV scores)
    min_num_positions=2,

    # the max number of positions to acquire (again ignoring the FOV scores)
    max_num_positions=4,

    # the minimum score defines 'acceptable' FOVs
    # the value of -0.5 here is empirical,
    # and assumes we use a regression model to predict the score
    min_score=-0.5
)


# -----------------------------------------------------------------------------
#
# z-stack settings for fluorescence channels
# (range, relative to the AFC point, and step size)
#
# -----------------------------------------------------------------------------
STAGE_LABEL = 'PiezoZ'
dev_fl_stack_settings = StackSettings(
    stage_label=STAGE_LABEL,
    relative_top=16.0,
    relative_bottom=-10.0,
    step_size=7.0
)

prod_fl_stack_settings = StackSettings(
    stage_label=STAGE_LABEL,
    relative_top=12.0,
    relative_bottom=-9.0,
    step_size=0.2
)

# brightfield stack settings (with wider range and coarser step size)
bf_stack_settings = StackSettings(
    stage_label=STAGE_LABEL,
    relative_top=20.0,
    relative_bottom=-20.0,
    step_size=1.0
)


# -----------------------------------------------------------------------------
#
# Channel settings for hoechst and GFP
#
# -----------------------------------------------------------------------------
# common names and settings shared between channels
CONFIG_GROUP = 'Channels-EMCCD'
LASER_LINE = 'Andor ILE-A'
CAMERA_NAME = 'Andor EMCCD'
DEFAULT_LASER_POWER = 10
DEFAULT_CAMERA_GAIN = 400.0

hoechst_channel_settings = ChannelSettings(
    config_group=CONFIG_GROUP,
    config_name='EMCCD_Confocal40_DAPI',
    camera_name=CAMERA_NAME,
    laser_line=LASER_LINE,
    laser_name='Laser 405-Power Setpoint',
    default_camera_gain=DEFAULT_CAMERA_GAIN,
    default_laser_power=DEFAULT_LASER_POWER,
    default_exposure_time=100.0
)

gfp_channel_settings = ChannelSettings(
    config_group=CONFIG_GROUP,
    config_name='EMCCD_Confocal40_GFP',
    camera_name=CAMERA_NAME,
    laser_line=LASER_LINE,
    laser_name='Laser 488-Power Setpoint',
    default_laser_power=DEFAULT_LASER_POWER,
    default_camera_gain=DEFAULT_CAMERA_GAIN,
    default_exposure_time=50.0
)

# provisional brightfield settings
# TODO: check that the config name is the right one
bf_channel_settings = ChannelSettings(
    config_group=CONFIG_GROUP,
    config_name='EMCCD_BF',
    camera_name=CAMERA_NAME,
    laser_line=None,
    laser_name=None,
    default_laser_power=None,
    default_exposure_time=100.0,
    default_camera_gain=400.0
)


# -----------------------------------------------------------------------------
#
# Autoexposure settings
#
# -----------------------------------------------------------------------------
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


