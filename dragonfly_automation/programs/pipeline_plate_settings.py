
'''
Settings for the 'pipeline_plate' program

NOTE: exposure times and camera gain values must be floats
TODO: Must laser powers also be floats? (they are ints in Nathan's script)

'''

from dragonfly_automation.settings import ChannelSettings


# -----------------------------------------------------------------------------
#
# Various free-floating constants used directly in 'pipeline_plate_program'
#
# -----------------------------------------------------------------------------

# the maximum number of FOVs/positions to acquire per well
# (note that if few FOVs pass the confluency test, 
# we may end up with fewer stacks than this number)
MAX_NUM_FOV_PER_WELL = 8

# z-stack range and stepsize (relative to the AFC point)
ZSTACK_REL_START = -6.0
ZSTACK_REL_END = 16.0
ZSTACK_STEP_SIZE = 0.2

# min/max/default exposure times (in milliseconds)
MIN_EXPOSURE_TIME = 30.0
MAX_EXPOSURE_TIME = 500.0


# -----------------------------------------------------------------------------
#
# Constants and default values for both DAPI and GFP channels
# (These are used only to instantiate the ChannelSettings objects below)
#
# -----------------------------------------------------------------------------
DEFAULT_LASER_POWER = 10
DEFAULT_CAMERA_GAIN = 400.0
DEFAULT_EXPOSURE_TIME = 50.0

# the name of the channel config group
# (to which channel configs are applied)
CONFIG_GROUP = 'Channels-EMCCD'

# the name of the laser line (which is a type of device)
LASER_LINE = 'Andor ILE-A'

# the name of the camera (another device)
CAMERA_NAME = 'Andor EMCCD'


# -----------------------------------------------------------------------------
#
# Channel settings
#
# -----------------------------------------------------------------------------
dapi_channel = ChannelSettings(
    config_group=CONFIG_GROUP,
    config_name='EMCCD_Confocal40_DAPI',
    camera_name=CAMERA_NAME,
    laser_line=LASER_LINE,
    laser_name='Laser 405-Power Setpoint',
    default_laser_power=DEFAULT_LASER_POWER,
    default_exposure_time=DEFAULT_EXPOSURE_TIME,
    default_camera_gain=DEFAULT_CAMERA_GAIN)


gfp_channel = ChannelSettings(
    config_group=CONFIG_GROUP,
    config_name='EMCCD_Confocal40_GFP',
    camera_name=CAMERA_NAME,
    laser_line=LASER_LINE,
    laser_name='Laser 488-Power Setpoint',
    default_laser_power=DEFAULT_LASER_POWER,
    default_exposure_time=DEFAULT_EXPOSURE_TIME,
    default_camera_gain=DEFAULT_CAMERA_GAIN)


