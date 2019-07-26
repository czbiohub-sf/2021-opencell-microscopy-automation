
'''
Hard-coded constants and configuration for the pipeline_plate program

NOTE: exposure times and camera gain values must be floats
TODO: Do laser powers also need to be floats?

TODO: determine which of these constants are global 
      (instead of specific to pipeline plate acquisition)

'''

# Hardware settings
XY_STAGE = 'XYStage'
PIEZO_STAGE = 'PiezoZ'
LASER_LINE = 'Andor ILE-A'
CAMERA_NAME = 'Andor EMCCD'
CAMERA_GAIN = 400.0
HARDWARE_CONFIG_GROUP = 'Channels-EMCCD'


# start/end z-positions for the stack (relative to the AFC point)
ZSTACK_REL_START = -6.0
ZSTACK_REL_END = 16.0
ZSTACK_STEP_SIZE = 0.2


# min/max/default exposure times (in milliseconds)
MIN_EXPOSURE_TIME = 30.0
MAX_EXPOSURE_TIME = 500.0
DEFAULT_EXPOSURE_TIME = 50.0


# laser power is an int, not a float, because that's how it appears in Nathan's script
DEFAULT_LASER_POWER = 10
DEFAULT_EXPOSURE_TIME = 50.0
DEFAULT_CAMERA_GAIN = 400.0

CHANNEL_405 = {
    'name': 'EMCCD_Confocal40_DAPI',
    'laser_name': 'Laser 405-Power Setpoint',
    'laser_power': DEFAULT_LASER_POWER,
    'exposure_time': DEFAULT_EXPOSURE_TIME,
}


CHANNEL_488 = {
    'name': 'EMCCD_Confocal40_GFP',
    'laser_name': 'Laser 488-Power Setpoint',
    'laser_power': DEFAULT_LASER_POWER,
    'exposure_time': DEFAULT_EXPOSURE_TIME,
}


# the maximum number of FOVs/positions to acquire per well
# (if few FOVs pass the confluency test, we may end up with fewer stacks than this number)
MAX_NUM_FOV_PER_WELL = 8
