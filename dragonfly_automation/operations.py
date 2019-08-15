
import numpy as np
from dragonfly_automation import global_settings


def autofocus(
    mm_studio, 
    mm_core, 
    channel_name, 
    laser_name, 
    laser_power, 
    camera_gain, 
    exposure_time,
    config_group=global_settings.HARDWARE_CONFIG_GROUP, 
    laser_line=global_settings.LASER_LINE,
    camera_name=global_settings.CAMERA_NAME):

    '''
    Autofocus using a given configuration

    Parameters
    ----------
    mm_studio, mm_core : 
    channel_name : 
    laser_name : 
    laser_power :
    camera_gain : 
    exposure_time : 

    TODO: optionally specify the autofocus method (either AFC or traditional autofocus)
    
    '''

    mm_core.setConfig(config_group, channel_name)
    mm_core.waitForConfig(config_group, channel_name)

    mm_core.setExposure(float(exposure_time))
    mm_core.setProperty(laser_line, laser_name, laser_power)
    mm_core.setProperty(camera_name, 'Gain', camera_gain)

    af_manager = mm_studio.getAutofocusManager()
    af_plugin = af_manager.getAutofocusMethod()

    try:
        af_plugin.fullFocus()
        print('AFC success')
    except Exception as error:
        print('AFC failure')
        print(error)

    # just to be safe
    mm_core.waitForSystem()


def acquire_snap(gate, mm_studio):
    '''
    Acquire an image using the current laser/camera/exposure settings
    and return the image data as a numpy memmap

    TODO: call gate.clearQueue (wait for new version of mm2python)
    TODO: check that meta.bitDepth is uint16
    '''

    mm_studio.live().snape(True)
    meta = gate.getLastMeta()

    data = np.memmap(
        meta.getFilepath(),
        dtype='uint16',
        mode='r+',
        offset=0,
        shape=(meta.getxRange(). meta.getyRange()))
    
    return data


def acquire_stack(mm_core, datastore, settings):
    '''
    Acquire a z-stack using the given settings
    and 'put' it in the datastore object
    '''

    # move_z(mm_core, zdevice, position=0, kind='absolute')

    pass


def update_channel_settings(mm_core, channel_settings):
    '''
    Convenience method to set the laser power, exposure time, and camera gain
    
    (KC: not sure whether the order of these operations matters,
    but to be safe the order here is preserved from Nathan's script)
    '''

    mm_core.setConfig(
        global_settings.HARDWARE_CONFIG_GROUP, 
        channel_settings['name'])

    # laser power
    mm_core.setProperty(
        global_settings.LASER_LINE,
        channel_settings['laser_name'],
        channel_settings['laser_power'])

    # exposure time
    mm_core.setExposure(
        float(channel_settings['exposure_time']))

    # camera gain
    mm_core.setProperty(
        global_settings.CAMERA_NAME,
        'Gain',
        channel_settings['camera_gain'])



def move_z(mm_core, zdevice, position=None, kind=None):
    '''
    Convenience method to move a z-stage
    (adapted from Nathan's script)

    TODO: basic sanity checks on the value of `position`
    (e.g., if kind=='relative', `position` shouldn't be a 'big' number)
    '''

    if kind not in ['relative', 'absolute']:
        raise ValueError("`kind` must be either 'relative' or 'absolute'")

    try:
        position = float(position)
    except ValueError:
        raise TypeError('`position` cannot be coerced to float')
    
    if np.isnan(position):
        raise TypeError('`position` cannot be nan')
    
    if kind=='absolute':
        mm_core.setPosition(zdevice, position)
    elif kind=='relative':
        mm_core.setRelativePosition(zdevice, position)

    mm_core.waitForDevice(zdevice)
    actual_position = mm_core.getPosition(zdevice)
    return actual_position


def autoexposure(config):
    '''
    
    Parameters
    ----------

    Returns
    -------

    laser_power  : float
        The calculated laser power

    exposure_time : float
        The calculated exposure time

    success : bool
        Whether the autoexposure algorithm was successful

    '''
    
    return None, None, True
