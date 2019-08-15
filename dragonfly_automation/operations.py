
import numpy as np


def autofocus(mm_studio, mm_core, channel_settings):

    '''
    Autofocus using a given configuration

    Parameters
    ----------
    mm_studio, mm_core : 
    channel_settings : 

    TODO: optionally specify the autofocus method (either AFC or traditional autofocus)
    
    '''

    # change to the right channel
    change_channel(mm_core, channel_settings)

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



def change_channel(mm_core, channel_settings):
    '''
    Convenience method to set the laser power, exposure time, and camera gain
    
    (KC: pretty sure the order of these operations doesn't matter,
    but to be safe the order here is preserved from Nathan's script)
    '''

    # hardware config
    mm_core.setConfig(
        channel_settings.config_group, 
        channel_settings.config_name)

    # TODO: is this waitForConfig call necessary?
    mm_core.waitForConfig(
        channel_settings.config_group, 
        channel_settings.config_name)

    # laser power
    mm_core.setProperty(
        channel_settings.laser_line,
        channel_settings.laser_name,
        channel_settings.laser_power)

    # exposure time
    mm_core.setExposure(
        float(channel_settings.exposure_time))

    # camera gain
    prop_name = 'Gain'
    mm_core.setProperty(
        channel_settings.camera_name, 
        prop_name, 
        channel_settings.camera_gain)



def move_z(mm_core, zdevice, position=None, kind=None):
    '''
    Convenience method to move a z-stage
    (adapted from Nathan's script)

    TODO: basic sanity checks on the value of `position`
    (e.g., if kind=='relative', `position` shouldn't be a 'big' number)
    '''

    # validate `kind`
    if kind not in ['relative', 'absolute']:
        raise ValueError("`kind` must be either 'relative' or 'absolute'")
    
    # validate `position`
    try:
        position = float(position)
    except ValueError:
        raise TypeError('`position` cannot be coerced to float')
    
    if np.isnan(position):
        raise TypeError('`position` cannot be nan')
    
    # move the stage
    if kind=='absolute':
        mm_core.setPosition(zdevice, position)
    elif kind=='relative':
        mm_core.setRelativePosition(zdevice, position)
    
    # return the actual position of the stage
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
