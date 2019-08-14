
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


def acquire_stack(datastore, config):
    pass


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
