
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

    mm_studio.live().snap(True)
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



def autoexposure(program, settings, channel_settings):
    '''
    
    Parameters
    ----------
    program : a program instance (for accessing gate, mm_studio, and mm_core)
    settings : 
    channel_settings : 


    Returns
    -------

    laser_power  : float
        The calculated laser power

    exposure_time : float
        The calculated exposure time

    autoexposure_did_succeed : bool
        Whether the autoexposure algorithm was successful


    Algorithm description
    ---------------------
    slice check:
    while an over-exposed slice exists:
      step through z-stack until an over-exposed slice is encountered
      lower exposure time and/or laser power

    stack check:
    one-time check for under-exposure using the overall max intensity;
    increase the exposure time, up to the maximum, if necessary

    '''
    
    laser_power = channel_settings.laser_power
    exposure_time = channel_settings.exposure_time

    # move to the bottom of the z-stack
    current_z_position = move_z(
        program.mm_core, 
        program.zstage, 
        position=settings.ZSTACK_REL_START, 
        kind='relative')

    while current_z_position <= settings.ZSTACK_REL_END:

        # snap an image
        program.mm_core.waitForSystem()
        snap = acquire_snap(program.gate, program.mm_studio)

        # check exposure
        laser_power, exposure_time, slice_was_overexposed = check_slice_for_overexposure(
            settings, snap, laser_power, exposure_time)

        if slice_was_overexposed:
            new_z_position = settings.ZSTACK_REL_START
            # set the laser power and exposure time

        else:
            new_z_position = settings.ZSTACK_STEP_SIZE

        # move to the new z-position 
        # (either the next slice or back to the start/bottom of the stack)
        current_z_position = move_z(
            program.mm_core, 
            program.zstage, 
            position=new_z_position,
            kind='relative')
    

    return None, None, True



def check_slice_for_overexposure(settings, snap, current_laser_power, current_exposure_time):
    '''
    Check a single slice for over-exposure and lower the exposure time or laser power
    if it is indeed over-exposed

    Parameters
    ----------
    settings : program-level settings object (with min/max/default exposure times etc)
    snap : the slice image data as a numpy array (assumed to be uint16)
    current_laser_power : the laser power used to acquire the snap
    current_exposure_time : the exposure time used to acquire the snap
    '''

    new_laser_power = current_laser_power
    new_exposure_time = current_exposure_time

    # KC: the 99.9th percentile corresponds to ~1000 pixels in a 1024x1024 image;
    # this value was empirically determined
    slice_was_overexposed = np.percentile(snap, 99.9) > settings.MAX_INTENSITY
    if slice_was_overexposed:
        new_exposure_time = settings.RELATIVE_EXPOSURE_STEP * current_exposure_time

        # if the new exposure time is below the minimum, turn down the laser instead
        if new_exposure_time < settings.MIN_EXPOSURE_TIME:
            new_exposure_time = settings.DEFAULT_EXPOSURE_TIME
            new_laser_power = settings.RELATIVE_EXPOSURE_STEP * current_laser_power

    return new_laser_power, new_exposure_time, slice_was_overexposed