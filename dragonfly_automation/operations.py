
import numpy as np


def log_operation(operation):

    def wrapper(*args, **kwargs):
        print('\nSTART OPERATION: %s' % operation.__name__)
        result = operation(*args, **kwargs)
        print('END OPERATION: %s\n' % operation.__name__)
        return result

    return wrapper


@log_operation
def autofocus(mm_studio, mm_core):

    '''
    Autofocus using a given configuration

    TODO: optionally specify the autofocus method (either AFC or traditional autofocus)
    
    '''

    # get the current AutofocusPlugin being used for autofocusing
    af_manager = mm_studio.getAutofocusManager()
    af_plugin = af_manager.getAutofocusMethod()

    try:
        af_plugin.fullFocus()
    except Exception as error:
        print('WARNING: AFC failure')
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
        shape=(meta.getxRange(), meta.getyRange()))
    
    return data


@log_operation
def acquire_stack(
    mm_studio, 
    mm_core, 
    datastore, 
    stack_settings, 
    position_ind, 
    channel_ind):
    '''
    Acquire a z-stack using the given settings
    and 'put' it in the datastore object
    '''

    # generate a list of the z positions to visit
    z_positions = np.arange(
        stack_settings.relative_bottom, 
        stack_settings.relative_top + stack_settings.step_size, 
        stack_settings.step_size)

    for z_ind, z_position in enumerate(z_positions):

        # move to the new z-position 
        move_z_stage(
            mm_core, 
            stack_settings.stage_label, 
            position=z_position,
            kind='absolute')

        # acquire an image
        mm_core.waitForImageSynchro()
        mm_core.snapImage()

        # convert the image
        # TODO: understand what's happening here
        tagged_image = mm_core.getTaggedImage()
        image = mm_studio.data().convertTaggedImage(tagged_image)

        # assign metadata to the image
        # NOTE that the TIFF filename is determined by the value passed to stagePosition 
        # (which has to be a int)
        # TODO: see if there is there a way to include the value passed to positionName
        # (which can be any string) in the TIFF filename
        image = image.copyWith(
            image.getCoords().copy().channel(channel_ind).z(z_ind).stagePosition(position_ind).build(),
            image.getMetadata().copy().positionName(str(position_ind)).build()
        )

        if datastore:
            datastore.putImage(image)



@log_operation
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


def move_z_stage(mm_core, stage_label, position=None, kind=None):
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
        mm_core.setPosition(stage_label, position)
    elif kind=='relative':
        mm_core.setRelativePosition(stage_label, position)
    
    # return the actual position of the stage
    mm_core.waitForDevice(stage_label)
    actual_position = mm_core.getPosition(stage_label)
    return actual_position


@log_operation
def autoexposure(
    gate,
    mm_studio,
    mm_core,
    stack_settings, 
    autoexposure_settings,
    channel_settings):
    '''

    Parameters
    ----------
    gate, mm_studio, mm_core : gateway objects
    stack_settings : an instance of StackSettings
    autoexposure_settings : an instance of AutoexposureSettings
    channel_settings : the ChannelSettings instance corresponding to the channel 
        on which to run the autoexposure algorithm
        NOTE that this method modifies the `laser_power` and `exposure_time` attributes


    Returns
    -------
    autoexposure_did_succeed : bool
        Whether the autoexposure algorithm was successful


    Algorithm description
    ---------------------
    slice check:
        while an over-exposed slice exists:
            step through the z-stack until an over-exposed slice is encountered,
            then lower the exposure time and/or laser power

    stack check:
        if no slices were over-exposed, check for under-exposure using the overall max intensity,
            and lower the exposure time if necessary

    '''
    
    autoexposure_did_succeed = True

    # move to the bottom of the z-stack
    current_z_position = move_z_stage(
        mm_core, 
        stack_settings.stage_label, 
        position=stack_settings.relative_bottom, 
        kind='absolute')
    
    # keep track of the maximum intensity
    stack_max_intensity = 0
    
    # keep track of whether any slices were ever over-exposed
    overexposure_did_occur = False

    # step through the z-stack and check each slice for over-exposure
    while current_z_position <= stack_settings.relative_top:

        # snap an image and check the exposure
        mm_core.waitForSystem()
        snap_data = acquire_snap(gate, mm_studio)

        # note that the 99.9th percentile here corresponds to ~1000 pixels in a 1024x1024 image
        slice_was_overexposed = np.percentile(snap_data, 99.9) > autoexposure_settings.max_intensity

        # if the slice was over-exposed, lower the exposure time or the laser power,
        # reset stack_max_intensity, and go back to the bottom of the z-stack
        if slice_was_overexposed:
            overexposure_did_occur = True
            print('z-slice at %s was overexposed' % current_z_position)

            # lower the exposure time; if it falls below the minimum, turn down the laser instead
            channel_settings.exposure_time *= autoexposure_settings.relative_exposure_step
            if channel_settings.exposure_time < autoexposure_settings.min_exposure_time:
                channel_settings.exposure_time = autoexposure_settings.default_exposure_time
                channel_settings.laser_power *= autoexposure_settings.relative_exposure_step
        
                # update the laser power
                mm_core.setProperty(
                    channel_settings.laser_line,
                    channel_settings.laser_name,
                    channel_settings.laser_power)

            # update the exposure time
            mm_core.setExposure(
                float(channel_settings.exposure_time))

            # prepare to return to the bottom of the stack
            new_z_position = stack_settings.relative_bottom

            # reset the max intensity
            stack_max_intensity = 0

            # break out of the while loop if the exposure has been lowered
            # as far as it can be and the slice is still over-exposed
            # KC: in practice, I believe this should rarely/never happen
            if channel_settings.laser_power < autoexposure_settings.min_laser_power:
                autoexposure_did_succeed = False
                break

        # if the slice was not over-exposed, 
        # update stack_max and move to the next z-slice
        else:
            stack_max_intensity = max(stack_max_intensity, snap_data.max())
            new_z_position = current_z_position + stack_settings.step_size
    
        # move to the new z-position 
        # (either the next slice or the bottom of the stack)
        current_z_position = move_z_stage(
            mm_core, 
            stack_settings.stage_label, 
            position=new_z_position,
            kind='absolute')


    # after exiting the while-loop, either
    # 1) some slices were over-exposed and the exposure is now adjusted, or
    # 2) no slices were over-exposed and we need to check for under-exposure
    # here, we check for scenario (2) and use stack_max_intensity to increase
    # the exposure time if it is too low
    if not overexposure_did_occur:
        intensity_ratio = autoexposure_settings.min_intensity / stack_max_intensity
        if intensity_ratio > 1:
            channel_settings.exposure_time *= intensity_ratio
            if channel_settings.exposure_time > autoexposure_settings.max_exposure_time:
                print('Warning: stack was under-exposed and maximum exposure time was exceeded')
                channel_settings.exposure_time = autoexposure_settings.max_exposure_time

    return autoexposure_did_succeed

