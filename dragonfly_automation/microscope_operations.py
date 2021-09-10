
import py4j
import time
import numpy as np


class MicroscopeOperations:

    def __init__(self, event_logger):
        self.event_logger = event_logger

    def __getattr__(self, name):
        operation = globals()[name]

        def wrapper(*args, **kwargs):
            self.event_logger('OPERATION INFO: Calling %s' % operation.__name__)
            result = operation(*args, **kwargs)
            self.event_logger('OPERATION INFO: Exiting %s' % operation.__name__)
            return result
        return wrapper


def go_to_position(micromanager_interface, position_ind):

    mm_position_list = micromanager_interface.mm_studio.getPositionList()
    mm_position = mm_position_list.getPosition(position_ind)

    # move the stage to the new position
    # note that `goToPosition` moves the stages specified in the position list
    # we try twice because, for large stage movements, MicroManager will throw a timeout error
    try:
        mm_position.goToPosition(mm_position, micromanager_interface.mm_core)
    except py4j.protocol.Py4JJavaError:
        mm_position.goToPosition(mm_position, micromanager_interface.mm_core)


def call_afc(micromanager_interface, event_logger, afc_logger=None, position_ind=None):

    '''
    Minimal wrapper around the `fullFocus` method of the active autofocus plugin,
    ** which is assumed to be AFC **

    TODO: consider switching to mm_core API, 
    which has its own fullFocus method - might be faster

    '''

    # get the active AutofocusPlugin (assumed to be AFC)
    af_manager = micromanager_interface.mm_studio.getAutofocusManager()
    af_plugin = af_manager.getAutofocusMethod()

    # the initial AFC score and FocusDrive position
    initial_afc_score = af_plugin.getCurrentFocusScore()
    initial_focusdrive_position = micromanager_interface.mm_core.getPosition('FocusDrive')

    # here we attempt to call AFC at various FocusDrive positions.
    # the logic of this is that, when AFC times out, it is usually because
    # the FocusDrive stage is too low, so here, when it times out,
    # we move the stage up in 10um steps and attempt to call AFC at each step
    successful_offset = None
    afc_error_message = None
    afc_did_succeed = False
    failed_offsets = []

    focusdrive_offsets = [0, 10, 20, 40, 60, -20]
    for offset in focusdrive_offsets:
        if afc_did_succeed:
            continue

        if offset != 0:
            # if we're here, it means AFC has failed once (at offset = 0),
            # which means we need to reset the FocusDrive to its original position 
            # and then move it up by the (now nonzero) offset
            # (note that when AFC times out, it lowers the FocusDrive by around 500um)
            focusdrive_position = initial_focusdrive_position + offset
            move_z_stage(
                micromanager_interface, 
                stage_label='FocusDrive', 
                position=focusdrive_position,
                kind='absolute'
            )
            # delay to help AFC 'adjust' to the new position (see comments below)
            time.sleep(0.5)

        try:
            af_plugin.fullFocus()
            afc_did_succeed = True
            successful_offset = offset
        except py4j.protocol.Py4JJavaError as error:
            event_logger("AUTOFOCUS INFO: AFC timed out at an offset of %sum" % offset)
            afc_error_message = str(error)
            failed_offsets.append(offset)

    # add an artificial delay before retrieving the AFC score
    # because, anecdotally, the score requires some time to update 
    # after the FocusDrive is moved
    # time.sleep(0.5)
    final_afc_score = af_plugin.getCurrentFocusScore()
    final_focusdrive_position = micromanager_interface.mm_core.getPosition('FocusDrive')

    # if AFC failed, move the FocusDrive back to where it was,
    # which is, at this point, the best we can do
    if afc_did_succeed:
        event_logger(
            'AUTOFOCUS INFO: AFC was called successfully at an offset of %sum '
            'and the FocusDrive position was updated from %s to %s'
            % (successful_offset, initial_focusdrive_position, final_focusdrive_position)
        )
    else:
        event_logger(
            'AUTOFOCUS ERROR: AFC timed out at all offsets and the FocusDrive will be reset to %s'
            % initial_focusdrive_position
        )

        move_z_stage(
            micromanager_interface,
            stage_label='FocusDrive',
            position=initial_focusdrive_position,
            kind='absolute'
        )

    if afc_logger is not None:
        afc_logger(
            initial_afc_score=initial_afc_score,
            final_afc_score=final_afc_score,
            final_focusdrive_position=final_focusdrive_position,
            initial_focusdrive_position=initial_focusdrive_position,
            last_afc_error_message=afc_error_message,
            failed_offsets=failed_offsets,
            afc_did_succeed=afc_did_succeed,
            position_ind=position_ind
        )

    return afc_did_succeed


def acquire_image(micromanager_interface, event_logger):
    '''
    This method just wraps _acquire_image and attempts to call it multiple times

    The motivation for this is that, on 2020-01-31, a MicroManager timeout error occurred
    during the call to mm_studio.live().snap that, while it allowed the call to return as usual,
    did not result in an image appearing in the queue, so that getLastMeta always returned None,
    and the _acquire_image method raised an uncaught TypeError that crashed the acquisition script
    '''

    data = None
    num_tries = 10
    wait_time = 10
    for _ in range(num_tries):
        try:
            data = _acquire_image(micromanager_interface)
            break
        except Exception as error:
            event_logger('ACQUIRE_IMAGE ERROR: %s' % str(error))
            time.sleep(wait_time)

    if data is None:
        event_logger('FATAL ERROR: All attempts to call _acquire_image failed')
        raise Exception('All attempts to call _acquire_image failed')
    return data


def _acquire_image(micromanager_interface):
    '''
    'snap' an image using the current laser/camera/exposure settings
    and return the image data as a numpy memmap
    '''

    # KC: not sure if this is necessary but it seems wise
    micromanager_interface.mm_core.waitForSystem()

    # number of times to try calling gate.getLastMeta()
    num_tries = 10

    # time in seconds to wait between calls to gate.getLastMeta()
    wait_time = .10

    # clear the mm2python queue
    # this ensure that gate.getLastMeta returns either None
    # or the image generated by the call to mm_studio.live().snap() below
    micromanager_interface.gate.clearQueue()

    # acquire an image using the current exposure settings
    # note that this method does not exit until the exposure is complete
    micromanager_interface.mm_studio.live().snap(True)

    # retrieve the mm2python metadata corresponding to the image acquired above
    # (this seems to require waiting for some amount of time between 30 and 100ms)
    for _ in range(num_tries):
        time.sleep(wait_time)
        meta = micromanager_interface.gate.getLastMeta()
        if meta is not None:
            break
    
    # if meta is still None, try again with a longer wait time
    # (KC: I have no reason to believe this would ever be necessary;
    # I've included it only out of an abundance of caution)
    if meta is None:
        wait_time *= 10
        for _ in range(num_tries):
            time.sleep(wait_time)
            meta = micromanager_interface.gate.getLastMeta()
            if meta is not None:
                break
    
    # if meta is still None, we're in big trouble
    if meta is None:
        raise TypeError('The meta object returned by gate.getLastMeta() is None')

    data = np.memmap(
        meta.getFilepath(),
        dtype='uint16',
        mode='r+',
        offset=0,
        shape=(meta.getxRange(), meta.getyRange())
    )
    return data


def acquire_stack(
    micromanager_interface,
    stack_settings, 
    channel_ind,
    position_ind, 
    position_name,
    event_logger
):
    '''
    Acquire a z-stack using the given settings and 'put' it in the datastore object

    This method results in the creation (via datastore.putImage)
    of a single TIFF stack with a filename of the form
    'MMSTack_{position_name}.ome.tif'

    Parameters
    ----------
    stack_settings : 
    channel_ind : int
        a position-unique channel index (usually 0 for hoechst an 1 for GFP)
    position_ind : int
        the experiment-unique position index
    position_name : str
        an arbitrary but experiment-unique name for the current position,
        used to determine the filename of the TIFF stack
    
    Context
    -------
    The MicroManager API calls that acquire and 'save' an image at each z-slice 
    are based on those that appear in the MicroManager v2 beanshell scripts. 
    The relevant block from these scripts is copied verbatim below for reference. 
    ```
    mmc.snapImage();
    tmp1 = mmc.getTaggedImage();
    Image channel0 = mm.data().convertTaggedImage(tmp1);
    channel0 = channel0.copyWith(
        channel0.getCoords().copy().channel(c).z(z).stagePosition(p).build(),
        channel0.getMetadata().copy().positionName(""+p).build());
    autosavestore.putImage(channel0);
    ```
    '''

    def snap_and_get_image(delay=0):
        '''
        wrapper to try this block multiple times, in an attempt to catch a camera hardware error
        '''
        # acquire an image
        micromanager_interface.mm_core.waitForImageSynchro()
        micromanager_interface.mm_core.snapImage()

        # optional wait time between snapImage and getTaggedImage calls
        if delay > 0:
            time.sleep(delay)

        # convert the image
        # TODO: understand what's happening here
        tagged_image = micromanager_interface.mm_core.getTaggedImage()
        image = micromanager_interface.mm_studio.data().convertTaggedImage(tagged_image)
        return image    

    # generate a list of the z positions to visit
    z_positions = np.arange(
        stack_settings.relative_bottom, 
        stack_settings.relative_top + stack_settings.step_size, 
        stack_settings.step_size
    )

    for z_ind, z_position in enumerate(z_positions):

        # move to the new z-position 
        move_z_stage(
            micromanager_interface, 
            stack_settings.stage_label, 
            position=z_position, 
            kind='absolute'
        )

        # this is an attempt to recover from the 'camera image buffer read failed' error
        # that is randomly and rarely thrown by the `getTaggedImage` call
        image = None
        num_tries = 10
        intertry_wait_time = 3
        intratry_wait_time = 0
        for _ in range(num_tries):
            try:
                image = snap_and_get_image(intratry_wait_time)
                break
            except Exception as error:
                event_logger(
                    'ERROR: An error occurred in snap_and_get_image with a delay of %ss: %s'
                    % (intratry_wait_time, str(error))
                )
                time.sleep(intertry_wait_time)
                intratry_wait_time += 1

        if image is None:
            message = 'All tries to call snap_and_get_image failed'
            event_logger('FATAL ERROR: %s' % message)
            raise Exception(message)

        # manually construct image coordinates (position, channel, z)
        # NOTE: a new TIFF stack will be created whenever a new and datastore-unique value
        # is passed to coords.stagePosition (this value must, however, be an int)
        coords = image.getCoords().copy()
        coords = coords.channel(channel_ind)
        coords = coords.z(z_ind)
        coords = coords.stagePosition(position_ind)
        coords = coords.build()

        # construct image metadata
        # NOTE: the filename of the TIFF stack is determined entirely 
        # by the value passed to metadata.positionName (and this value can be any string)
        metadata = image.getMetadata().copy()
        metadata = metadata.positionName(position_name)
        metadata = metadata.build()

        image = image.copyWith(coords, metadata)
        if micromanager_interface.has_open_datastore:
            micromanager_interface.datastore.putImage(image)

    # cleanup: reset the piezo stage
    # TODO: decide if this is necessary
    move_z_stage(micromanager_interface, stack_settings.stage_label, position=0.0, kind='absolute')


def change_channel(micromanager_interface, channel_settings):
    '''
    Convenience method to set the laser power, exposure time, and camera gain
    
    (KC: pretty sure the order of these operations doesn't matter,
    but to be safe the order here is preserved from Nathan's script)
    '''
    mm_core = micromanager_interface.mm_core

    # hardware config (this takes some time)
    mm_core.setConfig(channel_settings.config_group, channel_settings.config_name)
    mm_core.waitForConfig(channel_settings.config_group, channel_settings.config_name)

    # laser power
    if channel_settings.laser_line is not None:
        mm_core.setProperty(
            channel_settings.laser_line,
            channel_settings.laser_name,
            channel_settings.laser_power
        )

    # exposure time
    mm_core.setExposure(float(channel_settings.exposure_time))

    # camera gain
    property_name = 'Gain'
    mm_core.setProperty(channel_settings.camera_name, property_name, channel_settings.camera_gain)


def move_z_stage(micromanager_interface, stage_label, position=None, kind=None):
    '''
    Convenience method to move a z-stage
    TODO: basic sanity checks on the value of `position`
    (e.g., if kind=='relative', `position` shouldn't be a 'big' number)
    '''

    if kind not in ['relative', 'absolute']:
        raise ValueError("`kind` must be either 'relative' or 'absolute'")

    try:
        position = float(position)
    except ValueError:
        raise TypeError('`position` cannot be coerced to float')
    
    if not np.isfinite(position):
        raise TypeError('`position` cannot be nan')
    
    # move the stage
    if kind == 'absolute':
        micromanager_interface.mm_core.setPosition(stage_label, position)
    elif kind == 'relative':
        micromanager_interface.mm_core.setRelativePosition(stage_label, position)
    
    micromanager_interface.mm_core.waitForDevice(stage_label)


def autoexposure(
    micromanager_interface,
    stack_settings, 
    autoexposure_settings,
    channel_settings,
    event_logger
):
    '''

    Parameters
    ----------
    stack_settings : an instance of StackSettings
    autoexposure_settings : an instance of AutoexposureSettings
    channel_settings : the ChannelSettings instance 
        corresponding to the channel on which to run the autoexposure algorithm
        NOTE: this method modifies the `laser_power` and `exposure_time` attributes

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
        if no slices were over-exposed, check for under-exposure 
        using the overall max intensity and lower the exposure time if necessary

    '''
    
    autoexposure_did_succeed = True

    # keep track of the maximum intensity
    stack_max_intensity = 0
    
    # keep track of whether any slices were ever over-exposed
    overexposure_did_occur = False

    # start at the bottom of the stack
    z_position = stack_settings.relative_bottom

    # step through the z-stack and check each slice for over-exposure
    while z_position <= stack_settings.relative_top:

        # move to the next z-position 
        # (either the next slice or the bottom of the stack)
        move_z_stage(
            micromanager_interface, 
            stack_settings.stage_label, 
            position=z_position,
            kind='absolute'
        )

        # acquire an image and check the exposure
        image = acquire_image(micromanager_interface, event_logger)

        # use a percentile to calculate the 'max' intensity 
        # as a defense against hot pixels, anomalous bright spots/dust, etc
        # (the 99.99th percentile corresponds to ~100 pixels in a 1024x1024 image)
        slice_max_intensity = np.percentile(image, 99.99)
        event_logger(
            'AUTOEXPOSURE INFO: max_intensity = %d at z = %0.1f'
            % (slice_max_intensity, z_position)
        )

        # if the slice was over-exposed, lower the exposure time or the laser power,
        # reset stack_max_intensity, and go back to the bottom of the z-stack
        slice_was_overexposed = slice_max_intensity > autoexposure_settings.max_intensity
        if slice_was_overexposed:
            overexposure_did_occur = True

            # lower the exposure time
            channel_settings.exposure_time *= autoexposure_settings.relative_exposure_step
            event_logger(
                'AUTOEXPOSURE INFO: The slice at z = %0.1f was overexposed (max = %d) '
                'so the exposure time was reduced to %dms'
                % (z_position, slice_max_intensity, channel_settings.exposure_time)
            )

            # if the exposure time is now too low, turn down the laser instead
            if channel_settings.exposure_time < autoexposure_settings.min_exposure_time:
                channel_settings.exposure_time = autoexposure_settings.default_exposure_time
                channel_settings.laser_power *= autoexposure_settings.relative_exposure_step
                event_logger(
                    'AUTOEXPOSURE INFO: The minimum exposure time was exceeded '
                    'so the laser power was reduced to %0.1f%%'
                    % (channel_settings.laser_power)
                )

                # update the laser power
                micromanager_interface.mm_core.setProperty(
                    channel_settings.laser_line,
                    channel_settings.laser_name,
                    channel_settings.laser_power
                )

            # update the exposure time
            micromanager_interface.mm_core.setExposure(float(channel_settings.exposure_time))

            # prepare to return to the bottom of the stack
            z_position = stack_settings.relative_bottom

            # reset the max intensity
            stack_max_intensity = 0

            # break out of the while loop if the exposure has been lowered
            # as far as it can be and the slice is still over-exposed
            # KC: in practice, I believe this should rarely/never happen
            if channel_settings.laser_power < autoexposure_settings.min_laser_power:
                autoexposure_did_succeed = False
                event_logger(
                    'AUTOEXPOSURE ERROR: The laser power was lowered to its minimum '
                    'but the stack was still over-exposed'
                )
                break

        # if the slice was not over-exposed, 
        # update stack_max and move to the next z-slice
        else:
            stack_max_intensity = max(stack_max_intensity, slice_max_intensity)
            z_position += autoexposure_settings.z_step_size
    

    # after exiting the while-loop, either
    # 1) some slices were over-exposed and the exposure is now adjusted, or
    # 2) no slices were over-exposed and we need to check for under-exposure
    # here, we check for scenario (2) and use stack_max_intensity to increase
    # the exposure time if it is too low
    if not overexposure_did_occur:
        intensity_ratio = autoexposure_settings.min_intensity / stack_max_intensity
        if intensity_ratio > 1:
            channel_settings.exposure_time *= intensity_ratio
            event_logger(
                'AUTOEXPOSURE INFO: The stack was under-exposed (max = %d) '
                'so the exposure time was increased by %0.1fx to %dms'
                % (stack_max_intensity, intensity_ratio, channel_settings.exposure_time)
            )

            if channel_settings.exposure_time > autoexposure_settings.max_exposure_time:
                channel_settings.exposure_time = autoexposure_settings.max_exposure_time
                event_logger(
                    'AUTOEXPOSURE INFO: The stack was under-exposed '
                    'and the maximum exposure time was exceeded'
                )

    # reset the piezo stage
    move_z_stage(micromanager_interface, stack_settings.stage_label, position=0.0, kind='absolute')

    # log the final results
    event_logger(
        'AUTOEXPOSURE INFO: The final stack max is %d, the laser power is %0.1f%%, '
        'and the exposure time is %dms'
        % (stack_max_intensity, channel_settings.laser_power, channel_settings.exposure_time)
    )

    return autoexposure_did_succeed

