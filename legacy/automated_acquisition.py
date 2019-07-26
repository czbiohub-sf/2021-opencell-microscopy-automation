'''

Automated 96-well plate acquisition
-----------------------------------

This script is a heavily edited version of Nathan's 
original automated pipeline plate acqusition script
('AutomatedPlateAcquisition_MM2_python_v4.py').

*** It is not intended to be used ***

Keith Cheveralls
July 2019


Original overview and comments from the header of Nathan's script
-----------------------------------------------------------------

    ML Group automated plate acquisition script:

    Python version of MultiChannelwAutoExposure_AFS script from MicroManager. This script
    is to be used in MicroManager2 with Bryant's MM2Python bridge.  Within this script, one
    can both call MicroManager commands and Python functions, enabling on-the-fly QC
    and feature detection.

    Important functionality includes optimizing laser power and exposure for the GFP channel
    to account for different proteins in each well.  This optimization is done only for the first
    position in a well.  Another function is pre-filtering fields of view based on confluency, only
    capturing fields of view of optimal confluency (implemented in version 2).

    Version 1:
    Initial transfer from Beanshell script to Python, excluding GFP image condition calibration

    Version 2:
    Implement GFP image condition calibration, as well as laying infrastructure to implement SpreadTest

    Version 3:
    Implement SpreadTest

    Version 4:
    To try to stretch to imaging at edges, updates Autofocusing to account for if starting out of AFC range,
    trying to find using OughtaFocus, then trying AFC again.  Also, trying to adjust SpreadTest to find the
    true middle of the stack using blur calculations

'''


import os
import sys
import numpy as np
from skimage import filters

# --------------------------------------------------------------------------
#
# This script is meant to be run in test/dev mode only!
# For the relative path below to work, it must be run from this directory
#
ENV = 'test'
sys.path.insert(0, '../')
from automated_microscopy.gateway import gateway_utils
print('WARNING: automated_acquisition is using the mock gateway')
#
# --------------------------------------------------------------------------


# Local directory to which to save the TIFF stacks
AUTOSAVE_PATH = "D:/NC/MM2Py_AFCTest/"

# Channel settings
# NOTE: exposure_time and camera_gain must be floats
DAPI_CONFIG = {
    'channel_name': 'EMCCD_Confocal40_DAPI',
    'laser_name': 'Laser 405-Power Setpoint',
    'laser_power': 10,
    'exposure_time': 50.0,
    'camera_gain': 400.0,
}

GFP_CONFIG = {
    'channel_name': 'EMCCD_Confocal40_GFP',
    'laser_name': 'Laser 488-Power Setpoint',
    'laser_power': 10,
    'exposure_time': 50.0,
    'camera_gain': 400.0,
}

# Hardware settings
ZDEVICE = "PiezoZ"
LASER_LINE = 'Andor ILE-A'

HARDWARE_CONFIG = {
    'xydevice': "XYStage",
    'config_group': "Channels-EMCCD",
    'camera_name': "Andor EMCCD",
}

# start/end z-positions for the stack (relative to the AFC point)

if ENV=='test':
    ZSTACK_STEP_SIZE = 2
    ZSTACK_REL_START = -3
    ZSTACK_REL_END = 1

if ENV=='prod':
    ZSTACK_REL_START = -6.0
    ZSTACK_REL_END = 16.0
    ZSTACK_STEP_SIZE = 0.2

# min/max/default exposure times (in milliseconds)
MIN_EXPOSURE_TIME = 30
MAX_EXPOSURE_TIME = 500
DEFAULT_EXPOSURE_TIME = 50



def main():

    gate, mm_studio, mm_core = gateway_utils.get_gate(env='test')

    # NC: Set up acquisition
    temp_laser_power = [DAPI_CONFIG['laser_power'], GFP_CONFIG['laser_power']]
    temp_exposure_time = [DAPI_CONFIG['exposure_time'], GFP_CONFIG['exposure_time']]

    mm_core.setExposure(float(temp_exposure_time[0]))

    # width = mm_core.getImageWidth()
    # height = mm_core.getImageHeight()
    # bytes = mm_core.getBytesPerPixel()
    # depth = mm_core.getImageBitDepth()

    # NC: Create datastore to save to
    if ENV=='prod':
        autosavestore = mm_studio.data().createMultipageTIFFDatastore(AUTOSAVE_PATH, True, True)
        mm_studio.displays().createDisplay(autosavestore)

    # NC: Set autofocus
    af_manager = mm_studio.getAutofocusManager()
    af_manager.setAutofocusMethodByName("Adaptive Focus Control")
    af_plugin = af_manager.getAutofocusMethod()

    # NC: Image sync
    mm_core.assignImageSynchro(ZDEVICE)
    mm_core.assignImageSynchro(HARDWARE_CONFIG['xydevice'])
    mm_core.assignImageSynchro(mm_core.getShutterDevice())
    mm_core.assignImageSynchro(mm_core.getCameraDevice())
    mm_core.setAutoShutter(True)

    position_list = mm_studio.getPositionList()
    num_good_stacks = 0
    new_well_flag = True


    # --------------------------------------------------------------------------------
    #
    # Loop over positions
    #
    # --------------------------------------------------------------------------------
    for position_index in range(position_list.getNumberOfPositions()):

        print('----------------------------------- Position %d -----------------------------------' % position_index)

        # NC: Reset PiezoZ
        mm_core.setPosition(ZDEVICE, 0.0)

        # NC: Move to next position
        current_position = position_list.getPosition(position_index)
        current_position.goToPosition(current_position, mm_core)

        # NC: Reset good position counter if at first position of a new well
        # KC: new_well_flag will stay true until the autoexposure routine succeeds,
        # which may not occur on the first FOV
        label = current_position.getLabel()
        if "Site_0" in label or "Pos_000_000" in label:
            num_good_stacks = 0
            new_well_flag = True

        # NC: Check if enough good stacks have been acquired for this current well
        if num_good_stacks >= 8:
            continue


        # --------------------------------------------------------------------------------
        #
        # Autofocus
        # KC TODO: if both autofocus methods fail, shouldn't we note the failure 
        # and move to the next position?
        #
        # --------------------------------------------------------------------------------
        # NC: Autofocus using DAPI
        print("Focusing using DAPI channel...")
        mm_core.setConfig(HARDWARE_CONFIG['config_group'], DAPI_CONFIG['channel_name'])
        mm_core.waitForConfig(HARDWARE_CONFIG['config_group'], DAPI_CONFIG['channel_name'])

        mm_core.setExposure(float(temp_exposure_time[0]))
        mm_core.setProperty(LASER_LINE, DAPI_CONFIG['laser_name'], DAPI_CONFIG['laser_power'])
        mm_core.setProperty(HARDWARE_CONFIG['camera_name'], "Gain", DAPI_CONFIG['camera_gain'])

        try:
            af_plugin.fullFocus()
        except:
            # NC: If AFC fails, try getting closer with optical autofocus ("OughtFocus")
            af_manager.setAutofocusMethodByName("OughtaFocus")
            try:
                af_plugin.fullFocus()
            except:
                print("OughtaFocus failed")

            # NC: Reset to AFC and try again
            af_manager.setAutofocusMethodByName("Adaptive Focus Control")
            try:
                af_plugin.fullFocus()
            except:
                print("AFC FAIL")

    
        # --------------------------------------------------------------------------------
        #
        # Confluency check
        #
        # --------------------------------------------------------------------------------
        mm_core.waitForSystem()
        initial_z_position = mm_core.getPosition(ZDEVICE)
        print("Found focus")

        # NC: if confluency at this position is poor, move to next position
        spread_test_passed = spread_test(get_snap_data(mm_studio, gate))

        if ENV=='test':
            spread_test_passed = True
    
        if not spread_test_passed:
            print('WARNING: confluency test failed')
            continue

        print('Confluency test passed')
        num_good_stacks += 1

        # --------------------------------------------------------------------------------
        #
        # Loop over channels and z positions
        #
        # --------------------------------------------------------------------------------
        # NC: Move to bottom of stack
        print("Moving the stage to: %0.2f" % ZSTACK_REL_START)
        move_z_absolute(mm_core, ZDEVICE, initial_z_position)
        bottom_z_position = move_z_relative(mm_core, ZDEVICE, ZSTACK_REL_START)

        # NC: Channel loop
        for ind, channel_config in enumerate([DAPI_CONFIG, GFP_CONFIG]):
            print('------------------------------ Channel %d ------------------------------' % ind)

            mm_core.setConfig(HARDWARE_CONFIG['config_group'], channel_config['channel_name'])

            # KC: set the exposure time and laser power 
            # These are modified only when the auto-exposure routine is run,
            # which occurs only for GFP and only on the first FOV of each well
    
            # initial laser power
            mm_core.setProperty(
                LASER_LINE, 
                channel_config['laser_name'], 
                temp_laser_power[ind])

            # initial exposure time
            mm_core.setExposure(float(temp_exposure_time[ind]))

            # camera gain (this is not adjusted during autoexposure)
            mm_core.setProperty(
                HARDWARE_CONFIG['camera_name'], 
                "Gain", 
                channel_config['camera_gain'])

            current_z_position = mm_core.getPosition(ZDEVICE)

            # NC: These variables are all for current way of auto-exposure
            run_autoexposure_again = True

            # KC: whether at least one slice in the stack was over-exposed
            # KC NOTE: this flag is inverted in Nathan's script, where it is initially True
            # and set to False when a slice is found to be over-exposed
            at_least_one_slice_overexposed = False

            num_exposure_checks = 1
            overall_max = 0
            total_high_pixels = 0

            # NC: Z-position loop
            zslice_count = 0
            while current_z_position <= (initial_z_position + ZSTACK_REL_END):

                print("Innermost loop at (position=%d, channel='%s', z=%0.2f)" % \
                    (position_index, channel_config['channel_name'], current_z_position))

                # --------------------------------------------------------------------------------
                #
                # If we are on a new well and on the GFP channel, run the autoexposure routine
                # KC NOTE: this routine may be run multiple times on the same FOV
                #
                # --------------------------------------------------------------------------------
                
                # NC: If channel is GFP and all checks are still active, run autoexposure check
                # KC: run_autoexposure_again is only set to False after looping over the entire stack
                # KC: new_well_flag is 
                if new_well_flag and run_autoexposure_again and "_GFP" in channel_config['channel_name']:

                    # NC: Reset exposures at beginning of new well
                    # KC: ...and *only* if this is the first time running the autoexposure routine on this FOV
                    if num_exposure_checks==1 and new_well_flag:
                        temp_laser_power = [DAPI_CONFIG['laser_power'], GFP_CONFIG['laser_power']]
                        temp_exposure_time = [DAPI_CONFIG['exposure_time'], GFP_CONFIG['exposure_time']]

                    # NC: Get max of current slice
                    mm_core.waitForSystem()
                    snap_data = get_snap_data(mm_studio, gate)
                    im_max = np.max(snap_data)
                    if im_max > overall_max:
                        overall_max = im_max

                    print('Checking slice exposure at z=%0.2f' % current_z_position)
                    slice_overexposed_flag, exposure_time, laser_power, total_high_pixels = slice_assessment(
                        temp_exposure_time[ind], 
                        temp_laser_power[ind],
                        total_high_pixels,
                        snap_data)

                    if slice_overexposed_flag:
                        print("Slice was overexposed")
                        at_least_one_slice_overexposed = True
                        temp_laser_power[ind] = laser_power
                        temp_exposure_time[ind] = exposure_time

                        mm_core.setProperty(
                            LASER_LINE, 
                            channel_config['laser_name'], 
                            temp_laser_power[ind])
            
                        mm_core.setExposure(float(temp_exposure_time[ind]))

                    # NC: Move to next z-position
                    current_z_position = move_z_relative(mm_core, ZDEVICE, ZSTACK_STEP_SIZE)

                    # NC: At the end of the stack, make overall exposure checks and calibrations
                    if current_z_position > (initial_z_position + ZSTACK_REL_END):
                        print("Checking exposure over the entire stack")

                        # NC: If at least one slice was over-exposed, have to re-do
                        # KC: it seems that 'redo' here means move back to the bottom of the stack 
                        # and run the auto-exposure routine again
                        if at_least_one_slice_overexposed:  
                            overall_max = 0
                            total_high_pixels = 0
                            at_least_one_slice_overexposed = False
                            run_autoexposure_again = True
    
                        else:
                            laser_power, exposure_time = stack_assessment(
                                temp_exposure_time[ind], temp_laser_power[ind], overall_max, total_high_pixels)

                            # NC: Only situation in which exposure must be re-checked is if laser power has changed
                            run_autoexposure_again = laser_power!=temp_laser_power[ind]

                            if run_autoexposure_again: 
                                print("Resetting for new laser power")
                                temp_laser_power[ind] = laser_power

                                mm_core.setProperty(
                                    LASER_LINE, 
                                    channel_config['laser_name'], 
                                    temp_laser_power[ind])

                                overall_max = 0
                                total_high_pixels = 0
                                at_least_one_slice_overexposed = False

                            else:
                                # KC: this is the only way new_well_flag can be set to False,
                                # and serves to stop the autoexposure routine from running again on this FOV
                                new_well_flag = False

                            temp_exposure_time[ind] = exposure_time

                        mm_core.setExposure(float(temp_exposure_time[ind]))

                        # KC NOTE: the line below deviates from Nathan's script to correct a bug
                        # (initial and bottom z positions were incorrectly added together)
                        current_z_position = move_z_absolute(mm_core, ZDEVICE, bottom_z_position)

                        num_exposure_checks += 1
                        print("Exposure time is now %0.2f" % temp_exposure_time[ind])


                # --------------------------------------------------------------------------------
                #
                # Acquire the stack
                #
                # --------------------------------------------------------------------------------
                else: 

                    if ENV=='test':
                        print('AUTOSAVESTORE: (position=%d, channel=%d, z=%d)' % (position_index, ind, zslice_count))

                    if ENV=='prod':

                        # here we use mmc, rather than mm, to get the image
                        # (this is okay because we don't want to access the image from python)
                        mm_core.waitForImageSynchro()
                        mm_core.snapImage()
                        tmp1 = mm_core.getTaggedImage()

                        # TODO possibly use bytearray to get the image data from tmp1
                        # (bryant remembers this being very slow - 100ms per MB)

                        # this line 
                        channel0 = mm_studio.data().convertTaggedImage(tmp1)

                        # assigns metadata to the snap
                        channel0 = channel0.copyWith(
                            channel0.getCoords().copy().channel(ind).z(zslice_count).stagePosition(position_index).build(),
                            channel0.getMetadata().copy().positionName(str(position_index)).build()
                        )
                        
                        # NOTE: filename is determined by the value of stagePosition (which has to be a int)

                        # autosavestore is a dict-like object keyed by the coordinates
                        autosavestore.putImage(channel0)

                    # NC: Move to next z-position
                    current_z_position = move_z_relative(mm_core, ZDEVICE, ZSTACK_STEP_SIZE)
                    zslice_count += 1

            # move back to the bottom of the stack after acquring each channel
            move_z_absolute(mm_core, ZDEVICE, bottom_z_position)

        # move back to the initial z position (which Nathan calls 'focalPlane')
        # after looping over the channels (i.e., right before moving to the next position)
        move_z_absolute(mm_core, ZDEVICE, initial_z_position)


    # after looping over all positions
    if ENV=='prod':
        autosavestore.freeze()



def get_snap_data(mm_studio, gate):
    '''
    Nathan's method to get a snap of the current slice and return an np.memmap of it

    TODO: call gate.clearQueue (when mm2python version is updated)
    '''


    # NC: Snaps and writes to snap/live view
    # KC: this is the 'right' way to *retrieve* image data from Java
    mm_studio.live().snap(True)

    # NC: Retrieve data from memory mapped file (np.memmap is functionally same as np.array)
    meta = gate.getLastMeta()

    # NOTE: there's a meta.bitDepth method

    data = np.memmap(
        meta.getFilepath(), 
        dtype="uint16", 
        mode='r+', 
        offset=0, 
        shape=(meta.getxRange(), meta.getyRange()))

    return data



def slice_assessment(exposure_time, laser_power, total_high_pixels, snap_data):
    '''
    Assess a single slice for over-exposure and calculates re-calibration if it is over-exposed
    '''

    overexposed_flag = False

    # KC TODO: where does the value 60620 come from?
    num_high_pixels = len(snap_data[snap_data > 60620])  
    total_high_pixels += num_high_pixels

    # This is Nathan's definition of 'overexposed'
    if snap_data.max()==65535 or num_high_pixels > 100:
        overexposed_flag = True

        # KC: try to lower the exposure time; if it's as low as it can be, 
        # turn down the laser power and reset the exposure time to the default
        exposure_time = 0.8 * exposure_time
        if exposure_time < MIN_EXPOSURE_TIME:
            exposure_time = DEFAULT_EXPOSURE_TIME
            laser_power = 0.8 * laser_power

    return overexposed_flag, exposure_time, laser_power, total_high_pixels



def stack_assessment(exposure_time, laser_power, overall_max, total_high_pixels):
    '''
    Nathan's method to select an exposure time and laser power
    NC: Assess the whole stack for over-exposure and calculates re-calibration if it is over-exposed
    '''
    
    # NC: Scale exposure relative to the ratio of optimal if max is not high
    if overall_max < 45000:  
        print("Increasing exposure time in stack_assessment")
        exposure_time = float(exposure_time) * 45000 / float(overall_max)

        # KC: this sghould be an inner if block, because the line above
        # is the only way the exposure_time can be increased
        if exposure_time > MAX_EXPOSURE_TIME:
            print("Clamping to maximum exposure time in stack_assessment")
            exposure_time = MAX_EXPOSURE_TIME

    # KC: exposure_time cannot be less than the minimum because 
    # this same check is performed in slice_assessment
    if exposure_time < MIN_EXPOSURE_TIME or total_high_pixels > 1000:
        print("Lowering laser power in stack_assessment")
        laser_power = 0.8 * laser_power
        exposure_time = DEFAULT_EXPOSURE_TIME

    return laser_power, exposure_time



def move_z_absolute(mm_core, zdevice, zposition):
    '''
    Move FocusDrive to an absolute position
    '''
    mm_core.setPosition(zdevice, zposition)
    mm_core.waitForDevice(zdevice)
    current_z_position = mm_core.getPosition(zdevice)
    return current_z_position


def move_z_relative(mm_core, zdevice, offset):
    '''
    Move FocusDrive to a relative position
    KC TODO: is the current_z_position returned by mm_core.getPosition relative or absolute?
    (seems like it should be absolute, unless it 'knows' that setRelativePosition was called)
    '''
    mm_core.setRelativePosition(zdevice, offset)
    mm_core.waitForDevice(zdevice)
    current_z_position = mm_core.getPosition(zdevice)
    return current_z_position



def spread_test(slice):
    '''
    Nathan's confluency test
    '''

    # NC: Set thresholds for making confluency decisions
    global_lower_confluence_threshold = 15
    global_upper_confluence_threshold = 46
    tile_lower_confluence_threshold = 10
    tile_upper_confluence_threshold = 50
    total_image_pixels = float(1024.0 * 1024.0)
    slice_factor = 256
    sub_image_pixels = float(slice_factor * slice_factor)
    false_counter = 0

    if overall_confluency(slice, total_image_pixels, global_lower_confluence_threshold, global_upper_confluence_threshold):
        addition_x = 0
        addition_y = 0

        # NC: Splitting image into 16 256 x 256 tiles and testing confluency on each tile
        for n in range(16):
            sub_image = np.empty((slice_factor, slice_factor))
            for i in range(slice_factor):
                for j in range(slice_factor):
                    sub_image[i][j] = slice[i + addition_x][j + addition_y]

            # NC: Actual confluency check for single tile
            sub_image_confluency = overall_confluency(sub_image, sub_image_pixels, tile_lower_confluence_threshold, tile_upper_confluence_threshold)
            if sub_image_confluency == False:
                false_counter += 1

            # NC: Updates values for tiling
            if n == 3 or n == 7 or n == 11:
                addition_y += slice_factor
                addition_x = 0
            else:
                addition_x += slice_factor

        if false_counter > 4:
            # print("Bad spread")
            return False
        else:
            return True
    else:
        return False



def overall_confluency(mid_image, total_pixels, lower_confluence_threshold, upper_confluence_threshold):
    '''
    Nathan's confluence check (given user-defined thresholds)
    '''
    # NC: Using only middle slice, apply Gaussian filter
    filtered_mid_image = filters.gaussian(mid_image)

    # NC: Threshold between background and non-background using Li Thresholding
    val = filters.threshold_li(filtered_mid_image)

    # NC: Compute percentage of non-background pixels
    blue_pixels = len(filtered_mid_image[filtered_mid_image >= val])
    blue_pixel_percentage = (blue_pixels / total_pixels) * 100.0
    # NC: print("Blue pixels:", str(blue_pixels))
    # NC: print("Confluency:", str(blue_pixel_percentage))

    if blue_pixel_percentage >= lower_confluence_threshold and \
            blue_pixel_percentage <= upper_confluence_threshold:
        return True
    else:
        return False


if __name__ == "__main__":
    main()
