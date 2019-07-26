# ML Group automated plate acquisition script:
#
# Python version of MultiChannelwAutoExposure_AFS script from MicroManager. This script
# is to be used in MicroManager2 with Bryant's MM2Python bridge.  Within this script, one
# can both call MicroManager commands and Python functions, enabling on-the-fly QC
# and feature detection.
#
# Important functionality includes optimizing laser power and exposure for the GFP channel
# to account for different proteins in each well.  This optimization is done only for the first
# position in a well.  Another function is pre-filtering fields of view based on confluency, only
# capturing fields of view of optimal confluency (implemented in version 2).
#
# Version 1:
# Initial transfer from Beanshell script to Python, excluding GFP image condition calibration
#
# Version 2:
# Implement GFP image condition calibration, as well as laying infrastructure to implement SpreadTest
#
# Version 3:
# Implement SpreadTest
#
# Version 4:
# To try to stretch to imaging at edges, updates Autofocusing to account for if starting out of AFC range,
# trying to find using OughtaFocus, then trying AFC again.  Also, trying to adjust SpreadTest to find the
# true middle of the stack using blur calculations

from py4j.java_gateway import JavaGateway
import numpy as np
import os
from skimage import filters

# Save parameters
autoSave = True
# autoSave_path = "D:/NC/ML0147_20190621/"
autoSave_path = "D:/NC/MM2Py_AFCTest/"
prefix = "mNG96wp1_scriptedredo_"
num = 0
while(os.path.isdir(autoSave_path + prefix + str(num))):
    num += 1
fullpath = autoSave_path + prefix + str(num)


# Image parameters, order of each list must follow order of Channels list
Channels = [
    "EMCCD_Confocal40_DAPI", # DAPI must be first channel for SpreadTest
    "EMCCD_Confocal40_GFP"
]

Lasers = [
    "Laser 405-Power Setpoint",
    "Laser 488-Power Setpoint"
]

LaserPowers = [
    10,
    10
]

Exposures = [ # Must be double
    50.0,
    50.0
]

Gain = [ # Must be double
    400.0,
    400.0
]

ZStack = [
    -6.0,    # Z-stack begin, relative
    16.0,   # Z-stack end, relative
    0.2     # Z-stack step
]

# Hardware settings
zdevice = "PiezoZ"
xydevice = "XYStage"
config_group = "Channels-EMCCD"
camera = "Andor EMCCD"
laser_line = "Andor ILE-A"

######################
# No edits past here #
######################


def main():
    # Create objects to bridge MicroManager and Python
    gateway = JavaGateway()
    gate = gateway.entry_point

    mmc = gate.getCMMCore()
    mm = gate.getStudio()

    # Set up acquisition
    tempLP = LaserPowers
    tempExp = Exposures
    mmc.setExposure(float(tempExp[0]))
    tempGain = Gain

    # width = mmc.getImageWidth()
    # height = mmc.getImageHeight()
    # bytes = mmc.getBytesPerPixel()
    # depth = mmc.getImageBitDepth()
    # timelapse = 1

    # Create datastore to save to
    autosavestore = mm.data().createMultipageTIFFDatastore(fullpath, True, True)
    mm.displays().createDisplay(autosavestore)

    # Set autofocus
    af_manager = mm.getAutofocusManager()
    af_manager.setAutofocusMethodByName("Adaptive Focus Control")
    af_plugin = af_manager.getAutofocusMethod()

    # Image sync
    mmc.assignImageSynchro(zdevice)
    mmc.assignImageSynchro(xydevice)
    mmc.assignImageSynchro(mmc.getShutterDevice())
    mmc.assignImageSynchro(mmc.getCameraDevice())
    mmc.setAutoShutter(True)

    pl = mm.getPositionList()
    good_stacks = 0
    newWell = True
    position = 0
    position_log = {}

    # Position loop
    for p in range(pl.getNumberOfPositions()):
        # Reset PiezoZ
        mmc.setPosition(zdevice, 0.0)

        # Move to next position
        nextPosition = pl.getPosition(p)
        nextPosition.goToPosition(nextPosition, mmc)

        # Reset good position counter if at first position of a new well
        if "Site_0" in nextPosition.getLabel() or "Pos_000_000" in nextPosition.getLabel():
            good_stacks = 0
            newWell = True

        # Check if enough good stacks have been acquired for this current well
        if good_stacks < 8:
            # Autofocus
            print("Focusing...")
            mmc.setConfig(config_group, Channels[0])
            mmc.waitForConfig(config_group, Channels[0])
            mmc.setExposure(float(tempExp[0]))
            mmc.setProperty(laser_line, Lasers[0], tempLP[0])
            mmc.setProperty(camera, "Gain", tempGain[0])
            try:
                af_plugin.fullFocus()
            except:
                # If AFC fails, try getting closer with optical autofocus ("OughtFocus")
                af_manager.setAutofocusMethodByName("OughtaFocus")
                try:
                    af_plugin.fullFocus()
                except:
                    print("OughtaFocus failed")
                # Reset to AFC and try again
                af_manager.setAutofocusMethodByName("Adaptive Focus Control")
                try:
                    af_plugin.fullFocus()
                except:
                    print("AFC FAIL")
            mmc.waitForSystem()
            focalPlane = mmc.getPosition(zdevice)
            curPos = focalPlane
            print("Found focus.")
            mmc.waitForSystem()

            if spread_test(get_snap_data(mm, gate), mmc, zdevice):
                print("Good confluency")
                good_stacks += 1
                position_log['Pos' + str(position)] = 'Good confluency'

                # Move to bottom of stack
                print("Moving the stage to: " + str(ZStack[0]))
                move_z(mmc, zdevice, focalPlane)
                floor = move_z_relative(mmc, zdevice, ZStack[0])

                # Channel loop
                for c in range(len(Channels)):
                    print("Now imaging:" + Channels[c])
                    mmc.setConfig(config_group, Channels[c])
                    mmc.setProperty(laser_line, Lasers[c], tempLP[c])
                    mmc.setExposure(float(tempExp[c]))
                    mmc.setProperty(camera, "Gain", tempGain[c])
                    z = 0
                    curPos = mmc.getPosition(zdevice)

                    # KC NOTE: this is shorthand for 'run_autoexposure_routine_again'
                    checkExp = True

                    # KC NOTE: 'overExp' is shorthand for at_least_one_slice_overexposed,
                    # and it is inverted - it is initially True and set to False
                    # when at least one slice is found to be overexposed
                    overExp = True

                    checkExpCount = 1
                    overall_max = 0
                    total_high_pixels = 0

                    # Z-position loop
                    while curPos <= focalPlane + ZStack[1]:
                        # If channel is GFP and all checks are still active, run autoexposure check
                        if checkExp and "_GFP" in Channels[c] and newWell:
                            # Reset exposures at beginning of new well
                            if checkExpCount == 1 and newWell:
                                # print("GFP imaging condition check, position:", str(curPos))
                                tempExp = Exposures
                                tempLP = LaserPowers

                            # Get max of current slice
                            mmc.waitForSystem()
                            dat = get_snap_data(mm, gate)
                            im_max = np.max(dat)
                            if im_max > overall_max:
                                overall_max = im_max

                            # Check for over-exposure on single slice, adjust if over-exposed
                            overExp, newExp, newLP, total_high_pixels = slice_assessment(dat, im_max, total_high_pixels, overExp, tempExp[c], tempLP[c])
                            if overExp == False:
                                print("Slice overexposed")
                                tempLP[c] = newLP
                                tempExp[c] = newExp
                                mmc.setProperty(laser_line, Lasers[c], tempLP[c])
                                mmc.setExposure(float(tempExp[c]))

                            # Move to next z-position
                            curPos = move_z_relative(mmc, zdevice, ZStack[2])

                            # At the end of the stack, make overall exposure checks and calibrations
                            if curPos > (focalPlane + ZStack[1]):
                                print("Checking exposure...")
                                if overExp == False:  # If a single slice was over-exposed, have to re-do
                                    checkExp = True
                                    overExp = True
                                    overall_max = 0
                                    total_high_pixels = 0
                                else:
                                    checkExp, newLP, newExp, overall_max = stack_assessment(overall_max, tempExp[c], tempLP[c], total_high_pixels)
                                    if checkExp: # Only situation in which exposure must be re-checked is if laser power has changed
                                        print("Resetting for new laser power")
                                        tempLP[c] = newLP
                                        mmc.setProperty(laser_line, Lasers[c], tempLP[c])
                                        overExp = True
                                        overall_max = 0
                                        total_high_pixels = 0
                                    else:
                                        newWell = False
                                    tempExp[c] = newExp

                                # Set exposure to new optimized exposure, return to bottom of the stack
                                # KC NOTE: the line below is incorrect, because bottom_z_position (`floor`) is absolute,
                                # so there should be no need to add initial_z_position (what nathan calls `focalPlane`).
                                # It is not clear why this bug does not cause the while-loop to break before the GFP stack
                                # is acquired. 
                                # KC UPDATE: we think that this bug has no effect only because focalPlane is always zero, 
                                # which is true because the z-position of the PiezoZ device is reset to zero,
                                # in this script, at the beginning of every new position.
                                # This explanation assumes that PiezoZ is unaffected by the call to Position.goToPosition(),
                                # which we believe to be true by inspecting the device names in the JSON files of positions 
                                # (that is, goToPosition controls the motorized z-stage, not the Piezo stage)
                                mmc.setExposure(float(tempExp[c]))
                                curPos = move_z(mmc, zdevice, focalPlane + floor)
                                checkExpCount += 1
                                print("Exposure now " + str(tempExp[c]))

                        else: # This actually acquires data and saves it
                            mmc.waitForImageSynchro()
                            mmc.snapImage()
                            tmp1 = mmc.getTaggedImage()
                            channel0 = mm.data().convertTaggedImage(tmp1)
                            channel0 = channel0.copyWith(
                                channel0.getCoords().copy().channel(c).z(z).stagePosition(p).build(),
                                channel0.getMetadata().copy().positionName("" + str(p)).build()
                            )
                            autosavestore.putImage(channel0)

                            # Move to next z-position
                            curPos = move_z_relative(mmc, zdevice, ZStack[2])
                            z += 1

                    move_z(mmc, zdevice, floor)

                move_z(mmc, zdevice, focalPlane)

            else: # Confluency at this position is poor, moving to next position
                print("Poor confluency, skipping position.")
                position_log['Pos' + str(position)] = 'Poor confluency'

        position += 1

    autosavestore.freeze()

# Gets snap of current slice and returns np.memmap of it
def get_snap_data(mm, gate):
    # Snaps and writes to snap/live view
    mm.live().snap(True)

    # Retrieve data from memory mapped files, np.memmap is functionally same as np.array
    meta = gate.getLastMeta()
    dat = np.memmap(meta.getFilepath(), dtype="uint16", mode='r+', offset=0,
                    shape=(meta.getxRange(), meta.getyRange()))

    return dat

# Assess a single slice for over-exposure and calculates re-calibration if it is over-exposed
def slice_assessment(dat, im_max, total_high_pixels, overExp, oldExp, oldLP):
    newExp = oldExp
    newLP = oldLP
    updated_total = total_high_pixels

    if im_max >= 60620:  # If max is "high", check more thoroughly
        high_pixels = len(dat[dat > 60620])  # Count pixels between 62250 - 65535
        updated_total += high_pixels
        if 65535 in dat or high_pixels > 100:  # Adjust exposure if high_pixels > 100 or any pixel in max
            print("Over-exposed")
            newExp = 0.8 * oldExp
            if newExp < 30:  # If exposure is too low, scale down laser power and reset exposure
                newLP = 0.8 * oldLP
                newExp = 50
            else:
                newLP = oldLP
            overExp = False

    return overExp, newExp, newLP, updated_total

# Assess the whole stack for over-exposure and calculates re-calibration if it is over-exposed
def stack_assessment(overall_max, oldExp, oldLP, total_high_pixels):
    newLP = oldLP
    newExp = oldExp
    checkExp = False

    if overall_max < 45000:  # Scale exposure relative to the ratio of optimal if max is not high
        print("Scaling")
        newExp = float(oldExp) * 45000 / float(overall_max)

    if newExp > 500:  # Do not exceed 500ms exposures
        print("Maximum Exposure")
        newExp = 500
    if newExp < 30 or total_high_pixels > 1000:  # Scale laser power if exposure is less than 30 or too many high pixels
        print("Scaling laser power")
        newLP = 0.8 * oldLP
        newExp = 50
        checkExp = True

    return checkExp, newLP, newExp, overall_max

# Quick command to move FocusDrive to absolute position
def move_z(mmc, zdevice, newZ):
    mmc.setPosition(zdevice, newZ)
    mmc.waitForDevice(zdevice)
    curPos = mmc.getPosition(zdevice)
    return curPos

# Quick command to move FocusDrive relative to the current position
def move_z_relative(mmc, zdevice, offset):
    mmc.setRelativePosition(zdevice, offset)
    mmc.waitForDevice(zdevice)
    curPos = mmc.getPosition(zdevice)
    return curPos

# Test for good confluency overall and spread of cells
def spread_test(slice, mmc, zdevice):
    # Set thresholds for making confluency decisions
    global_lower_confluence_threshold = 15
    global_upper_confluence_threshold = 46
    tile_lower_confluence_threshold = 10
    tile_upper_confluence_threshold = 50
    total_image_pixels = float(1024.0 * 1024.0)
    slice_factor = 256
    sub_image_pixels = float(slice_factor * slice_factor)
    false_counter = 0

    # TODO: Find middle using blur calculation
    afc_point = mmc.getPosition(zdevice)
    move_z_relative(mmc, zdevice, -3.0)
    for i in range(13):
            
        move_z_relative(mmc, zdevice, 0.5)

    # How to do this?  We're at the AFC point, so we need to explore +/- 3 um
    # Find max blur
    # Check stuff

    if overall_confluency(slice, total_image_pixels, global_lower_confluence_threshold, global_upper_confluence_threshold):
        addition_x = 0
        addition_y = 0

        # Splitting image into 16 256 x 256 tiles and testing confluency on each tile
        for n in range(16):
            sub_image = np.empty((slice_factor, slice_factor))
            for i in range(slice_factor):
                for j in range(slice_factor):
                    sub_image[i][j] = slice[i + addition_x][j + addition_y]

            # Actual confluency check for single tile
            sub_image_confluency = overall_confluency(sub_image, sub_image_pixels, tile_lower_confluence_threshold, tile_upper_confluence_threshold)
            if sub_image_confluency == False:
                false_counter += 1

            # Updates values for tiling
            if n == 3 or n == 7 or n == 11:
                addition_y += slice_factor
                addition_x = 0
            else:
                addition_x += slice_factor

        if false_counter > 4:
            print("Bad spread")
            return False
        else:
            return True
    else:
        print("Overall confluency fail")
        return False

# Confluency check for image within user-given thresholds
def overall_confluency(mid_image, total_pixels, lower_confluence_threshold, upper_confluence_threshold):
    # Using only middle slice, apply Gaussian filter
    filtered_mid_image = filters.gaussian(mid_image)

    # Threshold between background and non-background using Li Thresholding
    val = filters.threshold_li(filtered_mid_image)

    # Compute percentage of non-background pixels
    blue_pixels = len(filtered_mid_image[filtered_mid_image >= val])
    blue_pixel_percentage = (blue_pixels / total_pixels) * 100.0
    # print("Blue pixels:", str(blue_pixels))
    # print("Confluency:", str(blue_pixel_percentage))

    if blue_pixel_percentage >= lower_confluence_threshold and \
            blue_pixel_percentage <= upper_confluence_threshold:
        return True
    else:
        return False


if __name__ == "__main__":
    main()
