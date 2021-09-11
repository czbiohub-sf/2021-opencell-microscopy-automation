import os
import re
import py4j
import json
import skimage
import numpy as np
import pandas as pd

from scipy import interpolate
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D as ax3

from dragonfly_automation import microscope_operations, utils


def find_nearest_well(mm_core, position_list):
    '''
    '''
    # current xy stage position
    current_pos = mm_core.getXPosition('XYStage'), mm_core.getYPosition('XYStage')

    # find the well closest the current position
    dists = []
    for ind, p in enumerate(position_list['POSITIONS']):
        xystage = [d for d in p['DEVICES'] if d['DEVICE'] == 'XYStage'][0]
        dist = np.sqrt(((np.array(current_pos) - np.array([xystage['X'], xystage['Y']]))**2).sum())
        dists.append(dist)
        
    ind = np.argmin(dists)
    well_id, site_num = utils.parse_hcs_site_label(position_list['POSITIONS'][ind]['LABEL'])
    print(
        'Nearest position is in well %s (ind = %d and distance = %d)'
        % (well_id, ind, min(dists))
    )



class StageVisitationManager:

    def __init__(self, micromanager_interface, well_ids_to_visit, position_list):
        '''
        '''
        self.well_ids_to_visit = well_ids_to_visit
        self.position_list = position_list
        self.micromanager_interface = micromanager_interface

        # the index of the current well in well_ids_to_visit
        self.current_ind = -1

        # initialize a dict, keyed by well_id, of the measured FocusDrive positions
        self.measured_focusdrive_positions = {}


    def go_to_next_well(self):
        self.current_ind = min(self.current_ind + 1, len(self.well_ids_to_visit) - 1)
        self._go_to_position()


    def go_to_previous_well(self):
        self.current_ind = max(0, self.current_ind - 1)
        self._go_to_position()


    def _go_to_position(self):
        self.current_well_id = self.well_ids_to_visit[self.current_ind]

        position_ind = self._get_current_position_ind()
        if position_ind is None:
            print('Error: the next well, %s, is not in the position list' % self.current_well_id)
            return

        print('Going to well %s' % self.current_well_id)
        if self.current_well_id == self.well_ids_to_visit[-1]:
            print('Warning: this is the last well in the list of wells to visit')

        # this try-except catches timeout errors triggered by large stage movements
        # (these errors are harmless)
        try:
            microscope_operations.go_to_position(self.micromanager_interface, position_ind)
        except py4j.protocol.Py4JJavaError:
            microscope_operations.go_to_position(self.micromanager_interface, position_ind)    
        print('Arrived at well %s' % self.current_well_id)


    def _get_current_position_ind(self):
        '''
        Get the index of the first position in the current well
        '''
        current_ind = None
        for ind, position in enumerate(self.position_list['POSITIONS']):
            position_well_id, position_site_num = utils.parse_hcs_site_label(position['LABEL'])
            if position_well_id == self.current_well_id:
                current_ind = ind
                break
        return current_ind


    def call_afc(self):
        '''
        call AFC (if it is in-range) and insert the updated FocusDrive position
        in the list of measured focusdrive positions
        '''
        print('Attempting to call AFC at well %s' % self.current_well_id)

        pos_before = self.micromanager_interface.mm_core.getPosition('FocusDrive')
        self.micromanager_interface.mm_core.fullFocus()
        pos_after = self.micromanager_interface.mm_core.getPosition('FocusDrive')

        self.measured_focusdrive_positions[self.current_well_id] = pos_after
        print('FocusDrive position before AFC: %s' % pos_before)
        print('FocusDrive position after AFC: %s' % pos_after)


def preview_interpolation(
    measured_focusdrive_positions, 
    top_left_well_id, 
    bottom_right_well_id,
    method='cubic'
):
    '''
    '''

    positions = []
    for well_id, zpos in measured_focusdrive_positions.items():
        positions.append((*utils.well_id_to_position(well_id), zpos))
    positions = np.array(positions)

    top_left_x, top_left_y = utils.well_id_to_position(top_left_well_id)
    bot_right_x, bot_right_y = utils.well_id_to_position(bottom_right_well_id)

    x = np.linspace(top_left_x, bot_right_x, 50)
    y = np.linspace(top_left_y, bot_right_y, 50)
    X, Y = np.meshgrid(x, y)
    Z = interpolate.griddata(positions[:, :2], positions[:, 2], (X, Y), method=method)

    plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    ax.scatter3D(positions[:, 0], positions[:, 1], positions[:, 2], color='red')


def interpolate_focusdrive_positions(
    position_list_filepath, 
    measured_focusdrive_positions, 
    top_left_well_id,
    bottom_right_well_id,
    method='cubic',
    offset=0
):
    '''

    Parameters
    ----------
    position_list_filepath: str
        Local path to a JSON list of positions generated by the HCS Site Generator plugin
    measured_focusdrive_positions : a dict of well_ids and measured FocusDrive positions
        e.g., {'B9': 7600, 'B5': 7500, ...}
    top_left_well_id : the top-left-most well of the imaging region
    bottom_right_well_id : the bottom-left-most well of the imaging region
    method : specifies the method kwarg of interpolate.griddata
    offset : a constant offset (in microns) to add to the interpolated positions
    '''

    # create an array of numeric (x,y,z) positions from the well_ids
    measured_positions = []
    for well_id, zpos in measured_focusdrive_positions.items():
        measured_positions.append((*utils.well_id_to_position(well_id), zpos))
    measured_positions = np.array(measured_positions)

    with open(position_list_filepath) as file:
        position_list = json.load(file)

    for ind, position in enumerate(position_list['POSITIONS']):
        well_id, site_num = utils.parse_hcs_site_label(position['LABEL'])
        x, y = utils.well_id_to_position(well_id)

        # the interpolated z-position of the current well
        interpolated_position = interpolate.griddata(
            measured_positions[:, :2], 
            measured_positions[:, 2], 
            (x, y), 
            method=method
        )

        # add the optional user-defined constant offset
        interpolated_position = float(interpolated_position + offset)

        if pd.isna(interpolated_position):
            raise ValueError('The interpolated position is NaN in well %s' % well_id)

        # the config entry for the 'FocusDrive' device (this is the motorized z-stage)
        focusdrive_config = {
            'X': interpolated_position,
            'Y': 0,
            'Z': 0,
            'AXES': 1,
            'DEVICE': 'FocusDrive',
        }

        # remove existing FocusDrive configuration
        position['DEVICES'] = [
            config for config in position['DEVICES'] if config['DEVICE'] != 'FocusDrive'
        ]

        # append the new FocusDrive config
        position['DEVICES'].append(focusdrive_config)
        position_list['POSITIONS'][ind] = position
        
    # save the new position_list
    ext = position_list_filepath.split('.')[-1]
    new_filepath = re.sub('.%s$' % ext, '_interpolated.%s' % ext, position_list_filepath)
    with open(new_filepath, 'w') as file:
        json.dump(position_list, file)
    return new_filepath, position_list


def visualize_interpolation(measured_focusdrive_positions, new_position_list):

    def xyz_from_pos(pos):
        well_id, site_num = utils.parse_hcs_site_label(pos['LABEL'])
        focusdrive = [d for d in pos['DEVICES'] if d['DEVICE'] == 'FocusDrive'][0]
        x, y = utils.well_id_to_position(well_id)
        z = focusdrive['X']
        return x, y, z

    measured_positions = np.array(
        [
            (*utils.well_id_to_position(well_id), zpos) 
            for well_id, zpos in measured_focusdrive_positions.items()
        ]
    )

    pos = np.array([xyz_from_pos(p) for p in new_position_list['POSITIONS']])

    plt.figure()
    ax = plt.axes(projection='3d')

    ax.scatter3D(pos[:, 0], pos[:, 1], pos[:, 2], color='gray')

    ax.scatter3D(
        measured_positions[:, 0], 
        measured_positions[:, 1], 
        measured_positions[:, 2], 
        color='red'
    )


def _least_squares_interpolator(positions):
    '''
    Interpolate using least-squares fit
    (currently unused)

    This is appropriate for small regions for which it is not practical/possible
    to measure the FocusDrive position at internal (non-edge) wells
    '''
    A = np.vstack(
        [positions[:, 0], positions[:, 1], np.ones(positions.shape[0])]
    )

    # these are the z-positions we want to interpolate
    z = positions[:, 2]

    # this is the least-squares solution
    p, _, _, _ = np.linalg.lstsq(A.T, z, rcond=None)

    # this method crudely mimics the behavior of interp2d.__call__
    def interpolator(x, y):
        x = np.atleast_1d(x)
        y = np.atleast_1d(y)
        Z = np.zeros((len(y), len(x)))
        for row_ind in range(Z.shape[0]):
            for col_ind in range(Z.shape[1]):
                Z[row_ind, col_ind] = x[col_ind]*p[0] + y[row_ind]*p[1] + p[2]
        if len(x) == 1 and len(y) == 1:
            Z = Z[0]
        elif len(x) == 1:
            Z = Z[:, 0]
        elif len(y) == 1:
            Z = Z[0, :]
        return Z

    return interpolator
