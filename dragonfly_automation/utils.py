import os
import re
import sys
import py4j
import json
import skimage
import datetime
import numpy as np

from scipy import interpolate
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D as ax3

from dragonfly_automation import operations, utils


def timestamp():
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def to_uint8(im, percentile=0):

    dtype = 'uint8'
    max_value = 255
    im = im.copy().astype(float)

    minn, maxx = np.percentile(im, (percentile, 100 - percentile))
    if minn == maxx:
        return (im * 0).astype(dtype)

    im = im - minn
    im[im < 0] = 0
    im = im/(maxx - minn)
    im[im > 1] = 1
    im = (im * max_value).astype(dtype)
    return im


def well_id_to_position(well_id):
    '''
    'A1' to (0, 0), 'H12' to (7, 11), etc
    '''
    pattern = r'^([A-H])([0-9]{1,2})$'
    result = re.findall(pattern, well_id)
    row, col = result[0]
    row_ind = list('ABCDEFGH').index(row)
    col_ind = int(col) - 1
    return row_ind, col_ind


def parse_hcs_site_label(label):
    '''
    Parse an HCS site label
    ** copied from PipelinePlateProgram **
    '''
    pattern = r'^([A-H][0-9]{1,2})-Site_([0-9]+)$'
    result = re.findall(pattern, label)
    well_id, site_num = result[0]
    site_num = int(site_num)
    return well_id, site_num


def find_nearest_well(mmc, position_list):
    '''
    '''
    # current xy stage position
    current_pos = mmc.getXPosition('XYStage'), mmc.getYPosition('XYStage')

    # find the well closest the current position
    dists = []
    for ind, p in enumerate(position_list['POSITIONS']):
        xystage = [d for d in p['DEVICES'] if d['DEVICE'] == 'XYStage'][0]
        dist = np.sqrt(((np.array(current_pos) - np.array([xystage['X'], xystage['Y']]))**2).sum())
        dists.append(dist)
        
    ind = np.argmin(dists)
    well_id, site_num = parse_hcs_site_label(position_list['POSITIONS'][ind]['LABEL'])
    print('Nearest position is in well %s (ind = %d and distance = %d)' % (well_id, ind, min(dists)))



class StageVisitationManager:

    def __init__(self, well_ids_to_visit, position_list, mms, mmc):
        self.well_ids_to_visit = well_ids_to_visit
        self.position_list = position_list
        self.mmc = mmc
        self.mms = mms

        # generate the list of well_ids to visit and consume (via .pop())
        self.unvisited_well_ids = self.well_ids_to_visit[::-1]

        # initialize a dict, keyed by well_id, of the measured FocusDrive positions
        self.measured_focusdrive_positions = {}

    def go_to_next_well(self):
        '''
        go to the next well in the well_id list
        '''
        self.current_well_id = self.unvisited_well_ids.pop()
        ind = self.well_id_to_position_ind(self.current_well_id)
        print('Going to well %s' % self.current_well_id)

        # this try-except catches timeout errors triggered by large stage movements
        # (these errors are harmless)
        try:
            operations.go_to_position(self.mms, self.mmc, ind)
        except py4j.protocol.Py4JJavaError:
            operations.go_to_position(self.mms, self.mmc, ind)    
        print('Arrived at well %s' % self.current_well_id)

    def well_id_to_position_ind(self, well_id):
        '''
        find the index of the first position in a given well
        '''
        for ind, position in enumerate(self.position_list['POSITIONS']):
            position_well_id, position_site_num = parse_hcs_site_label(position['LABEL'])
            if position_well_id == well_id:
                break
        return ind
        
    def call_afc(self):
        '''
        call AFC (if it is in-range) and insert the updated FocusDrive position
        in the list of measured focusdrive positions
        '''
        print('Attempting to call AFC at well %s' % self.current_well_id)

        pos_before = self.mmc.getPosition('FocusDrive')
        self.mmc.fullFocus()
        pos_after = self.mmc.getPosition('FocusDrive')

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
        positions.append((*well_id_to_position(well_id), zpos))
    positions = np.array(positions)

    top_left_x, top_left_y = well_id_to_position(top_left_well_id)
    bot_right_x, bot_right_y = well_id_to_position(bottom_right_well_id)

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
    positions = []
    for well_id, zpos in measured_focusdrive_positions.items():
        positions.append((*well_id_to_position(well_id), zpos))
    positions = np.array(positions)

    with open(position_list_filepath) as file:
        position_list = json.load(file)

    for ind, pos in enumerate(position_list['POSITIONS']):
        well_id, site_num = parse_hcs_site_label(pos['LABEL'])
        x, y = well_id_to_position(well_id)

        # the interpolated z-position of the current well
        interpolated_position = interpolate.griddata(
            positions[:, :2], 
            positions[:, 2], 
            (x, y), 
            method=method
        )

        # add the optional user-defined constant offset
        interpolated_position += offset

        # the config entry for the 'FocusDrive' device (this is the motorized z-stage)
        focusdrive_config = {
            'X': float(interpolated_position),
            'Y': 0,
            'Z': 0,
            'AXES': 1,
            'DEVICE': 'FocusDrive',
        }
        position_list['POSITIONS'][ind]['DEVICES'].append(focusdrive_config)
    
    # save the new position_list
    ext = position_list_filepath.split('.')[-1]
    new_filepath = re.sub('.%s$' % ext, '_interpolated.%s' % ext, position_list_filepath)
    with open(new_filepath, 'w') as file:
        json.dump(position_list, file)
    return new_filepath, position_list


def visualize_interpolation(measured_focusdrive_positions, new_position_list):

    def xyz_from_pos(pos):
        well_id, site_num = parse_hcs_site_label(pos['LABEL'])
        focusdrive = [d for d in pos['DEVICES'] if d['DEVICE'] == 'FocusDrive'][0]
        x, y = well_id_to_position(well_id)
        z = focusdrive['X']
        return x, y, z

    measured_positions = np.array(
        [
            (*well_id_to_position(well_id), zpos) 
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
