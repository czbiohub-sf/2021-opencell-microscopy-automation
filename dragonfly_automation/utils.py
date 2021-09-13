import datetime
import os
import re

import numpy as np
import pandas as pd


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
    im = im / (maxx - minn)
    im[im > 1] = 1
    im = (im * max_value).astype(dtype)
    return im


def multiply_and_clip_to_uint16(im, scale):

    dtype = 'uint16'
    max_value = 65535
    im_dst = im.copy().astype(float)
    im_dst *= scale
    im_dst[im_dst > max_value] = max_value
    return im_dst.astype(dtype)


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
