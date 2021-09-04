import os
import re
import sys
import py4j
import json
import skimage
import datetime
import numpy as np

from dragonfly_automation import operations, utils


top_left_well_id = 'A1'
bottom_right_well_id = 'H12'

# these are real measured positions from a full-plate acquisition (PML0312 on 2020-06-16)
measured_focusdrive_positions = {
    'A1': 6256.7289200000005,
    'A4': 6315.793935000001,
    'A8': 6357.911480000001,
    'A12': 6345.625135,
    'D12': 6400.021415,
    'E8': 6420.134645,
    'D4': 6376.704375,
    'E1': 6324.45031,
    'H1': 6326.341395,
    'H4': 6380.273465,
    'H8': 6419.40789,
    'H12': 6424.030965,
}

# from PML0334
measured_focusdrive_positions = {
    'A1': 6120.02288,
    'A4': 6195.396125,
    'A8': 6240.0021400000005,
    'A12': 6224.862045000001,
    'D12': 6279.707315000001,
    'E8': 6309.964675,
    'D4': 6256.04402,
    'E1': 6191.020375,
    'H1': 6185.746645,
    'H4': 6255.530345,
    'H8': 6303.899505,
    'H12': 6294.01792
}

# from PML0330
measured_focusdrive_positions = {
    'A1': 6120.384355,
    'A4': 6181.652465,
    'A8': 6218.770240000001,
    'A12': 6208.778310000001,
    'D12': 6272.386495000001,
    'E8': 6286.11113,
    'D4': 6242.441145000001,
    'E1': 6197.86557,
    'H1': 6203.16974,
    'H4': 6248.60144,
    'H8': 6291.228855,
    'H12': 6302.704735
}

# position_list_filepath = '../tests/data/20200609_raw_positions_interpolated.pos'
# with open(position_list_filepath, 'r') as file:
#     position_list = json.load(file)

# new_position_list_filepath, new_position_list = utils.interpolate_focusdrive_positions(
#     position_list_filepath,
#     measured_focusdrive_positions,
#     top_left_well_id,
#     bottom_right_well_id,
#     method='cubic'
# )
