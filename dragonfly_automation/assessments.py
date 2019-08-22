
import numpy as np

from dragonfly_automation import utils


def assess_confluency(im):
    '''
    Assess confluency of a FOV given a single image (z-slice)
    that is potentially somewhat out-of-focus

    Returns
    -------
    confluency_is_good : bool
        whether the confluency looks 'good'
    confluency_label : str
        details about the confluency
        'good', 'low', 'high', 'anisotropic'
    '''

    # parameters used for decision tree
    # min/max number of nuclei in the FOV
    min_count = 15
    max_count = 50

    # max offset of the center of mass and and max asymmetry (eigenvalue ratio)
    max_offset = .3
    max_asymmetry = 1.0

    # hard-coded approximate nucleus radius
    nucleus_radius = 15

    # hard-coded image dimensions
    image_size = 1024

    # default values
    confluency_label = None
    confluency_is_good = True

    # find the positions of each nucleus
    nucleus_positions = utils.find_nuclei(im, nucleus_radius)

    # calculate some properties of the spatial distribution of nuclei 
    count, offset, asymmetry = utils.analyze_nucleus_positions(nucleus_positions, image_size)

    # very rudimentary logic to assess confluency using these properties
    # too many nuclei
    if count > max_count:
        confluency_label = 'high'
        confluency_is_good = False

    # too few nuclei
    elif count < min_count:
        confluency_label = 'low'
        confluency_is_good = False

    # distribution of nuclei is not isotropic
    elif offset > max_offset or asymmetry > max_asymmetry:
        confluency_label = 'anisotropic'
        confluency_is_good = False
    
    return confluency_is_good, confluency_label



