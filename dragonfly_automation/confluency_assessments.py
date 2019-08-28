
import numpy as np

from scipy import ndimage

import skimage
from skimage import feature
from skimage import morphology

from dragonfly_automation import utils


def assess_confluency(im):
    '''
    Assess confluency of a single FOV given a single z-slice
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
    min_num_nuclei = 15
    max_num_nuclei = 50

    # max offset of the center of mass and and max asymmetry (eigenvalue ratio)
    max_rel_com_offset = .3
    max_eval_ratio = 1.0

    # hard-coded approximate nucleus radius
    nucleus_radius = 15

    # hard-coded image dimensions
    image_size = 1024

    # default values
    confluency_label = None
    confluency_is_good = True

    # find the positions of the nuclei in the image
    nucleus_positions = _identify_nuclei(im, nucleus_radius)

    # calculate some properties of the spatial distribution of nucleus positions
    num_nuclei, rel_com_offset, eval_ratio = _calculate_features_of_nucleus_positions(
        nucleus_positions, image_size)

    # very rudimentary logic to assess confluency using these properties
    # too many nuclei
    if num_nuclei > max_num_nuclei:
        confluency_label = 'high'
        confluency_is_good = False

    # too few nuclei
    elif num_nuclei < min_num_nuclei:
        confluency_label = 'low'
        confluency_is_good = False

    # distribution of nuclei is not isotropic
    elif rel_com_offset > max_rel_com_offset or eval_ratio > max_eval_ratio:
        confluency_label = 'anisotropic'
        confluency_is_good = False
    
    return confluency_is_good, confluency_label



def _identify_nuclei(im, nucleus_radius):
    '''

    '''
    
    # smooth the raw image
    imf = skimage.filters.gaussian(im, sigma=5)
    
    # background mask
    mask = imf > skimage.filters.threshold_li(imf)
    
    # smoothed distance transform
    dist = ndimage.distance_transform_edt(mask)
    distf = skimage.filters.gaussian(dist, sigma=1)
        
    # the indicies of the local maximima in the distance transform
    # correspond roughly to the positions of the nuclei
    local_max_inds = skimage.feature.peak_local_max(
        distf, indices=True, min_distance=nucleus_radius, labels=mask)

    return local_max_inds


def _calculate_features_of_nucleus_positions(positions, image_size):
    '''

    '''

    # the number of nuclei
    num_nuclei = positions.shape[0]
    
    # the distance of the center of mass from the center of the image
    # (relative to the size of the image)
    rel_com_offset = ((positions.mean(axis=0) - (image_size/2))**2).sum()**.5 / (image_size/2)

    # eigenvalues of the covariance matrix
    evals, evecs = np.linalg.eig(np.cov(positions.transpose()))

    # the ratio of eigenvalues is a measure of asymmetry 
    eval_ratio = (max(evals) - min(evals))/min(evals)
    
    return num_nuclei, rel_com_offset, eval_ratio

