import os
import sys
import skimage
import numpy as np

from scipy import ndimage
from skimage import feature
from skimage import morphology


def find_nuclei(im, nucleus_radius):
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
    local_max_inds = skimage.feature.peak_local_max(distf, indices=True, min_distance=nucleus_radius, labels=mask)
    
    return local_max_inds


def analyze_nucleus_positions(positions, image_size):
    '''

    '''

    # the number of nuclei
    count = positions.shape[0]
    
    # the distance of the center of mass from the center of the image
    # (relative to the size of the image)
    position_mean = positions.mean(axis=0)
    offset = ((position_mean - (image_size/2))**2).sum()**.5 / (image_size/2)
    
    # eigenvalues of the covariance matrix
    vals, vec = np.linalg.eig(np.cov(positions.transpose()))

    # the ratio of eigenvalues is a measure of asymmetry 
    eigenvalue_ratio = (max(vals) - min(vals))/min(vals)
    
    return count, offset, eigenvalue_ratio

