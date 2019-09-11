import os
import json
import skimage
import sklearn
import tifffile
import datetime
import numpy as np
import pandas as pd
from scipy import ndimage
from sklearn import cluster
from sklearn import metrics
from skimage import feature
from skimage import morphology
from matplotlib import pyplot as plt

from dragonfly_automation import utils


def classify_fov(image, classifier, log_dir=None, position_ind=None):
    '''
    Assess confluency and spread of a single FOV given a single z-slice
    that may be somewhat out-of-focus

    Parameters
    ----------
    image : numpy array
        an image of the current z-slice (assumed to be 2D and uint16)
    log_dir : str, optional
        Local path to the directory in which to save log files
        (if None, no logging is performed)
    position_ind : int, optional (but required for logging)
        The index of the current position
        Note that we use an index, and not a label, because it's not clear
        that we can assume that the labels in the position list will always be unique 
        (e.g., they may not be when the position list is generated manually,
        rather than by the HCS Site Generator plugin)
        
    Returns
    -------
    fov_is_good : bool
        whether the FOV was classified as a 'good' FOV
    assessment_did_succeed : bool
        whether any errors occurred 
        (if they did, fov_is_good will be False)
    '''

    # empirical hard-coded approximate nucleus radius
    nucleus_radius = 15

    # hard-coded image dimensions
    image_size = 1024

    # empirical hard-coded absolute minimum intensity
    # (used to determine if there are no nuclei present in the FOV)
    min_absolute_intensity = 1000

    # conservative empirical thresholds on the number of nuclei
    # in a candidate 'good' FOV
    min_num_nuclei = 20
    max_num_nuclei = 80

    # crude check for whether there are any nuclei in the FOV at all
    thresh = skimage.filters.threshold_li(image)
    if thresh < min_absolute_intensity:
        # TODO: how to handle and log this
        pass

    # find the positions of the nuclei in the image
    mask = _generate_background_mask(image)
    nucleus_positions = _find_nucleus_positions(mask, nucleus_radius)

    # check whether there are way too few or way too many 'nuclei' in the FOV
    # this will occur if either
    # 1) there are very few or very many real nuclei in the FOV, or
    # 2) there are _no_ real nuclei in the FOV,
    #    and the nucleus positions correspond to noise, dust, etc
    num_positions = nucleus_positions.shape[0]
    if num_positions < min_num_nuclei or num_positions > max_num_nuclei:
        # TODO: how to handle and log this
        pass

    # default values
    fov_is_good = False
    error_has_occurred = False
    feature_calculation_error = None
    classifier_error = None

    # attempt to calculate the features
    features = None
    try:
        features = _calculate_all_features(nucleus_positions, image_size)
    except Exception as exception:
        error_has_occurred = True
        feature_calculation_error = str(exception)

    # attempt to call the classifier
    if not error_has_occurred:
        ordered_features = _order_features(features)
        X = np.array(ordered_features)[None, :]
        try:
            fov_is_good = classifier.predict(X)[0]
        except Exception as exception:
            error_has_occurred = True
            classifier_error = str(exception)

    # log the image, features, and errors
    if log_dir is not None:
        properties = {
            'fov_is_good': fov_is_good, 
            'classifier_error': classifier_error,
            'feature_calculation_error': feature_calculation_error
        }
        if features is not None:
            properties.update(features)

        log_dir = os.path.join(log_dir, 'fov-assessment')
        _log_fov_assessment(image, properties, nucleus_positions, log_dir, position_ind)
    
    assessment_did_succeed = not error_has_occurred
    return fov_is_good, assessment_did_succeed



def _log_fov_assessment(image, properties, nucleus_positions, log_dir, position_ind):
    '''
    '''

    # make the directory for the images
    image_dir = os.path.join(log_dir, 'fov-images')
    os.makedirs(image_dir, exist_ok=True)

    # filename for the image itself
    def image_filename(tag):
        return 'FOV_%05d_%s.tif' % (position_ind, tag)

    # create the row to append to the logfile
    row = {
        'timestamp': utils.timestamp(),
        'position_ind': position_ind,
        'image_filename': image_filename('RAW'), 
    }
    row.update(properties)

    
    # create the log file if it does not exist
    log_filepath = os.path.join(log_dir, 'fov-assessment.csv')

    # append the row to the log file
    if os.path.isfile(log_filepath):
        d = pd.read_csv(log_filepath)
        d = d.append(row, ignore_index=True)
    else:
        d = pd.DataFrame([row])
    d.to_csv(log_filepath, index=False, float_format='%0.2f')

    # save the raw image
    tifffile.imsave(os.path.join(image_dir, image_filename('RAW')), image.astype('uint16'))
    
    # save an autogained version of the image (for Windows preview)
    image = _to_uint8(image)
    tifffile.imsave(os.path.join(image_dir, image_filename('UINT8')), image)

    # crudely annotate the image by marking the nucleus positions with white squares
    width = 3
    white = 255
    shape = image.shape

    # lower the brightness of the autogained image so that the squares are easier to see
    image = (image/2).astype('uint8')

    # draw a square on the image at each nucleus position
    for pos in nucleus_positions:
        image[
            int(max(0, pos[0] - width)):int(min(shape[0], pos[0] + width)), 
            int(max(0, pos[1] - width)):int(min(shape[1], pos[1] + width))] = white

    tifffile.imsave(os.path.join(image_dir, image_filename('ANT')), image)


def _to_uint8(im):

    dtype = 'uint8'
    max_value = 255

    im = im.copy().astype(float)

    percentile = 1
    minn, maxx = np.percentile(im, (percentile, 100 - percentile))
    if minn==maxx:
        return (im * 0).astype(dtype)

    im = im - minn
    im[im < minn] = 0
    im = im/(maxx - minn)
    im[im > 1] = 1
    im = (im * max_value).astype(dtype)
    return im


def _generate_background_mask(im):
    '''
    '''
    # smooth the raw image
    imf = skimage.filters.gaussian(im, sigma=5)

    # background mask from minimum cross-entropy
    mask = imf > skimage.filters.threshold_li(imf)

    # erode once to eliminate isolated pixels in the mask
    # (a defense against thresholding noise)
    mask = skimage.morphology.erosion(mask)

    # remove regions that are too small to be nuclei
    min_region_area = 1000
    mask_label = skimage.measure.label(mask)
    props = skimage.measure.regionprops(mask_label)
    for prop in props:
        if prop.area < min_region_area:
            mask[mask_label==prop.label] = False

    return mask


def _find_nucleus_positions(mask, nucleus_radius):
    '''

    '''
    # smoothed distance transform
    dist = ndimage.distance_transform_edt(mask)
    distf = skimage.filters.gaussian(dist, sigma=1)

    # the positions of the local maximima in the distance transform
    # correspond roughly to the centers of mass of the individual nuclei
    local_max_inds = skimage.feature.peak_local_max(
        distf, indices=True, min_distance=nucleus_radius, labels=mask)

    return local_max_inds


def _show_nucleus_positions(positions, im=None, ax=None):
    '''
    Plot the nucleus positions, 
    optionally overlaid on the image itself and the background mask
    '''
    
    if ax is None:
        plt.figure(figsize=(10, 10))
        ax = plt.gca()

    # show the image and the background mask
    if im is not None:
        mask = _generate_background_mask(im)
        ax.imshow(
            skimage.color.label2rgb(~mask, image=_to_uint8(im), colors=('black', 'yellow')))

    # plot the positions themselves
    ax.scatter(positions[:, 1], positions[:, 0], color='red')
    ax.set_xlim([0, 1024])
    ax.set_ylim([0, 1024])
    ax.set_aspect('equal')


def _calculate_nucleus_position_features(positions, image_size):
    '''

    '''

    # the number of nuclei
    num_nuclei = positions.shape[0]
    
    # the distance of the center of mass from the center of the image
    com_offset = ((positions.mean(axis=0) - (image_size/2))**2).sum()**.5
    rel_com_offset = com_offset / image_size

    # eigenvalues of the covariance matrix
    evals, evecs = np.linalg.eig(np.cov(positions.transpose()))

    # the ratio of eigenvalues is a measure of asymmetry 
    eval_ratio = (max(evals) - min(evals))/min(evals)

    return {
        'num_nuclei': num_nuclei, 
        'rel_com_offset': rel_com_offset, 
        'eval_ratio': eval_ratio,
    }


def _calculate_mask_features(positions, image_size, nucleus_radius):
    '''
    Calculate properties of a simulated nucleus mask
    (obtained by thresholding the distance transform of the nucleus positions)

    Note that it is necessary to simulate a nucleus mask,
    rather than use the mask returned by _generate_background_mask, 
    because the area of the foreground regions in the generated mask
    depends on the focal plane.

    '''

    position_mask = np.zeros((image_size, image_size))
    for position in positions:
        position_mask[position[0], position[1]] = 1

    mask = ndimage.distance_transform_edt(~position_mask.astype(bool)) > nucleus_radius
    props = skimage.measure.regionprops(skimage.measure.label(mask))

    num_regions = len(props)
    total_area = mask.sum() / (mask.shape[0]*mask.shape[1])
    median_region_area = np.median([p.area for p in props])
    max_distance = ndimage.distance_transform_edt(~mask).max()

    return {
        'num_regions': num_regions, 
        'total_area': total_area, 
        'median_region_area': median_region_area, 
        'max_distance': max_distance
    }


def _calculate_nucleus_cluster_features(positions, image_size):
    '''
    Cluster positions using DBSCAN 
    and calculate various measures of cluster homogeneity
    '''
    
    # empirically-selected parameters for dbscan
    eps = 0.1
    min_samples = 3

    dbscan = sklearn.cluster.DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
    dbscan.fit(positions/image_size)

    labels = dbscan.labels_
    cluster_labels = set(labels)
    num_clusters = len(cluster_labels)
    num_unclustered = (labels==-1).sum()

    if num_clusters > 1:
        sil_score = sklearn.metrics.silhouette_score(positions, labels)
        db_score = sklearn.metrics.davies_bouldin_score(positions, labels)
        ch_score = sklearn.metrics.calinski_harabasz_score(positions, labels)
    else:
        sil_score, db_score, ch_score = None, None, None

    return {
        'num_clusters': num_clusters, 
        'num_unclustered': num_unclustered, 
        'sil_score': sil_score, 
        'db_score': db_score, 
        'ch_score': ch_score
    }


def _calculate_all_features(positions, image_size):
    '''
    '''

    position_features = _calculate_nucleus_position_features(positions, image_size)

    # note that cluster features can be None if none or one clusters was found
    cluster_features = _calculate_nucleus_cluster_features(positions, image_size)

    # the nucleus radius here was selected empirically
    mask_features = _calculate_mask_features(positions, image_size, nucleus_radius=50)

    # concat features
    features = {}
    features.update(position_features)
    features.update(mask_features)
    features.update(cluster_features)
    return features


def _order_features(features):

    order = (
        'num_nuclei',
        'rel_com_offset',
        'eval_ratio',
        'num_regions',
        'total_area',
        'median_region_area',
        'max_distance',
        'num_clusters',
        'num_unclustered',
        'sil_score',
        'db_score',
        'ch_score'
    )

    return tuple([features.get(key) for key in order])