import os
import re
import sys
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
from sklearn import ensemble
from skimage import feature
from skimage import morphology
from matplotlib import pyplot as plt

from dragonfly_automation import utils


def printr(s):
    sys.stdout.write('\r%s' % s)


def catch_errors(method):
    '''
    Wrapper for instance methods called in self.score_raw_fov
    that catches and logs *all* exceptions and, if an exception occurs,
    sets the score to None and prevents subsequent wrapped methods from executing
    '''
    method_name = method.__name__

    def wrapper(self, *args, **kwargs):
        if self.allow_errors:
            return method(self, *args, **kwargs)

        # do not call the method if a score has already been assigned
        # (this happens when an error ocurred in an earlier wrapped method)
        result = None
        if self.score_has_been_assigned:
            return result

        try:
            result = method(self, *args, **kwargs)
        except Exception as error:
            # if an error has occured, we set the score to None
            # note that self.assign_score sets the score_has_been_assigned flag to True,
            # which will prevent the execution of all subsequent catch_errors-wrapped methods
            self.assign_score(
                score=None, comment=('Error in %s: %s' % (method_name, str(error)))
            )

        return result
    return wrapper


class PipelineFOVScorer:

    def __init__(
        self, 
        save_dir, 
        mode='prediction', 
        model_type='regression', 
        random_state=None, 
        log_dir=None
    ):
        '''
        mode : 'training' or 'prediction'
        model : 'classification' or 'regression'
        log_dir : str, optional
            path to a local directory in which to save log files
        '''
        self.save_dir = save_dir
        self.log_dir = log_dir

        # whether the methods wrapped by catch_errors
        # can raise errors or not (explicitly set to False in self.score_raw_fov)
        self.allow_errors = True

        if mode not in ['training', 'prediction']:
            raise ValueError("`mode` must be either 'training' or 'prediction'")
        self.mode = mode

        if model_type not in ['classification', 'regression']:
            raise ValueError("`model` must be either 'classification' or 'regression'")
        self.model_type = model_type

        self.training_data = None
        self.cached_training_metadata = None
        self.current_training_metadata = None

        # hard-coded image size
        self.image_size = 1024

        # hard-coded feature order for the model
        self.feature_order = (
            'num_nuclei',
            'com_offset',
            'eval_ratio',
            'total_area',
            'max_distance',
            'num_clusters',
            'num_unclustered',
        )

        if self.model_type == 'classification':
            self.model = sklearn.ensemble.RandomForestClassifier(
                n_estimators=300,
                max_features='sqrt',
                oob_score=True,
                random_state=random_state
            )

        if self.model_type == 'regression':
            self.model = sklearn.ensemble.RandomForestRegressor(
                n_estimators=300,
                max_features='auto',
                oob_score=True,
                random_state=random_state
            )


    def load(self):
        '''
        Load existing training data and metadata
        '''
        # reset the current validation results
        self.current_training_metadata = None
    
        # load the training data        
        self.training_data = pd.read_csv(os.path.join(self.save_dir, 'training_data.csv'))

        # load the cached validation results
        training_metadata_filepath = os.path.join(self.save_dir, 'training_metadata.json')
        if os.path.isfile(training_metadata_filepath):
            with open(training_metadata_filepath, 'r') as file:
                self.cached_training_metadata = json.load(file)
        else:
            print('Warning: no cached model metadata found')


    def save(self, save_dir, overwrite=False):
        '''
        Save the training data and the metadata
        '''
        os.makedirs(save_dir, exist_ok=True)

        if self.mode != 'training':
            raise ValueError("Cannot save training data unless mode = 'training'")
        
        # don't overwrite existing data
        training_data_filepath = os.path.join(save_dir, 'training_data.csv')
        if os.path.isfile(training_data_filepath) and not overwrite:
            raise ValueError('Training data already saved to %s' % training_data_filepath)

        # save the training data
        self.training_data.to_csv(training_data_filepath, index=False)
        print('Training data saved to %s' % training_data_filepath)

        # save the metadata
        training_metadata_filepath = os.path.join(save_dir, 'training_metadata.json')
        if self.current_training_metadata is None:
            print('Warning: no metadata found to save')
            return

        self.current_training_metadata['filepath'] = os.path.abspath(training_data_filepath)
        with open(training_metadata_filepath, 'w') as file:
            json.dump(self.current_training_metadata, file)
        print('Metadata saved to %s' % training_metadata_filepath)


    def process_existing_fov(self, filepath):
        '''
        Process and calculate features from a single extant FOV
        (in the form of either a z-projection or a single in-focus z-slice)

        This method is intended to process an existing image,
        either to generate training data or to predict a score for an existing FOV;
        it is *not* intended to process an image directly from the microscope itself
        (for this application, see self.score_raw_fov)

        Note that this method behaves like a class method, but cannot formally be one
        because the feature extraction methods themselves are instance methods
        (which they must be for the catch_errors wrapper to work)

        Parameters
        ----------
        filepath : absolute path to a single FOV (as a single-page TIFF)
        '''

        # hard-coded values for FOV pre-processing thresholds
        # a value of 700 for min_otsu_thresh and 10 for min_num_nuclei were always used,
        # to both generate training data in Oct 2019, 
        # and for scoring all OpenCell FOVs from 2019-2021
        min_otsu_thresh = 700
        min_num_nuclei = 10

        result = {
            'filename': filepath.split(os.sep)[-1],
            'score': None,
        }
    
        if not os.path.isfile(filepath):
            result['error'] = 'File does not exist'
            return result

        try:
            im = tifffile.imread(filepath)
    
            # check whether there are any nuclei in the FOV
            nuclei_in_fov = self.are_nuclei_in_fov(im, min_otsu_thresh)
            if not nuclei_in_fov:
                result['error'] = 'No nuclei in the FOV'
                return result

            # calculate the background mask and nucleus positions
            mask = self.generate_background_mask(im)
            positions = self.find_nucleus_positions(mask)

            # determine if the are too few nuclei in the mask to proceed
            enough_nuclei_in_fov = self.are_enough_nuclei_in_fov(positions, min_num_nuclei)
            if not enough_nuclei_in_fov:
                result['error'] = 'Too few nuclei in the FOV'
                return result

            # calculate features and predict the score
            features = self.calculate_features(positions)
            result.update(features)

            score = self.predict_score(features)
            result['score'] = score

        except Exception as error:
            result['error'] = str(error)
    
        return result


    def train(self):
        '''
        Train a model to predict the score 

        Note that the model can be either a classification or regression model,
        since the score is a categorical variable (e.g., -1, 0, 1 for bad/neutral/good)
        '''

        label = 'score'

        # turn on errors in feature extraction methods
        self.allow_errors = True

        training_metadata = {
            'model_type': self.model_type,
            'training_label': label,
            'training_timestamp': utils.timestamp(),
            'model': {
                'class': str(self.model.__class__),
                'params': self.model.get_params(),
            }
        }

        # drop rows with any missing/nan features
        # (note that, for now, we do not worry about what gets dropped here)
        data = self.training_data.copy()
        mask = data[list(self.feature_order)].isna().sum(axis=1)
        if mask.sum():
            print(
                'Warning: %d rows of training data have missing features and will be dropped'
                % mask.sum()
            )
            data = data.loc[mask == 0]

        # mask to identify training data with and without annotations
        mask = data[label].isna()
        data = data.loc[~mask]

        X = data[list(self.feature_order)].values
        y = data[label].values.astype(float)

        # train the model
        self.model.fit(X, y)

        # log the oob_score
        training_metadata['oob_score'] = '%0.2f' % self.model.oob_score_
        training_metadata['training_data_shape'] = list(X.shape)
        print('oob_score: %0.2f' % self.model.oob_score_)

        self.current_training_metadata = training_metadata


    def validate(self):
        '''
        Sanity checks when in 'prediction' mode and after calling self.train
        Intended use is right after loading and training from cached data on the microscope 

        Steps:
            1) Print the cached and current cross-validation results
               for manual inspection/validation
            2) Check that the number of features returned by self.calculate_features
               matches the number expected by the trained model.
               (note that self.train implicitly validates whether all the features
               listed in self.feature_order appear in the cached training data,
               but there is no guarantee that self.feature_order is consistent
               with the features actually returned by self.calculate_features)
        '''
        if self.cached_training_metadata is None or self.current_training_metadata is None:
            print('Warning: cannot validate without cached and current metadata to compare')
            return

        print(
            'Cached and current training data shape: (%s, %s)'
            % (
                self.cached_training_metadata['training_data_shape'],
                self.current_training_metadata['training_data_shape']
            )
        )
        print(
            'Cached and current oob_score: (%s, %s)'
            % (
                self.cached_training_metadata['oob_score'],
                self.current_training_metadata['oob_score']
            )
        )

        # make sure the number of features is consistent 
        self.allow_errors = True
        mock_positions = np.array([[100, 500], [500, 500], [500, 100]])
        features = self.calculate_features(mock_positions)
        score = self.predict_score(features)
        print('Mock predicted score (should be falsy): %s' % score)


    def score_raw_fov(self, image, min_otsu_thresh, min_num_nuclei, position_props=None):
        '''
        Predict a score for a raw, uncurated FOV from the microscope itself

        Steps:
            validate the image object (yes, no, error)
            check that there are nuclei in the image (yes, no, error)
            generate the mask and find positions (continue, error)
            check the number of nuclei (yes, no, error)
            calculate features (continue, error)
            make the prediction (yes, no, error)

        Poorly documented failure modes:
          - if there is some giant but uniform artifact that doesn't break the thresholding
            (e.g., some large region is uniformly zero), we'll never know

        Parameters
        ----------
        image : numpy.ndarray (2D and uint16)
            The raw field of view to score; assumed to be close to in-focus
        min_otsu_thresh : int, intensity threshold used to determine if any nuclei are in the FOV
        min_num_nuclei : int, threshold number of nuclei used to determine 
            whether there are enough nuclei in the FOV to justify scoring it
        position_props : dict, optional (but required for logging)
            The properties (name, label, ind, etc) of the current position;
            these are created in PipelinePlateAcquisition.run 
            by parsing the position labels generated by the HCS Site Generator
        '''

        # required for the catch_errors wrapper to actually catch (and log) errors
        self.allow_errors = False

        # reset the 'state' of the score-assignment logic
        self.score_has_been_assigned = False

        # raw FOV properties (modified in this method and in self.assign_score)
        self.raw_fov_props = {
            'raw_image': None,
            'score': None,
            'comment': None,
            'features': {},
        }

        # validate the image: check that it's a 2D uint16 ndarray
        # note that, because validate_raw_fov is wrapped by the catch_errors method,
        # we must check that the validation_result is not None before accessing the 'flag' key.
        # also note that we only include the image in self.raw_fov_props if it passes validation
        # (otherwise, errors may result when the image itself is logged in self.log_raw_fov_props)
        is_valid_fov, comment = self.validate_raw_fov(image)
        if is_valid_fov:
            self.raw_fov_props['raw_image'] = image
        else:
            self.assign_score(score=None, comment=comment)
    
        # check whether there are any nuclei in the FOV
        nuclei_in_fov = self.are_nuclei_in_fov(image, min_otsu_thresh)
        if not nuclei_in_fov:
            self.assign_score(score=None, comment='No nuclei in the FOV')

        # determine if the are too few nuclei in the mask to proceed
        mask = self.generate_background_mask(image)
        positions = self.find_nucleus_positions(mask)
        enough_nuclei_in_fov = self.are_enough_nuclei_in_fov(positions, min_num_nuclei)
        if not enough_nuclei_in_fov:
            self.assign_score(score=None, comment='Too few nuclei in the FOV')

        # calculate features from the positions
        features = self.calculate_features(positions)
        self.raw_fov_props['features'] = features

        # finally, use the trained model to generate a prediction
        score = self.predict_score(features)
        self.raw_fov_props['score'] = score
        if score is not None:
            self.assign_score(score, comment='Model prediction')
    
        if self.log_dir is not None and position_props is not None:
            self.log_raw_fov_props(position_props)
        return self.raw_fov_props


    def assign_score(self, score, comment):
        '''
        '''
        # do nothing if a score has already been assigned
        if not self.score_has_been_assigned:
            self.score_has_been_assigned = True
            self.raw_fov_props['score'] = score
            self.raw_fov_props['comment'] = comment


    def log_raw_fov_props(self, position_props):
        '''
        '''
        self.logged_image_dir = os.path.join(self.log_dir, 'fov-images')
        os.makedirs(self.logged_image_dir, exist_ok=True)

        # log the raw image itself
        snap_filepath = os.path.join(
            self.logged_image_dir, 'FOV_%s_RAW.tif' % (position_props['name'],)
        )
        raw_image = self.raw_fov_props.get('raw_image')
        if raw_image is not None:
            tifffile.imsave(snap_filepath, raw_image)

        # construct the CSV log row
        row = {
            'score': self.raw_fov_props.get('score'),
            'comment': self.raw_fov_props.get('comment'),
            'timestamp': utils.timestamp(),
            'image_filepath': snap_filepath,
        }

        # the position attributes (ind, label, name, well_id, site_num)
        for key, val in position_props.items():
            row['position_%s' % key] = val

        # the FOV features
        features = self.raw_fov_props.get('features')
        if features is not None:
            row.update(features)

        # append the row to the log file
        log_filepath = os.path.join(self.log_dir, 'fov-score-log.csv')
        if os.path.isfile(log_filepath):
            log = pd.read_csv(log_filepath)
            log = log.append(row, ignore_index=True)
        else:
            log = pd.DataFrame([row])
        log.to_csv(log_filepath, index=False, float_format='%0.2f')


    @catch_errors
    def validate_raw_fov(self, image):
        
        is_valid_fov = False
        comment = None
        if not isinstance(image, np.ndarray):
            comment = 'Image is not an np.ndarray'
        elif image.dtype != 'uint16':
            comment = 'Image is not uint16'
        elif image.ndim != 2:
            comment = 'Image is not 2D'
        elif image.shape[0] != self.image_size or image.shape[1] != self.image_size:
            comment = 'Image shape is not (%s, %s)' % (self.image_size, self.image_size)
        else:
            is_valid_fov = True
        return is_valid_fov, comment


    @catch_errors
    def are_nuclei_in_fov(self, image, min_otsu_thresh):
        '''
        Check whether there are *any* real nuclei in the image,
        using an empirically-determine minimum Otsu threshold
        (defined in process_existing_fov and in FOVSelectionSettings)
        '''
        otsu_thresh = skimage.filters.threshold_li(image)
        nuclei_in_fov = otsu_thresh > min_otsu_thresh
        return nuclei_in_fov


    @catch_errors
    def are_enough_nuclei_in_fov(self, positions, min_num_nuclei):
        '''
        Check whether there are way too few 'nuclei' in the FOV

        This will occur if either
           1) there are very few real nuclei in the FOV, or
           2) there are _no_ real nuclei in the FOV,
              and the nucleus positions correspond to noise, dust, etc
              (we attempt to defend against this by first calling are_nuclei_in_fov)
        '''
        num_positions = positions.shape[0]
        is_candidate = num_positions > min_num_nuclei
        return is_candidate


    @catch_errors
    def generate_background_mask(self, image):

        # smooth the raw image
        imf = skimage.filters.gaussian(image, sigma=5)

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
                mask[mask_label == prop.label] = False

        return mask


    @catch_errors
    def find_nucleus_positions(self, mask):
        '''
        '''
        # empirically estimated nucleus radius
        nucleus_radius = 15

        # smoothed distance transform
        dist = ndimage.distance_transform_edt(mask)
        distf = skimage.filters.gaussian(dist, sigma=1)

        # the positions of the local maximima in the distance transform
        # correspond roughly to the centers of mass of the individual nuclei
        positions = skimage.feature.peak_local_max(distf, min_distance=nucleus_radius, labels=mask)
        return positions    


    @catch_errors
    def calculate_features(self, positions):
        '''
        Calculate a variety of features from the list of nucleus positions
        These fall into three categories:
        1) basic summary statistics: number of nuclei, their relative center of mass, 
            and orientation (from the eigenvalues of the covariance matrix)
        2) total nucleus area and max distance from a nucleus, from a simulated nucleus mask
            obtained by thresholding the distance transform of the nucleus positions.
            (Note that it is necessary to simulate a nucleus mask because the mask
            returned by generate_background_mask depends on the focal plane)
        3) results from using DBSCAN to cluster the nucleus positions
        '''
        # the number of nuclei
        num_nuclei = positions.shape[0]

        # the distance of the center of mass from the center of the image
        com_offset = ((positions.mean(axis=0) - (self.image_size/2))**2).sum()**.5
        rel_com_offset = com_offset / self.image_size

        # the ratio of eigenvalues (a crude measure of asymmetry)
        evals, evecs = np.linalg.eig(np.cov(positions.transpose()))
        eval_ratio = (max(evals) - min(evals))/min(evals)

        # calculate total area and max distance from a nucleus using a simulated nucleus mask
        # (the nucleus radius here was selected empirically)
        nucleus_radius = 50
        position_mask = np.zeros((self.image_size, self.image_size))
        for position in positions:
            position_mask[position[0], position[1]] = 1

        dist = ndimage.distance_transform_edt(~position_mask.astype(bool))
        total_area = (dist < nucleus_radius).sum() / (self.image_size*self.image_size)
        max_distance = dist.max()

        # cluster positions using DBSCAN
        # eps is an empirically-selected neighborhood size in pixels,
        # and min_samples = 3 is the minimum required for non-trivial clustering
        # (nb the clustering is very sensitive to `eps`)
        eps = 100
        min_samples = 3
        dbscan = sklearn.cluster.DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
        dbscan.fit(positions)
        labels = dbscan.labels_
        num_clusters = len(set(labels))
        num_unclustered = (labels == -1).sum()

        features = dict(
            num_nuclei=num_nuclei, 
            com_offset=rel_com_offset, 
            eval_ratio=eval_ratio, 
            total_area=total_area,
            max_distance=max_distance, 
            num_clusters=num_clusters, 
            num_unclustered=num_unclustered
        )
        return features


    @catch_errors
    def predict_score(self, features):
        '''
        Use the pre-trained model to predict a score
        '''
        # construct the feature array of shape (1, num_features)
        X = np.array([features.get(name) for name in self.feature_order])[None, :]

        # use predict_proba if the model is a classifier
        # (note that predict_proba returns [p_false, p_true])
        if self.model_type == 'classification':
            score = self.model.predict_proba(X)[0][1]
        else:
            score = self.model.predict(X)[0]

        return score


    def show_nucleus_positions(self, positions, im=None, ax=None):
        '''
        Convenience method to visualize the nucleus positions, 
        optionally overlaid on the image itself and the background mask
        '''
        if ax is None:
            plt.figure(figsize=(10, 10))
            ax = plt.gca()

        # show the image and the background mask
        if im is not None:
            mask = self.generate_background_mask(im)
            ax.imshow(
                skimage.color.label2rgb(mask, image=utils.to_uint8(im), colors=('black', 'yellow'))
            )

        # plot the positions themselves
        ax.scatter(positions[:, 1], positions[:, 0], color='red')
        ax.set_xlim([0, 1024])
        ax.set_ylim([0, 1024])
        ax.set_aspect('equal')
