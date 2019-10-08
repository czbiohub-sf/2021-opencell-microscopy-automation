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
    Wrapper for instance methods called in self.classify_raw_fov
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

        # attempt the method call
        try:
            result = method(self, *args, **kwargs)

        except Exception as error:
            error_info = dict(method_name=method_name, error_message=str(error))
            
            # if an error has occured, we set the score to None
            # note that self.assign_score sets the score_has_been_assigned flag to True,
            # which will prevent the execution of all subsequent catch_errors-wrapped methods
            self.assign_score(
                score=None, 
                comment=("Error in method `FOVClassifier.%s`" % method_name),
                error_info=error_info)

        return result
    return wrapper


class FOVClassifier:

    def __init__(self, log_dir=None, mode='prediction', model_type='regression'):
        '''
        log_dir : str, optional
            path to a local directory in which to save log files
        mode : 'training' or 'prediction'

        model : 'classification' or 'regression'
        '''

        # whether the methods wrapped by catch_errors
        # can raise errors or not (explicitly set to False in self.classify_raw_fov)
        self.allow_errors = True

        if mode not in ['training', 'prediction']:
            raise ValueError("`mode` must be either 'training' or 'prediction'")
        self.mode = mode

        if model_type not in ['classification', 'regression']:
            raise ValueError("`model` must be either 'classification' or 'regression'")
        self.model_type = model_type

        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir

        self.cached_training_metadata = None
        self.current_training_metadata = None
        
        # an optional external event logger assigned after instantiation
        def dummy_event_logger(*args, **kwargs): pass
        self.external_event_logger = dummy_event_logger

        # hard-coded image size
        self.image_size = 1024

        # hard-coded feature order for the classifier
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
                oob_score=True)

        if self.model_type == 'regression':
            self.model = sklearn.ensemble.RandomForestRegressor(
                n_estimators=300,
                max_features='auto',
                oob_score=True)


    def training_data_filepath(self):
        return os.path.join(self.save_dir, 'training_data.csv')

    def training_metadata_filepath(self):
        return os.path.join(self.save_dir, 'training_metadata.json')


    def load(self, save_dir):
        '''
        Load existing training data and metadata

        save_dir : str, required
            location to which to save the training data, if in training mode, 
            or from which to load existing training data, if in prediction mode
        
        Steps
        1) load the training dataset and the cached metadata 
           (including cross-validation results)
        2) train the classifier (self.model)
        3) verify that the cross-validation results are comparable to the cached results
        '''

        self.save_dir = save_dir

        # reset the current validation results
        self.current_training_metadata = None
    
        # load the training data        
        self.training_data = pd.read_csv(self.training_data_filepath())

        # load the cached validation results
        if os.path.isfile(self.training_metadata_filepath()):
            with open(self.training_metadata_filepath(), 'r') as file:
                self.cached_training_metadata = json.load(file)
        else:
            print('Warning: no cached model metadata found')


    def save(self, save_dir=None, overwrite=False):
        '''
        Save the training data and the metadata

        If save_dir is None, save to the directory specified by self.save_dir
        (which is set in self.load)
        '''

        if save_dir is None and self.save_dir is None:
            raise ValueError('A save directory must be specified')

        if save_dir is None:
            save_dir = self.save_dir
        else:
            self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        if self.mode != 'training':
            raise ValueError("Cannot save training data unless mode = 'training'")
        
        # don't overwrite existing data
        filepath = self.training_data_filepath()
        if os.path.isfile(filepath) and not overwrite:
            raise ValueError('Training data already saved to %s' % self.save_dir)

        # save the training data
        self.training_data.to_csv(filepath, index=False)
        print('Training data saved to %s' % filepath)

        # save the metadata
        if self.current_training_metadata is None:
            print('Warning: no metadata found to save')
        else:
            self.current_training_metadata['filepath'] = os.path.abspath(filepath)
            with open(self.training_metadata_filepath(), 'w') as file:
                json.dump(self.current_training_metadata, file)
            print('Metadata saved to %s' % self.training_metadata_filepath())


    def process_training_data(self, data):
        '''
        Calculate the features from the training data images,
        and drop any images that are either not candidates or that yield errors

        Note that there is minimal error handling, since the training data has been curated
        and we should be able to find nuclei in every image without errors occurring

        Parameters
        ----------
        data : a pd.DataFrame with a 'filename' column and one or more label columns

        '''

        # create columns for the calculated features
        for feature_name in self.feature_order:
            data[feature_name] = None

        for ind, row in data.iterrows():
            printr(row.filename)
            im = tifffile.imread(row.filename)
            mask = self.generate_background_mask(im)
            positions = self.find_nucleus_positions(mask)
            features = self.calculate_features(positions)
            for feature_name, feature_value in features.items():
                data.at[ind, feature_name] = feature_value

        self.training_data = data


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
        # note that, for now, we *do not* drop images 
        # that would not pass the is_fov_candidate test
        data = self.training_data.copy()
        mask = data[list(self.feature_order)].isna().sum(axis=1)
        if mask.sum():
            print('\nWarning: some training data is missing features and will be dropped (see self.dropped_data)')
            self.dropped_data = data.loc[mask > 0]
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

        print('Cached and current training data shape: (%s, %s)' % (
            self.cached_training_metadata['training_data_shape'],
            self.current_training_metadata['training_data_shape']))

        print('Cached and current oob_score: (%s, %s)' % (
            self.cached_training_metadata['oob_score'],
            self.current_training_metadata['oob_score']))

        # make sure the number of features is consistent 
        self.allow_errors = True
        mock_positions = np.array([[100, 500], [500, 500], [500, 100]])
        features = self.calculate_features(mock_positions)
        score = self.predict_score(features)
        print('Mock predicted score (should be falsy): %s' % score)


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


    def score_raw_fov(self, image, position_ind=None):
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
          - if the image is random noise, then we usually get 'not a candidate'
          - if there is some giant but uniform artifact that doesn't break the thresholding
            (e.g., some large region is uniformly zero), we'll never know

        Parameters
        ----------
        image : numpy.ndarray (2D and uint16)
            The raw field of view to classify; assumed to be close to in-focus
        position_ind : int, optional (but required for logging)
            The index of the current position
            Note that we use an index, and not a label, because it's not clear
            that we can assume that the labels in the position list will always be unique 
            (e.g., they may not be when the position list is generated manually,
            rather than by the HCS Site Generator plugin)
        
        '''

        # required for the catch_errors wrapper to actually catch (and log) errors
        self.allow_errors = False

        # reset the 'state' of the score-assignment logic
        self.assigned_score = None
        self.score_has_been_assigned = False

        # reset the log info
        # note that this dict is modified in this method *and* in self.assign_score
        self.log_info = {
            'position_ind': position_ind
        }

        # validate the image: check that it's a 2D uint16 ndarray
        # note that, because validate_raw_fov is wrapped by the catch_errors method,
        # we must check that the validation_result is not None before accessing the 'flag' key.
        # also note that we only include the image in self.log_info if it passes validation
        # (otherwise, errors may result when the image itself is logged in self.save_log_info)
        validation_result = self.validate_raw_fov(image)
        if validation_result is not None:
            if validation_result.get('flag'):
                self.log_info['raw_image'] = image
            else:
                self.assign_score(score=None, comment=validation_result.get('message'))
    
        # check whether there are any nuclei in the FOV
        nuclei_in_fov = self.are_nuclei_in_fov(image)
        if not nuclei_in_fov:
            self.assign_score(score=None, comment='No nuclei in the FOV')

        # calculate the background mask and nucleus positions
        mask = self.generate_background_mask(image)
        positions = self.find_nucleus_positions(mask)
        self.log_info['positions'] = positions

        # determine if the FOV is a candidate
        fov_is_candidate = self.is_fov_candidate(positions)
        if not fov_is_candidate:
            self.assign_score(score=None, comment='FOV is not a candidate')

        # calculate features from the positions
        features = self.calculate_features(positions)
        self.log_info['features'] = features

        # finally, use the trained model to generate a prediction
        score = self.predict_score(features)
        self.log_info['score'] = score
        if score is not None:
            self.assign_score(score, comment='Model prediction')
    
        # log everything we've accumulated in self.log_info
        self.save_log_info()
        return self.assigned_score


    def assign_score(self, score, comment, error_info=None):
        '''
        '''
        # do nothing if a score has already been assigned
        if self.score_has_been_assigned:
            return

        self.assigned_score = score
        self.score_has_been_assigned = True

        # update the log
        self.log_info['score'] = score
        self.log_info['comment'] = comment
        self.log_info['error_info'] = error_info


    def save_log_info(self):
        '''
        '''
        
        log_info = self.log_info
        error_info = log_info.get('error_info')

        # message for the external event logger 
        comment = log_info.get('comment')
        score = log_info.get('score')
        score = '%0.2f' % score if score is not None else score
        event_log_message = "CLASSIFIER INFO: The FOV score was %s (comment: '%s')" % (score, comment)
        self.external_event_logger(event_log_message)

        # if there's no log dir, we fall back to printing the score and error (if any)
        if self.log_dir is None:
            print(event_log_message)
            if error_info is not None:
                print("Error during classification in method `%s`: '%s'" % \
                    (error_info.get('method_name'), error_info.get('error_message')))
            return

        # if we're still here, we need a position_ind
        position_ind = log_info.get('position_ind')
        if position_ind is None:
            print('Warning: a position_ind must be provided to log classification info')
            return

        # directory and filepaths for logged images
        image_dir = os.path.join(self.log_dir, 'fov-images')
        os.makedirs(image_dir, exist_ok=True)
        def logged_image_filepath(tag):
            return os.path.join(image_dir, 'FOV_%05d_%s.tif' % (position_ind, tag))
        
        # construct the CSV log row
        row = {
            'score': score,
            'comment': comment,
            'position_ind': position_ind,
            'timestamp': utils.timestamp(),
            'image_filepath': logged_image_filepath('RAW'),
        }

        # dict of {method_name, error_message}
        if error_info is not None:
            row.update(error_info)

        # dict of {num_nuclei, com_offset, etc}
        features = log_info.get('features')
        if features is not None:
            row.update(features)

        # append the row to the log file
        log_filepath = os.path.join(self.log_dir, 'fov-classification-log.csv')
        if os.path.isfile(log_filepath):
            log = pd.read_csv(log_filepath)
            log = log.append(row, ignore_index=True)
        else:
            log = pd.DataFrame([row])
        log.to_csv(log_filepath, index=False, float_format='%0.2f')

        # log the raw image itself
        raw_image = log_info.get('raw_image')
        if raw_image is not None:
            tifffile.imsave(logged_image_filepath('RAW'), raw_image)

            # create a uint8 version 
            # (uint16 images can't be opened in Windows image preview)
            scaled_image = utils.to_uint8(raw_image)
            tifffile.imsave(logged_image_filepath('UINT8'), scaled_image)

            # create and save the annotated image
            # (in which nucleus positions are marked with white squares)
            positions = log_info.get('positions')
            if positions is not None:
                
                # spot width and whitepoint intensity
                width = 3
                white = 255
                sz = scaled_image.shape

                # lower the brightness of the autogained image 
                # so that the squares are easier to see
                ant_image = (scaled_image/2).astype('uint8')

                # draw a square on the image at each nucleus position
                for pos in positions:
                    ant_image[
                        int(max(0, pos[0] - width)):int(min(sz[0] - 1, pos[0] + width)), 
                        int(max(0, pos[1] - width)):int(min(sz[1] - 1, pos[1] + width))
                    ] = white

                tifffile.imsave(logged_image_filepath('ANT'), ant_image)


    @catch_errors
    def validate_raw_fov(self, image):
        
        flag = False
        message = None
        if not isinstance(image, np.ndarray):
            message = 'Image is not an np.ndarray'
        elif image.dtype != 'uint16':
            message = 'Image is not uint16'
        elif image.ndim != 2:
            message = 'Image is not 2D'
        elif image.shape[0] != self.image_size or image.shape[1] != self.image_size:
            message = 'Image shape is not (%s, %s)' % (self.image_size, self.image_size)
        else:
            flag = True
        
        return dict(flag=flag, message=message)


    @catch_errors
    def are_nuclei_in_fov(self, image):
        '''
        Check whether there are *any* real nuclei in the image

        This is accomplished by using an empirically-determine minimum Otsu threshold,
        which is predicated on the observation that that the background intensity 
        in raw FOVs is around 500.

        *** Note that this value is sensitive to the exposure settings! ***
        (presumably, mostly the exposure time and the camera gain)
        '''
    
        min_otsu_thresh = 1000
        otsu_thresh = skimage.filters.threshold_li(image)
        nuclei_in_fov = otsu_thresh > min_otsu_thresh
        return nuclei_in_fov


    @catch_errors
    def is_fov_candidate(self, positions):
        '''
        Check whether there are way too few or way too many 'nuclei' in the FOV

        This will occur if either
           1) there are very few or very many real nuclei in the FOV, or
           2) there are _no_ real nuclei in the FOV,
              and the nucleus positions correspond to noise, dust, etc
              (we attempt to defend against this by first calling are_nuclei_in_fov)
        '''

        is_candidate = True
        min_num_nuclei = 10
        max_num_nuclei = 100

        num_positions = positions.shape[0]
        if num_positions < min_num_nuclei or num_positions > max_num_nuclei:
            is_candidate = False

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
                mask[mask_label==prop.label] = False

        return mask


    @catch_errors
    def find_nucleus_positions(self, mask):
        '''

        '''

        nucleus_radius = 15

        # smoothed distance transform
        dist = ndimage.distance_transform_edt(mask)
        distf = skimage.filters.gaussian(dist, sigma=1)

        # the positions of the local maximima in the distance transform
        # correspond roughly to the centers of mass of the individual nuclei
        positions = skimage.feature.peak_local_max(
            distf, indices=True, min_distance=nucleus_radius, labels=mask)

        return positions    


    @catch_errors
    def calculate_features(self, positions):
        '''
        '''

        features = {}

        # -------------------------------------------------------------------------
        #
        # Simple aggregate features of the nucleus positions
        # (center of mass and asymmetry) 
        #
        # -------------------------------------------------------------------------

        # the number of nuclei
        num_nuclei = positions.shape[0]
        
        # the distance of the center of mass from the center of the image
        com_offset = ((positions.mean(axis=0) - (self.image_size/2))**2).sum()**.5
        rel_com_offset = com_offset / self.image_size

        # eigenvalues of the covariance matrix
        evals, evecs = np.linalg.eig(np.cov(positions.transpose()))

        # the ratio of eigenvalues is a measure of asymmetry 
        eval_ratio = (max(evals) - min(evals))/min(evals)

        features.update({
            'num_nuclei': num_nuclei, 
            'com_offset': rel_com_offset, 
            'eval_ratio': eval_ratio,
        })

        # -------------------------------------------------------------------------
        #
        # Features derived from a simulated nucleus mask
        # (obtained by thresholding the distance transform of the nucleus positions)
        #
        # Note that it is necessary to simulate a nucleus mask,
        # rather than use the mask returned by _generate_background_mask, 
        # because the area of the foreground regions in the generated mask
        # depends on the focal plane.
        #
        # -------------------------------------------------------------------------

        # the nucleus radius here was selected empirically
        nucleus_radius = 50

        position_mask = np.zeros((self.image_size, self.image_size))
        for position in positions:
            position_mask[position[0], position[1]] = 1

        dt = ndimage.distance_transform_edt(~position_mask.astype(bool))
        total_area = (dt < nucleus_radius).sum() / (self.image_size*self.image_size)
        max_distance = dt.max()

        features.update({
            'total_area': total_area, 
            'max_distance': max_distance
        })

        # -------------------------------------------------------------------------
        #
        # Cluster positions using DBSCAN 
        # and calculate various measures of cluster homogeneity
        #
        # -------------------------------------------------------------------------
        
        # empirically-selected neighborhood size in pixels
        # (the clustering is *very* sensitive to changes in this parameter)
        eps = 100

        # min_samples = 3 is the minimum required for non-trivial clustering
        min_samples = 3

        dbscan = sklearn.cluster.DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
        dbscan.fit(positions)

        labels = dbscan.labels_
        num_clusters = len(set(labels))
        num_unclustered = (labels==-1).sum()

        features.update({
            'num_clusters': num_clusters, 
            'num_unclustered': num_unclustered, 
        })

        return features


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
                skimage.color.label2rgb(~mask, image=utils.to_uint8(im), colors=('black', 'yellow')))

        # plot the positions themselves
        ax.scatter(positions[:, 1], positions[:, 0], color='red')
        ax.set_xlim([0, 1024])
        ax.set_ylim([0, 1024])
        ax.set_aspect('equal')

