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


class FOVClassifier(object):

    def __init__(self, mode='train'):
        '''
        mode : 'train' or 'predict'
        '''
        pass


    @classmethod
    def load(save_dir):
        '''
        Load from an existing cached sklearn classifier instance

        Steps
        1) load the cached training dataset, the trained model, and the cross-validation results
        2) verify that the trained model is compatible with self.extract_features (how? requires a test FOV)
        3) verify the cached cross-validation results by re-training the model using the cached training data
        '''
        pass


    def save(self, save_dir):
        '''
        Cache/save the training dataset, 
        trained sklearn classifier instance, and cross-validation results

        Intended for use only after a model has been trained when mode='train'
        '''
        pass


    def classify(self, image):
        '''
        Classify a candidate FOV
        Assumes that self.model exists
        (e.g., that either mode='train' or we instantiated from a cached trained model)
        '''
        pass


    def train(self, label):
        '''
        Train and cross-validate a classifier to predict the given label
    
        dataset: a pd.DataFrame with all feature and label columns
        label : the label to predict (must be a boolean column in the dataset)
        '''

        # generate X from the dataset DataFrame (requires ordering features)
        # generate y - should be something like `y = dataset[label]`
    

    def log_prod_error(self, message):
        '''
        Log an error during production
        '''
        pass


    @staticmethod
    def preprocess_fov(image):
        '''
        Basic quality/sanity checks
        Checks whether there are either
        - no nuclei in the image
        - way too few or way too many nuclei in the image

        Returns
        -------
        is_candidate : bool
            whether the FOV is a candidate or not
        positions : np array
            the nucleus positions (None if is_candidate is False)
        '''
    

    @staticmethod
    def find_nucleus_positions(image):
        pass
    
    
    @staticmethod
    def calculate_features(positions):
        '''
        '''
        pass

