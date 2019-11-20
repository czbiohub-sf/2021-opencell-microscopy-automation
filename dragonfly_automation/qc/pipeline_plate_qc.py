
import os
import re
import sys
import glob
import json
import time
import dask
import shutil
import tifffile
import datetime
import dask.diagnostics

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from dragonfly_automation import utils
from dragonfly_automation.qc.hcs_site_well_ids import hcs_site_well_ids

from pipeline_process.imaging import image


class PipelinePlateQC:

    def __init__(self, root_dir):
        '''
        root_dir is the top-level experiment directory
        (of the form 'dragonfly-automation-tests/ML0196_20191009/')
        '''

        self.root_dir = root_dir
        self.log_dir = os.path.join(self.root_dir, 'logs')
        self.raw_data_dir = os.path.join(self.root_dir, 'raw_data')

        self.exp_name = re.sub('%s+$' % os.sep, '', self.root_dir).split(os.sep)[-1]
        self.exp_id = self.exp_name.split('_')[0]

        if not os.path.isdir(self.log_dir):
            raise ValueError('No log directory found for %s' % self.exp_name)
            
        if not os.path.isdir(self.raw_data_dir):
            print('Warning: no raw data directory found for %s' % self.exp_name)
            self.raw_data_dir = None
    
        # the name of the FOV log dir in early datasets
        score_log_dir = os.path.join(self.log_dir, 'fov-classification')
        if os.path.isdir(score_log_dir):
            self.score_log_dir = score_log_dir
            self.score_log = pd.read_csv(os.path.join(score_log_dir, 'fov-classification-log.csv'))

        # the name of the FOV log dir in later datasets
        score_log_dir = os.path.join(self.log_dir, 'fov-scoring')        
        if os.path.isdir(score_log_dir):
            self.score_log_dir = score_log_dir
            self.score_log = pd.read_csv(os.path.join(score_log_dir, 'fov-score-log.csv'))
        
        # if there's no FOV log dir, we're in trouble
        if not hasattr(self, 'score_log_dir'):
            raise ValueError('No FOV log dir found for %s' % self.exp_name)
        
        # create a QC dir to which to save figures
        self.qc_dir = os.path.join(self.root_dir, 'QC')
        os.makedirs(self.qc_dir, exist_ok=True)

        # load the acquisition log
        self.aq_log = pd.read_csv(glob.glob(os.path.join(self.log_dir, '*acquired-images.csv'))[0])

        # load the global metadata
        with open(glob.glob(os.path.join(self.log_dir, '*experiment-metadata.json'))[0]) as file:
            self.exp_metadata = json.load(file)
    
        # possibly load the AFC log, which may not exist
        afc_log_filepath = glob.glob(os.path.join(self.log_dir, '*afc-calls.csv'))
        if not len(afc_log_filepath):
            self.afc_log = None
            print('Warning: no AFC log found for %s' % self.exp_name)
        else:
            self.afc_log = pd.read_csv(afc_log_filepath[0])

        # fix the FOV image filepaths in the FOV log
        filepaths = self.score_log.image_filepath.values
        self.score_log['filename'] = [
            os.path.join(self.score_log_dir, 'fov-images', path.split('\\')[-1]) for path in filepaths]

        # the number of channels acquired
        # (this is a bit hackish, but the acquire_bf_stacks flag was not always logged)
        self.num_channels = len(self.aq_log.config_name.unique())
        
        # the number of positions visited and scored in each well
        # (prior to 2019-11-19, this value was not explicitly logged anywhere, but was always 25)
        if 'position_site_num' in self.score_log.columns:
            self.num_sites_per_well = self.score_log.position_site_num.max() + 1
        else:
            print('Warning: assuming 25 sites per well in %s' % self.exp_name)
            self.num_sites_per_well = 25

        # append the well_id to the score_log if it is not present
        # (assumes a canonical 5x5 grid of positions in the region spanned by B2 to G9)
        if 'position_well_id' in self.score_log.columns:
            self.score_log.rename(columns={'position_well_id': 'well_id'}, inplace=True)
        else:
            print('Warning: manually appending well_ids to the score_log in %s' % self.exp_name)
            self.score_log = pd.merge(
                self.score_log,
                pd.DataFrame(data=hcs_site_well_ids), 
                left_on='position_ind', 
                right_on='ind', 
                how='inner')


    def summarize(self):

        # start time
        t0 = datetime.datetime.strptime(self.exp_metadata['setup_timestamp'], '%Y-%m-%d %H:%M:%S')

        # experiments that crashed may lack cleanup timestamps
        if self.exp_metadata.get('cleanup_timestamp'):
           tf = datetime.datetime.strptime(self.exp_metadata['cleanup_timestamp'], '%Y-%m-%d %H:%M:%S')    
        else:
            print('Warning: there is no cleanup_timestamp in the experiment metadata for %s' % self.exp_name)
            tf = t0

        total_seconds = (tf - t0).seconds
        hours = int(np.floor(total_seconds/3600))
        minutes = int(np.floor((total_seconds - hours*3600)/60))

        print(f'''
        Summary for {self.exp_id}
        Number of channels:        {self.num_channels}
        Number of sites per well:  {self.num_sites_per_well}
        Number of acquired FOVs:   {self.aq_log.shape[0]/self.num_channels}
        Number of scores > 0.5:    {(self.score_log.score > 0).sum()}
        Number of scores > -0.5:   {(self.score_log.score > -.5).sum()}
        Number of scored FOVs:     {self.score_log.shape[0]}
        Acquisition duration:      {hours}h{minutes}m
        ''')


    def plot_counts_and_scores(self, save_plot=False):
        '''
        Plot the median FOV score, max FOV score, and number of FOVs acquired per well
        '''
        fig, axs = plt.subplots(2, 1, figsize=(16, 6))

        d = self.aq_log.groupby('well_id').count().reset_index()
        axs[0].plot(d.well_id, d.timestamp/3)
        axs[0].set_ylim([0, 7])
        axs[0].set_title('Number of FOVs acquired per well')

        d = self.score_log.groupby('well_id').max().reset_index()
        axs[1].plot(d.well_id, d.score, label='max score')

        d = self.score_log.groupby('well_id').median().reset_index()
        axs[1].plot(d.well_id, d.score, label='median score')

        axs[1].set_ylim([-1.1, 1.1])
        plt.legend()
        axs[1].set_title('Median and maximum FOV score per well')
        plt.subplots_adjust(hspace=.3)

        if save_plot:
            plt.savefig(os.path.join(self.qc_dir, 'scores-and-counts.pdf'))



    def generate_z_projections(self):
        '''
        Generate z-projections of each channel of each acquired FOV
        '''
        
        dst_dirpath = os.path.join(self.qc_dir, 'z-projections')
        os.makedirs(dst_dirpath, exist_ok=True)

        filepaths = sorted(glob.glob(os.path.join(self.raw_data_dir, '*.ome.tif')))
        tasks = [dask.delayed(self.generate_z_projection)(dst_dirpath, filepath)
            for filepath in filepaths]
        
        with dask.diagnostics.ProgressBar():
            dask.compute(*tasks)


    @staticmethod
    def generate_z_projection(dst_dirpath, filepath):
        '''
        '''
        tiff = image.MicroManagerTIFF(filepath)
        tiff.parse_micromanager_metadata()
        channel_inds = tiff.mm_metadata.channel_ind.unique()
        for channel_ind in channel_inds:
            stack = np.array([
                tiff.tiff.pages[ind].asarray() 
                for ind in tiff.mm_metadata.loc[tiff.mm_metadata.channel_ind==channel_ind].page_ind
            ])
    
            filename = filepath.split(os.sep)[-1]
            dst_filename = filename.replace('.ome.tif', '_C%d-PROJ-Z.tif' % channel_ind)
            tifffile.imsave(os.path.join(dst_dirpath, dst_filename), stack.max(axis=0))


    def tile_acquired_fovs(self, channel_ind=0, save_plot=False):
        '''
        Plate-like tiled array of the top two FOVs in each well
        '''

        # hard-coded rows and columns for half-plate imaging
        rows = 'BCDEFG'
        cols = range(2, 10)

        blank_fov = np.zeros((1024, 1024), dtype='uint8')
        border = (np.ones((30, 1024))*255).astype('uint8')
        
        # acquired z-stacks (channel is arbitrary, since we will only need the well_ids and site_nums)
        dapi_aq_log = self.aq_log.loc[self.aq_log.config_name=='EMCCD_Confocal40_DAPI']

        # merge the FOV scores from the score_log
        dapi_aq_log = pd.merge(
            dapi_aq_log,
            self.score_log[['position_ind', 'score']], 
            left_on='position_ind',
            right_on='position_ind',
            how='left')
            
        fig, axs = plt.subplots(len(rows), len(cols), figsize=(20, 20))
        for row_ind, row in enumerate(rows):
            for col_ind, col in enumerate(cols):
                ax = axs[row_ind][col_ind]
                well_id = '%s%s' % (row, col)

                # the acquired FOVs for this well, sorted by score
                d = dapi_aq_log.loc[dapi_aq_log.well_id==well_id]
                d = d.sort_values(by='score', ascending=False).reset_index()
                
                # plot the top two FOVs
                ims = [blank_fov, blank_fov]
                for ind, d_row in d.iloc[:2].iterrows():

                    # HACK: this hard-coded filename must match
                    # the dst_filename in generate_z_projections
                    filename = f'MMStack_{d_row.position_ind}-{d_row.well_id}-{d_row.site_num}_C{channel_ind}-PROJ-Z.tif'
                    filepath = os.path.join(self.qc_dir, 'z-projections', filename)
                    if not os.path.isfile(filepath):
                        print('Warning: no z-projection found for %s' % filename)
                        continue

                    im = tifffile.imread(filepath)
                    ims[ind] = autogain(im, p=1)
    
                ax.imshow(np.concatenate((ims[0], border, ims[1]), axis=0), cmap='gray')
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title('%s (N = %s)' % (well_id, d.shape[0]))

        plt.subplots_adjust(left=.01, right=.99, top=.95, bottom=.01, wspace=0.01)
        plt.savefig(os.path.join(self.qc_dir, 'Tiled-FOVs-TOP2-C%d.pdf' % channel_ind))


    def rename_raw_tiffs(self, preview=True):
        '''
        Rename the acquired stacks to include the sample ('true') well_id
        and target name
        '''
        pass
    

def autogain(im, p=0):
    # HACK: this is more or less a copy from the same method
    # in opencell-process repo
    im = im.copy().astype(float)
    minn, maxx = np.percentile(im, (p, 100 - p))
    if minn==maxx:
        return (im * 0).astype('uint8')
        
    im = im - minn
    im[im < minn] = 0
    im = im/(maxx - minn)
    im[im > 1] = 1
    
    im = (im*255).astype('uint8')
    return im
