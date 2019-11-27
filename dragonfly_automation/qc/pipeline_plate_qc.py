
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
import jsonschema
import dask.diagnostics

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from dragonfly_automation.qc import half_plate_layout
from dragonfly_automation.qc.hcs_site_well_ids import hcs_site_well_ids

try:
    from pipeline_process.imaging import image
except ImportError:
    print('Warning: pipeline_process package not found')


# schema for the manually-defined metadata required for each pipeline_plate acquisition
EXTERNAL_METADATA_SCHEMA = {
    'type': 'object',
    'properties': {

        # one of 'first-half', 'second-half', 'full', 'custom'
        # (if 'custom', a custom platemap must exist at '<root_dir>/custom_platemap.csv';
        # if not 'custom', all of the properties below are required)
        'platemap_type': {'type': 'string'},

        # the parental line (as of Nov 2019, always 'smNG' for 'split mNeonGreen')
        'parental_line': {'type': 'string'},

        # the plate_id (of the form 'P0001')
        'plate_id': {'type': 'string'},

        # electroporation number (as of Nov 2019, always 'EP01')
        'electroporation_id': {'type': 'string'},

        # imaging round number corresponds to freeze-thaw count
        # (e.g., the first time imaging a thawed plate is 'R02')
        'imaging_round_id': {'type': 'string'},

    },
    'required': ['platemap_type'],
}


class PipelinePlateQC:

    def __init__(self, root_dir):
        '''
        root_dir is the top-level experiment directory
        (of the form 'dragonfly-automation-tests/ML0196_20191009/')

        '''

        self.root_dir = root_dir

        # load the user-defined external/global metadata
        # this specifies the platemap_type, plate_id, ep_id, and imaging_round_id
        metadata_filepath = os.path.join(self.root_dir, 'metadata.json')
        if not os.path.isfile(metadata_filepath):
            raise ValueError("No user-defined 'metadata.json' file found")

        with open(metadata_filepath, 'r') as file:
            self.external_metadata = json.load(file)
        self.validate_external_metadata(self.external_metadata)
        
        # load the platemap according to the platemap_type property defined in external_metadata
        # (that is, either the platemap for canonical half-plate imaging
        # or a custom platemap provided by the user)
        self.load_platemap()

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


    @staticmethod
    def validate_external_metadata(external_metadata):
        '''
        Validation for the user-defined external metadata

        This validation is important because the external metadata file
        is a freely-edited JSON file, so we need to check for typos etc
        '''
        md = external_metadata
        
        # raises a ValidationError if validation fails
        jsonschema.validate(md, EXTERNAL_METADATA_SCHEMA)

        if md['platemap_type'] not in ['first-half', 'second-half', 'custom']:
            raise ValueError("`platemap_type` must be one of 'first-half', 'second-half', or 'custom'")

        # if there is a custom platemap, no other external_metadata properties are required        
        if md['platemap_type'] == 'custom':
            return
            
        # check the plate_id
        result = re.match(r'^P[0-9]{4}$', md['plate_id'])
        if not result:
            raise ValueError('Invalid plate_id %s')
    
        # TODO: check the ep_id and the round_id


    def load_and_validate_custom_platemap(self):
        '''
        Custom platemaps are intended for manual-redo imaging
        in which an arbtrary subset of wells on the imaging plate are imaged, 
        each of which may correspond to an arbitrary pipeline well_id from *any* pipeline plate

        The total absence of constraints/assumptions requires that all of the metadata properties
        that are 'normally' defined in the external_metadata object (see external_metadata_schema)
        be explicitly defined in the platemap for each well

        '''
        platemap_filepath = glob.glob(os.path.join(self.root_dir, '*platemap.csv'))
        if len(platemap_filepath) != 1:
            raise ValueError("Exactly one custom platemap must exist when platemap_type is 'custom'")
        platemap = pd.read_csv(platemap_filepath[0])

        # check for required and unexpected columns
        # (there must be a column for each property 
        # that would otherwise have been defined in the external_metadata)
        required_columns = set(EXTERNAL_METADATA_SCHEMA['properties'].keys()).difference(['platemap_type'])
        if required_columns.difference(platemap.columns):
            raise ValueError('Missing columns in the custom platemap')

        # TODO: validate plate_id, ep_id, and round_id columns
        # TODO: check that the well_ids are all valid
        return platemap


    def load_platemap(self):
        '''
        '''

        # construct the platemap from imaging_well_id to pipeline_well_id (i.e., the 'true' well_id)


        if self.external_metadata['platemap_type'] != 'custom':

            if self.external_metadata['platemap_type'] == 'first-half':
                platemap = pd.DataFrame(data=half_plate_layout.first_half)

            if self.external_metadata['platemap_type'] == 'second-half':
                platemap = pd.DataFrame(data=half_plate_layout.second_half)

            for key, value in self.external_metadata.items():
                if key == 'platemap_type':
                    continue
                platemap[key] = value

        else:
            platemap = self.load_and_validate_custom_platemap()

        self.platemap = platemap


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

        plate_ids = self.external_metadata.get('plate_id') or self.platemap.plate_id.unique()
        print(f'''
        Summary for {self.exp_id} 
        Platemap:                  {self.external_metadata['platemap_type']}
        Plate(s):                  {plate_ids}
        Number of channels:        {self.num_channels}
        Number of sites per well:  {self.num_sites_per_well}
        Number of acquired FOVs:   {self.aq_log.shape[0]/self.num_channels}
        Number of scores > 0.5:    {(self.score_log.score > 0).sum()}
        Number of scores > -0.5:   {(self.score_log.score > -.5).sum()}
        Number of unscored FOVs:   {(self.score_log.score.isna()).sum()}
        Number of visited FOVs:    {self.score_log.shape[0]}
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

        TODO: for now, assumes canonical half-plate layout
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
                
                # the pipeline well_id
                pipeline_well_id = self.half_plate_imaging_well_to_pipeline_well(well_id)

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
                    ims[ind] = self.autogain(im, p=1)
    
                ax.imshow(np.concatenate((ims[0], border, ims[1]), axis=0), cmap='gray')
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title('%s (N = %s) %s' % (well_id, d.shape[0], pipeline_well_id))

        plt.subplots_adjust(left=.01, right=.99, top=.95, bottom=.01, wspace=0.01)
        plt.savefig(os.path.join(self.qc_dir, 'Tiled-FOVs-TOP2-C%d.pdf' % channel_ind))


    def half_plate_imaging_well_to_pipeline_well(self, imaging_well_id):

        if imaging_well_id not in self.platemap.imaging_well_id.values:
            print('Warning: imaging_well_id %s not found in the platemap' % imaging_well_id)
            return 'unknown'

        row = self.platemap.loc[self.platemap.imaging_well_id==imaging_well_id].iloc[0]
        return row.pipeline_well_id


    def rename_raw_tiffs_from_half_plate(self, plate_num, imaging_round_num, preview=True):
        '''
        Rename the acquired stacks to include the sample ('true') well_id
        and target name
        '''
        

        # all of the raw TIFFs
        src_filepaths = sorted(glob.glob(os.path.join(self.raw_data_dir, '*.ome.tif')))

        # create plate_id and imaging_round_id
        plate_id = 'P%04d' % plate_num
        imaging_round_id = 'R%02d' % imaging_round_num

        # hard-coded electroporation ID
        ep_id = 'EP01'

        # hard-coded parental line
        parental_line = 'smNG'

        dst_filenames = []
        src_filenames = [filepath.split(os.sep)[-1] for filepath in src_filepaths]
        for src_filename in src_filenames:

            # parse the raw TIFF filename
            imaging_well_id, site_num = self.parse_raw_tiff_filename(src_filename)
            
            # the pipeline plate well_id that corresponds to the imaging well_id
            well_id = self.half_plate_imaging_well_to_pipeline_well(imaging_well_id)

            # pad the well_id
            well_id = self.pad_well_id(well_id)

            # create the site_id
            site_id = 'S%02d' % site_num

            # look up the target name using the plate_id and well_id
            target_name = 'target_name'

            dst_filenames.append(
                f'{parental_line}-{plate_id}-{ep_id}-{imaging_round_id}-{self.exp_id}-{well_id}-{site_id}-{target_name}.ome.tif'
            )

        return list(zip(src_filenames, dst_filenames))


    @staticmethod
    def parse_raw_tiff_filename(filename):
        '''
        Parse the well_id and site_num from a raw TIFF filename, which is of the form
        'MMStack_{position_ind}-{well_id}-{site_num}.ome.tif'
        '''

        result = re.match('^MMStack_([0-9]+)-([A-H][1-9][0-2]?)-([0-9]+).ome.tif$', filename)
        if not result:
            print('Warning: cannot parse raw TIFF filename %s' % filename)
            return None, None

        position_ind, well_id, site_num = result.groups()
        return well_id, int(site_num)

    
    @staticmethod
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


    @staticmethod
    def pad_well_id(well_id):
        '''
        'A1' -> 'A01'
        '''
        well_row, well_col = re.match('([A-H])([1-9][0-2]?)', well_id).groups()
        well_id = '%s%02d' % (well_row, int(well_col))
        return well_id