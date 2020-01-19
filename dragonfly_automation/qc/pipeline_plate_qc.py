
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

# local opencell repo
sys.path.append('/Users/keith.cheveralls/projects/opencell')
# opencell repo from a docker container on ESS
sys.path.append('/home/projects/opencell')
from opencell.imaging import images


# schema for the manually-defined metadata required for each pipeline_plate acquisition
EXTERNAL_METADATA_SCHEMA = {
    'type': 'object',
    'properties': {
        
        # the manually-generated experiment ID (of the form 'PML0001')
        'pml_id': {'type': 'string'},

        # one of 'first-half', 'second-half', 'full', 'custom'
        # if 'custom', a custom platemap must exist at '<root_dir>/custom_platemap.csv'
        # if not 'custom', all of the properties below are required
        'platemap_type': {'type': 'string'},

        # the parental line (as of Jan 2020, always 'czML0383')
        'parental_line': {'type': 'string'},

        # the plate_id (of the form 'P0001')
        'plate_id': {'type': 'string'},

        # imaging round number corresponds to freeze-thaw count
        # (e.g., the first time imaging a thawed plate is 'R02')
        'imaging_round_id': {'type': 'string'},

    },
    'required': ['pml_id', 'platemap_type'],
}


class PipelinePlateQC:

    def __init__(self, root_dir):
        '''
        root_dir is the top-level experiment directory
        (of the form 'ml_group/KC/dragonfly-automation-tests/PML0196/')

        '''

        self.root_dir = root_dir

        # the directory name should be the same as the pml_id
        self.root_dirname = re.sub('%s+$' % os.sep, '', self.root_dir).split(os.sep)[-1]

        # load the user-defined external/global metadata
        # this specifies the platemap_type, plate_id, ep_id, and imaging_round_id
        metadata_filepath = os.path.join(self.root_dir, 'metadata.json')
        if not metadata_filepath:
            raise ValueError("No user-defined 'metadata.json' file found")

        with open(metadata_filepath, 'r') as file:
            self.external_metadata = json.load(file)
        self.validate_external_metadata(self.external_metadata)
        
        # load the platemap according to the platemap_type property defined in external_metadata
        # (that is, either the platemap for canonical half-plate imaging
        # or a custom platemap provided by the user)
        self.load_platemap()

        # the log dir (which must exist)
        self.log_dir = os.path.join(self.root_dir, 'logs')
        if not os.path.isdir(self.log_dir):
            raise ValueError('No log directory found in %s' % self.root_dir)
            
        # the raw data subdir (which may not exist)
        self.raw_data_dir = os.path.join(self.root_dir, 'raw_data')
        if not os.path.isdir(self.raw_data_dir):
            print('Warning: no raw data directory found in %s' % self.root_dir)
            self.raw_data_dir = None
    
        # the name of the FOV-scoring log dir in early datasets
        score_log_dir = os.path.join(self.log_dir, 'fov-classification')
        if os.path.isdir(score_log_dir):
            self.score_log_dir = score_log_dir
            self.score_log = pd.read_csv(os.path.join(score_log_dir, 'fov-classification-log.csv'))

        # the name of the FOV-scoring log dir in later datasets
        score_log_dir = os.path.join(self.log_dir, 'fov-scoring')        
        if os.path.isdir(score_log_dir):
            self.score_log_dir = score_log_dir
            self.score_log = pd.read_csv(os.path.join(score_log_dir, 'fov-score-log.csv'))
        
        # rarely, for acquisitions in which FOVs were selected manually,
        # there is no FOV-scoring log
        if not hasattr(self, 'score_log_dir'):
            print('Warning: no FOV log dir found in %s' % self.root_dir)
            self.has_score_log = False
        else:
            self.has_score_log = True
            self.score_log_summary = self.parse_score_log()

        # load the acquisition log
        self.aq_log = pd.read_csv(glob.glob(os.path.join(self.log_dir, '*acquired-images.csv'))[0])

        # load the global metadata
        with open(glob.glob(os.path.join(self.log_dir, '*experiment-metadata.json'))[0]) as file:
            self.exp_metadata = json.load(file)
    
        # possibly load the AFC log, which may not exist
        afc_log_filepath = glob.glob(os.path.join(self.log_dir, '*afc-calls.csv'))
        if not len(afc_log_filepath):
            self.afc_log = None
            print('Warning: no AFC log found in %s' % self.root_dir)
        else:
            self.afc_log = pd.read_csv(afc_log_filepath[0])

        # the number of channels acquired
        # (this is a bit hackish, but the acquire_bf_stacks flag was not always logged)
        self.num_channels = len(self.aq_log.config_name.unique())

        # create a QC directory to which to save figures
        self.qc_dir = os.path.join(self.root_dir, 'QC')
        os.makedirs(self.qc_dir, exist_ok=True)

    
    def parse_score_log(self):
        '''
        Extract statistics from the log of FOV scores
        '''

        # fix the FOV image filepaths in the FOV log
        filenames = [path.split('\\')[-1] for path in self.score_log.image_filepath]
        self.score_log['filename'] = [
            os.path.join(self.score_log_dir, 'fov-images', filename) for filename in filenames]

        # the number of positions visited and scored in each well
        # (prior to 2019-11-19, this value was not explicitly logged anywhere, but was always 25)
        if 'position_site_num' in self.score_log.columns:
            num_sites_per_well = self.score_log.position_site_num.max() + 1
        else:
            print('Warning: assuming 25 sites per well in %s' % self.root_dirname)
            num_sites_per_well = 25

        # append the well_id to the score_log if it is not present
        # (assumes a canonical 5x5 grid of positions in the region spanned by B2 to G9)
        if 'position_well_id' in self.score_log.columns:
            self.score_log.rename(columns={'position_well_id': 'well_id'}, inplace=True)
        else:
            print('Warning: manually appending well_ids to the score_log in %s' % self.root_dirname)
            self.score_log = pd.merge(
                self.score_log,
                pd.DataFrame(data=hcs_site_well_ids), 
                left_on='position_ind', 
                right_on='ind', 
                how='inner')

        # summary of the score log
        num_fovs = self.score_log.shape[0]
        summary = {
            'num_visited_fovs': num_fovs,
            'num_sites_per_well': num_sites_per_well,
            'pct_scores_gt_phalf': int(100*(self.score_log.score > .5).sum() / num_fovs),
            'pct_scores_le_nhalf': int(100*(self.score_log.score < -.5).sum() / num_fovs),
            'pct_unscored_fovs': int(100*(self.score_log.score.isna()).sum() / num_fovs),
        }

        return summary



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

        # validate the pml_id
        result = re.match(r'^PML[0-9]{4}$', md['pml_id'])
        if not result:
            raise ValueError('Invalid pml_id %s in %s' % (md['pml_id'], self.root_dirname))
        
        # validate the platemap_type
        if md['platemap_type'] not in ['first-half', 'second-half', 'custom']:
            raise ValueError("Invalid platemap_type '%s' in %s" % (md['platemap_type'], self.root_dirname))

        # if there is a custom platemap, no other external metadata properties are required        
        if md['platemap_type'] == 'custom':
            return
            
        # validate the plate_id
        result = re.match(r'^P[0-9]{4}$', md['plate_id'])
        if not result:
            raise ValueError('Invalid plate_id %s in %s' % (md['plate_id'], self.root_dirname))
    
        # TODO: validate the parental_line and imaging_round_id


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
        if not platemap_filepath:
            raise ValueError('No custom platemap found in %s' % self.root_dirname)
        if len(platemap_filepath) > 1:
            print('Warning: more than one custom platemap found in %s' % self.root_dirname)
        platemap = pd.read_csv(platemap_filepath[0])

        # check for required and unexpected columns
        # (there must be a column for each property 
        # that would otherwise have been defined in the external_metadata)
        required_columns = set(EXTERNAL_METADATA_SCHEMA['properties'].keys())\
            .difference(['pml_id', 'platemap_type'])
    
        if required_columns.difference(platemap.columns):
            raise ValueError('Missing columns in the custom platemap in %s' % self.root_dir)

        # TODO: validate plate_id, ep_id, and round_id columns
        # TODO: check that the well_ids are all valid
        return platemap


    def load_platemap(self):
        '''
        Load the platemap as a pandas dataframe
        that maps imaging_well_id to pipeline_well_id (i.e., the 'true' well_id)
        '''

        if self.external_metadata['platemap_type'] != 'custom':
            if self.external_metadata['platemap_type'] == 'first-half':
                platemap = pd.DataFrame(data=half_plate_layout.first_half)
            if self.external_metadata['platemap_type'] == 'second-half':
                platemap = pd.DataFrame(data=half_plate_layout.second_half)
        else:
            platemap = self.load_and_validate_custom_platemap()

        # append the remaining global metadata attributes
        # (for first-half or second-half, this includes the parental_line and the plate_id)
        for key, value in self.external_metadata.items():
            platemap[key] = value

        self.platemap = platemap


    def summarize(self):

        # start time
        t0 = datetime.datetime.strptime(self.exp_metadata['setup_timestamp'], '%Y-%m-%d %H:%M:%S')

        # experiments that crashed may lack cleanup timestamps
        if self.exp_metadata.get('cleanup_timestamp'):
           tf = datetime.datetime.strptime(self.exp_metadata['cleanup_timestamp'], '%Y-%m-%d %H:%M:%S')    
        else:
            print('Warning: there is no cleanup_timestamp in the experiment metadata in %s' % self.root_dirname)
            tf = t0

        total_seconds = (tf - t0).seconds
        hours = int(np.floor(total_seconds/3600))
        minutes = int(np.floor((total_seconds - hours*3600)/60))
        plate_ids = self.external_metadata.get('plate_id') or self.platemap.plate_id.unique()

        summary = {
            'timestamp': self.exp_metadata['setup_timestamp'],
            'plate_id': plate_ids,
            'platemap_type': self.external_metadata['platemap_type'],
            'num_channels': self.num_channels,
            'num_acquired_fovs': self.aq_log.shape[0]/self.num_channels,
            'acquisition_duration': f'{hours}h{minutes}m',
        }

        # append the score log summary
        if self.has_score_log:
            summary.update(self.score_log_summary)

        print(f'Summary for {self.root_dirname}')
        for key, val in summary.items():
            print(f'{key:<25}{val}')


    def plot_counts_and_scores(self, save_plot=False):
        '''
        Plot the median FOV score, max FOV score, and number of FOVs acquired per well
        '''
        fig, axs = plt.subplots(2, 1, figsize=(16, 6))

        ax = axs[0]
        ax.set_title('Number of FOVs acquired per well')
        d = self.aq_log.groupby('well_id').count().reset_index()
        ax.plot(d.well_id, d.timestamp/self.num_channels)
        ax.set_ylim([0, 7])

        ax = axs[1]
        ax.set_title('Median and maximum FOV score per well')
        if self.has_score_log:
            d = self.score_log.groupby('well_id').max().reset_index()
            ax.plot(d.well_id, d.score, label='max score')

            d = self.score_log.groupby('well_id').median().reset_index()
            ax.plot(d.well_id, d.score, label='median score')

            ax.set_ylim([-1.1, 1.1])
            plt.subplots_adjust(hspace=.3)
            plt.legend()

        if save_plot:
            plt.savefig(os.path.join(self.qc_dir, 'scores-and-counts.pdf'))


    def generate_z_projections(self):
        '''
        Generate z-projections of each channel of each acquired FOV
        '''
        
        dst_dirpath = os.path.join(self.qc_dir, 'z-projections')
        os.makedirs(dst_dirpath, exist_ok=True)

        src_filepaths = sorted(glob.glob(os.path.join(self.raw_data_dir, '*.ome.tif')))
        tasks = [dask.delayed(self.generate_z_projection)(src_filepath, dst_dirpath)
            for src_filepath in src_filepaths]
        
        with dask.diagnostics.ProgressBar():
            dask.compute(*tasks)


    @staticmethod
    def generate_z_projection(src_filepath, dst_dirpath):
        '''
        '''
        tiff = images.MicroManagerTIFF(src_filepath)
        tiff.parse_micromanager_metadata()
        channel_inds = tiff.mm_metadata.channel_ind.unique()
        for channel_ind in channel_inds:
            stack = np.array([
                tiff.tiff.pages[ind].asarray() 
                for ind in tiff.mm_metadata.loc[tiff.mm_metadata.channel_ind==channel_ind].page_ind
            ])
    
            src_filename = src_filepath.split(os.sep)[-1]
            dst_filename = src_filename.replace('.ome.tif', '_PROJ-CH%d.tif' % channel_ind)
            tifffile.imsave(os.path.join(dst_dirpath, dst_filename), stack.max(axis=0))


    def tile_acquired_fovs(self, channel_ind=0, save_plot=False):
        '''
        Plate-like tiled array of the top two FOVs in each well

        TODO: for now, assumes canonical half-plate layout
        '''

        # hackish way to parse the well_ids
        rows = sorted(list(set([well_id[0] for well_id in self.platemap.imaging_well_id])))
        cols = sorted(list(set([int(well_id[1:]) for well_id in self.platemap.imaging_well_id])))

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
            
        fig, axs = plt.subplots(len(rows), len(cols), figsize=(len(cols)*2, len(rows)*4))
        for row_ind, row in enumerate(rows):
            for col_ind, col in enumerate(cols):
                ax = axs[row_ind][col_ind]
                well_id = '%s%s' % (row, col)
                
                # the pipeline well_id
                sample_well_id = self.sample_well_id_from_imaging_well_id(well_id)

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
                ax.set_title('%s (N = %s) %s' % (well_id, d.shape[0], sample_well_id))

        plt.subplots_adjust(left=.01, right=.99, top=.95, bottom=.01, wspace=0.01)
        plt.savefig(os.path.join(self.qc_dir, 'Tiled-FOVs-TOP2-C%d.pdf' % channel_ind))


    def sample_well_id_from_imaging_well_id(self, imaging_well_id):

        if imaging_well_id not in self.platemap.imaging_well_id.values:
            print('Warning: imaging_well_id %s not found in the platemap' % imaging_well_id)
            return 'unknown'

        row = self.platemap.loc[self.platemap.imaging_well_id==imaging_well_id].iloc[0]
        return row.pipeline_well_id


    def rename_raw_tiffs(self, preview=True):
        '''
        Rename the acquired stacks to include the plate_id and the sample well_id
        '''

        # all of the raw TIFFs
        src_filepaths = sorted(glob.glob(os.path.join(self.raw_data_dir, '*.ome.tif')))

        dst_filenames = []
        src_filenames = [filepath.split(os.sep)[-1] for filepath in src_filepaths]
        for src_filename in src_filenames:

            # parse the raw TIFF filename
            imaging_well_id, site_num = self.parse_raw_tiff_filename(src_filename)
            
            # the platemap row corresponding to this imaging well_id
            row = self.platemap.loc[self.platemap.imaging_well_id==imaging_well_id].iloc[0]
            if not row.shape[0]:
                dst_filename = None
                print('Warning: no platemap row for imaging_well_id %s' % imaging_well_id)

            else:
                well_id = self.pad_well_id(row.pipeline_well_id)
                site_id = 'S%02d' % site_num
                dst_filename = \
                    f'{row.parental_line}-{row.plate_id}-{well_id}-{row.pml_id}-{site_id}__{src_filename}'
            dst_filenames.append(dst_filename)

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