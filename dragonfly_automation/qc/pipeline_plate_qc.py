
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
        if not os.path.isdir(root_dir):
            raise ValueError('The directory %s does not exist' % root_dir)

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
        
        # for acquisitions in which FOVs were selected manually,
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

        # load the platemap 
        # (note that if platemap_type is 'none', an identity platemap is created
        # that includes all well_ids that appears in self.aq_log.well_id)
        self.load_platemap()

        # load the manual flags, if any
        manual_flags_filepath = os.path.join(self.root_dir, 'manual-flags.json')
        self.manual_flags = self.load_manual_flags(manual_flags_filepath)
    
        # the number of channels acquired
        # (this is a bit hackish, but the acquire_bf_stacks flag was not always logged)
        self.num_channels = len(self.aq_log.config_name.unique())

        # create a QC directory to which to save figures
        self.qc_dir = os.path.join(self.root_dir, 'QC')
        os.makedirs(self.qc_dir, exist_ok=True)


    def load_manual_flags(self, filepath):
        '''
        Manual flags are optional and appear in a user-defined JSON object
        that lists imaging plate row_ids and well_ids from which all FOVs should be discarded
        (and, crucially, not inserted into the database).

        The purpose of these flags is to identify FOVs that are 'bad' or unacceptable
        for a reason that is difficult or impossible to identify computationally. 

        The schema of the JSON object is
        {
            "flags": [
                {
                    "rows": ["B", "C"],
                    "reason": "Free-form user-defined explanation"
                },{
                    "wells": ["A1", "G9"],
                    "reason": "Free-form user-defined explanation"
                },
            ]
        }
        '''
        manual_flags = {'well_ids': [], 'row_ids': []}
        if not os.path.isfile(filepath):
            return manual_flags

        with open(filepath, 'r') as file:
            d = json.load(file)
        flags = d.get('flags') or []

        [manual_flags['row_ids'].extend(flag.get('rows') or []) for flag in flags]
        [manual_flags['well_ids'].extend(flag.get('wells') or []) for flag in flags]    

        print('Warning: the following manual flags were found: %s' % manual_flags)
        return manual_flags


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


    def validate_external_metadata(self, external_metadata):
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
        if md['platemap_type'] not in ['first-half', 'second-half', 'custom', 'none']:
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
        Custom platemaps are required for any acquisition
        in which an arbitrary subset of wells on the imaging plate were imaged, 
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
        required_columns = (
            set(EXTERNAL_METADATA_SCHEMA['properties'].keys())
            .difference(['pml_id', 'platemap_type'])
        )

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

        platemap_type = self.external_metadata['platemap_type']
        if platemap_type == 'first-half':
            platemap = pd.DataFrame(data=half_plate_layout.first_half)
        elif platemap_type == 'second-half':
            platemap = pd.DataFrame(data=half_plate_layout.second_half)
        elif platemap_type == 'custom':
            platemap = self.load_and_validate_custom_platemap()
        elif platemap_type == 'none':
            print('Warning: platemap_type is None')
            print('An identity platemap will be constructed for all well_ids in the acquisition log')
            well_ids = self.aq_log.well_id.unique()
            plate_layout = [{
                'imaging_well_id': well_id, 
                'pipeline_well_id': well_id
            } for well_id in well_ids]
            platemap = pd.DataFrame(data=plate_layout)
        else:
            raise ValueError("Invalid platemap_type '%s'" % platemap_type)

        # append the remaining global metadata attributes
        # (for first-half or second-half, this includes the parental_line and the plate_id)
        for key, value in self.external_metadata.items():
            if key not in platemap.columns:
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

        # append the manual flags (usually empty)
        summary.update({
            'manually_flagged_rows': self.manual_flags['row_ids'],
            'manually_flagged_wells': self.manual_flags['well_ids']})

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
        src_filename = src_filepath.split(os.sep)[-1]

        tiff = images.RawPipelineTIFF(src_filepath, verbose=False)
        tiff.parse_micromanager_metadata()
        tiff.validate_micromanager_metadata()
        tiff.split_channels()

        if tiff.did_split_channels:
            for ind, channel in enumerate([tiff.laser_405, tiff.laser_488]):
                dst_filename = src_filename.replace('.ome.tif', '_PROJ-CH%d.tif' % ind)
                dst_filepath = os.path.join(dst_dirpath, dst_filename)
                tiff.project_stack(channel_name=channel, axis='z', dst_filepath=dst_filepath)
        else:
            print("Warning: could not split channels of raw TIFF '%s'" % src_filename)

        tiff.tiff.close()


    def tile_acquired_fovs(self, channel_ind=0, save_plot=False):
        '''
        Plate-like tiled array of the top two FOVs in each well

        TODO: for now, assumes canonical half-plate layout
        '''

        # hackish way to parse the well_ids
        rows = sorted(list(set([well_id[0] for well_id in self.platemap.imaging_well_id])))
        cols = sorted(list(set([int(well_id[1:]) for well_id in self.platemap.imaging_well_id])))

        downsample_by = 2
        im_sz = int(1024/downsample_by)
        blank_fov = np.zeros((im_sz, im_sz), dtype='uint8')

        # hard-coded borders for the 2x2 tile of FOVs from each well
        border_width = 10
        vertical_border = (255*np.ones((im_sz, border_width))).astype('uint8')
        horizontal_border = (255*np.ones((border_width, 2*im_sz + border_width))).astype('uint8')
        
        # acquired z-stacks (channel is arbitrary, since we will only need the well_ids and site_nums)
        gfp_aq_log = self.aq_log.loc[self.aq_log.config_name == 'EMCCD_Confocal40_GFP'].copy()

        # merge the FOV scores from the score_log
        if self.has_score_log:
            gfp_aq_log = pd.merge(
                gfp_aq_log,
                self.score_log[['position_ind', 'score']], 
                left_on='position_ind',
                right_on='position_ind',
                how='left')
        else:
            gfp_aq_log['score'] = 0
    
        # note that figsize is (width, height)
        fig, axs = plt.subplots(len(rows), len(cols), figsize=(len(cols)*4, len(rows)*4))

        if len(rows) == 1 and len(cols) == 1:
            axs = [[axs]]
        elif len(rows) == 1:
            axs = axs[None, :]
        elif len(cols) == 1:
            axs = axs[:, None]

        for row_ind, row_label in enumerate(rows):
            for col_ind, col_label in enumerate(cols):
                ax = axs[row_ind][col_ind]

                # construct the imaging plate well_id
                imaging_well_id = '%s%s' % (row_label, col_label)
                
                # the acquired FOVs for this well, sorted by score
                well_aq_log = gfp_aq_log.loc[gfp_aq_log.well_id == imaging_well_id].copy()
                well_aq_log = well_aq_log.sort_values(by='score', ascending=False).reset_index()

                # load the z-projetions of the four top-scoring FOVs
                ims = 4*[blank_fov]
                for ind, row in well_aq_log.iloc[:4].iterrows():

                    # HACK: manually construct filenames to match the dst_filenames
                    # generated in self.generate_z_projections
                    filenames = [
                        f'MMStack_{row.position_ind}-{imaging_well_id}-{row.site_num}_C{channel_ind}-PROJ-Z.tif',
                        f'MMStack_{row.position_ind}-{imaging_well_id}-{row.site_num}_PROJ-CH{channel_ind}.tif',
                    ]
                    im = None
                    for filename in filenames:
                        filepath = os.path.join(self.qc_dir, 'z-projections', filename)
                        if not os.path.isfile(filepath):
                            continue
                        im = tifffile.imread(filepath)[::downsample_by, ::downsample_by]
                        ims[ind] = self.autogain(im, p=1)

                    if im is None:
                        print('Warning: no z-projection found for %s' % filenames[0])

                # concat the images into a 2x2 tiled array    
                tile = np.concatenate(
                    (
                        np.concatenate((ims[0], vertical_border, ims[1]), axis=1),
                        horizontal_border,
                        np.concatenate((ims[2], vertical_border, ims[3]), axis=1),
                    ), 
                    axis=0)

                # plot the tiled images
                ax.imshow(tile, cmap='gray')
                ax.set_xticks([])
                ax.set_yticks([])

                parenthetical_info = 'No FOVs'
                if  well_aq_log.shape[0]:
                    gfp_laser_power = well_aq_log.iloc[0].laser_power
                    gfp_laser_power = '%d' % gfp_laser_power if gfp_laser_power > 1 else '%0.2f' % gfp_laser_power
                    gfp_exposure_time = '%d' % well_aq_log.iloc[0].exposure_time
                    parenthetical_info = '%sms at %s%%' % (gfp_exposure_time, gfp_laser_power)

                # the pipeline plate_id and well_id
                plate_id, sample_well_id = self.sample_well_id_from_imaging_well_id(imaging_well_id)
                plate_info = 'Not in platemap'
                if plate_id is not None:
                    plate_info = 'P%d-%s' % (int(plate_id[1:]), sample_well_id)

                title = f' {imaging_well_id}  |  {plate_info}  |  {parenthetical_info} '
                ax.set_title(title, fontsize=16)

        plt.subplots_adjust(left=.01, right=.99, top=.95, bottom=.01, wspace=0.01, hspace=0.15)
        plt.savefig(os.path.join(self.qc_dir, '%s-top-scoring-FOVs-CH%d.pdf' % (self.root_dirname, channel_ind)))


    def sample_well_id_from_imaging_well_id(self, imaging_well_id):
        '''
        Get the sample plate_id and well_id for a given imaging well_id
        '''
        plate_id, sample_well_id = None, None
        if imaging_well_id in self.platemap.imaging_well_id.values:        
            row = self.platemap.loc[self.platemap.imaging_well_id == imaging_well_id].iloc[0]
            plate_id = row.plate_id
            sample_well_id = row.pipeline_well_id
        else:
            print('Warning: imaging_well_id %s not found in the platemap' % imaging_well_id)
        
        return plate_id, sample_well_id


    def construct_fov_metadata(self, renamed=False, overwrite=False):
        '''
        Construct the metadata for each raw TIFF file,
        including the site_id and the filename to which the raw TIFF should be renamed

        The new filenames prepend the parental_line, plate_id, well_id, pml_id and site_id
        to the original raw filenames, separated by a double underscore:
        {parental_line-plate_id-well_id-pml_id-site_id}__{raw_filename}

        renamed : whether the raw TIFFs have already been renamed
        overwrite : whether to overwrite the existing cached raw_metadata (if any)
        '''

        # find all of the raw TIFF files
        raw_tiff_pattern = 'MMStack*.ome.tif'
        if renamed:
            raw_tiff_pattern = '*__MMStack*.ome.tif'

        src_filepaths = sorted(glob.glob(os.path.join(self.raw_data_dir, raw_tiff_pattern)))
        src_filenames = [filepath.split(os.sep)[-1] for filepath in src_filepaths]

        # reconstruct the original raw filenames
        if renamed:
            src_filenames = [filename.split('__')[-1] for filename in src_filenames]

        if not src_filenames:
            raise ValueError('Warning: no raw TIFF files found in %s' % self.root_dir)

        fov_metadata = []
        for src_filename in src_filenames:

            # parse the raw TIFF filename
            imaging_well_id, site_num = self.parse_raw_tiff_filename(src_filename)
            
            # the platemap row corresponding to this imaging well_id
            row = self.platemap.loc[self.platemap.imaging_well_id==imaging_well_id].iloc[0]
            if not row.shape[0]:
                dst_filename = None
                print('Warning: there is an FOV but no platemap entry for imaging_well_id %s' % imaging_well_id)

            else:
                site_id = 'S%02d' % site_num
                well_id = self.pad_well_id(row.pipeline_well_id)

                # construct the filename to which to rename the raw TIFF
                dst_filename = (
                    f'{row.parental_line}-{row.plate_id}-{well_id}-{row.pml_id}-{site_id}'
                    f'__{src_filename}'
                )

                fov_metadata_row = dict(row)
                fov_metadata_row.update({
                    'site_num': site_num,
                    'src_filename': src_filename,
                    'dst_filename': dst_filename,
                    'src_dirpath': os.path.join(self.root_dirname, 'raw_data'),
                })
                fov_metadata.append(fov_metadata_row)
        fov_metadata = pd.DataFrame(data=fov_metadata)

        # check for imaging_well_ids in the platemap without any FOVs
        # (these are presumably wells in which, for some reason, no FOVs were acquired)
        missing_wells = set(self.platemap.imaging_well_id).difference(fov_metadata.imaging_well_id)
        if missing_wells:
            print('Warning: no FOVs found for imaging wells %s' % (missing_wells,))

        # flag imaging wells that should *not* be inserted into the opencell database
        fov_metadata['manually_flagged'] = False
        if self.manual_flags is not None:
            for ind, row in fov_metadata.iterrows():
                imaging_row_id = row.imaging_well_id[0]
                if row.imaging_well_id in self.manual_flags['well_ids'] or imaging_row_id in self.manual_flags['row_ids']:
                    fov_metadata.at[ind, 'manually_flagged'] = True

        # pad the well_ids and sort
        for column in ['imaging_well_id', 'pipeline_well_id']:
            fov_metadata[column] = [self.pad_well_id(well_id) for well_id in fov_metadata[column]]
        fov_metadata.sort_values(by=['plate_id', 'pipeline_well_id', 'site_num'], inplace=True)

        filepath = os.path.join(self.root_dir, 'fov-metadata.csv')
        if os.path.isfile(filepath) and not overwrite:
            print('Warning: cached fov metadata already exists and will not be overwritten')
        else:
            fov_metadata.to_csv(filepath, index=False)
        return fov_metadata


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


    def rename_raw_tiffs(self):
        '''
        Rename the raw TIFF files according to the dst_filenames 
        generated in construct_fov_metadata
        '''
        metadata = self.construct_fov_metadata(renamed=False)
        for ind, row in metadata.iterrows():
            print('Renaming %s' % row.src_filename)
            src_filepath = os.path.join(self.raw_data_dir, row.src_filename)
            dst_filepath = os.path.join(self.raw_data_dir, row.dst_filename)
            os.rename(src_filepath, dst_filepath)
        return metadata


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