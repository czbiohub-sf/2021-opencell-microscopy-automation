
import os
import re
import git
import json
import time
import py4j
import shutil
import datetime
import numpy as np
import pandas as pd

from dragonfly_automation import utils
from dragonfly_automation import operations
from dragonfly_automation.gateway import gateway_utils
from dragonfly_automation.settings_schemas import ChannelSettingsManager
from dragonfly_automation.acquisitions import pipeline_plate_settings as settings


class Acquisition:
    '''
    Base class for acquisition scripts

    Public methods
    --------------
    event_logger : 


    Public attributes
    -----------------
    env : 
    datastore : 
    verbose : 
    operations : 
    gate, mm_studio, mm_core : the py4j objects exposed by mm2python

    '''

    def __init__(self, root_dir, env='dev', verbose=True, test_mode=None):
        '''

        datastore : py4j datastore object
            passed explicitly here to avoid an unusual py4j error
        root_dir : the imaging experiment directory 
            (usually ends with a directory of the form 'ML0000_20190823')
        env : 'prod' or 'dev'
        verbose : whether to print log messages

        '''

        # strip trailing slashes
        root_dir = re.sub(f'{os.sep}+$', '', root_dir)
        self.root_dir = root_dir

        self.env = env
        self.verbose = verbose

        # the name of the experiment is the name of the directory
        self.experiment_name = os.path.split(self.root_dir)[-1]

        # subdirectory for logfiles
        self.log_dir = os.path.join(self.root_dir, 'logs')

        # subdirectory for raw data (where the datastore will be created)
        self.data_dir = os.path.join(self.root_dir, 'raw_data')

        # check whether data and/or logs already exist for the root_dir
        if os.path.isdir(self.log_dir):
            if env == 'prod':
                raise ValueError('The experiment directory %s is not empty' % self.root_dir)
            if env == 'dev':
                print('WARNING: Removing existing experiment directory')
                shutil.rmtree(self.root_dir)
        
        os.makedirs(self.log_dir, exist_ok=True)

        # event logs (plaintext)
        self.all_events_log_file = os.path.join(self.log_dir, 'all-events.log')
        self.error_events_log_file = os.path.join(self.log_dir, 'error-events.log')
        self.important_events_log_file = os.path.join(self.log_dir, 'important-events.log')
        
        # acquisition metadata log (JSON)
        self.metadata_log_file = os.path.join(self.log_dir, 'experiment-metadata.json')
        
        # acquisition log (CSV)
        self.acquisition_log_file = os.path.join(self.log_dir, 'acquired-images.csv')

        # AFC log (CSV)
        self.afc_log_file = os.path.join(self.log_dir, 'afc-calls.csv')

        # log the current commit
        try:
            repo = git.Repo('..')
            current_commit = repo.commit().hexsha
            self.acquisition_metadata_logger('git_commit', current_commit)
        except Exception:
            print('Warning: no git repo found and git commit hash will not be logged')

        # log the experiment name and root directory
        self.acquisition_metadata_logger('root_directory', self.root_dir)
        self.acquisition_metadata_logger('experiment_name', self.experiment_name)

        # create the wrapped py4j objects (with logging enabled)
        self.gate, self.mm_studio, self.mm_core = gateway_utils.get_gate(
            env=self.env, 
            wrap=True, 
            logger=self.event_logger,
            test_mode=test_mode
        )

        # create the operations instance (with logging enabled)
        self.operations = operations.Operations(self.event_logger)


    def event_logger(self, message, newline=False):
        '''
        Append a message to the event log

        Note that this method is also passed to, and called from, 
        - some methods in the operations module
        - the wrappers around the gate, mm_studio, and mm_core objects
        - the logging method of the FOVScorer instance at self.fov_scorer

        For now, we rely on the correct manual hard-coding of log messages 
        to identify, in the logfile, which of these contexts this method was called from

        '''

        log_filepaths = [self.all_events_log_file]

        # prepend a timestamp to the message
        message = '%s %s' % (utils.timestamp(), message)
        if newline:
            message = '\n%s' % message
 
        # manually-defined 'important' event categories
        important_labels = [
            'ACQUISITION', 'SCORING', 'AUTOFOCUS', 'AUTOEXPOSURE', 'ERROR', 'WARNING',
        ]

        message_is_important = False
        for label in important_labels:
            if label in message:
                message_is_important = True
    
        if message_is_important:
            log_filepaths.append(self.important_events_log_file)
        
        message_is_error = 'ERROR' in message
        if message_is_error:
            log_filepaths.append(self.error_events_log_file)

        # write the message to the appropriate logs
        for filepath in log_filepaths:
            with open(filepath, 'a') as file:
                file.write('%s\n' % message)
        
        # finally, print the message
        if self.verbose and message_is_important:
            print(message)


    def acquisition_metadata_logger(self, key, value):
        '''
        Append a key-value pair to the acquisition-level metadata (which is just a JSON object)

        This log is intended to capture metadata like the name of the acquisition subclass,
        the name of the imaging experiment, the start and end times of the acquisition itself,
        the git commit hash of the dragonfly-automation repo when the acquisition was run,
        and also all of the acquisition-level settings 
        (autoexposure, stack, and default channel settings)

        TODO: check for key collisions
        '''

        if os.path.isfile(self.metadata_log_file):
            with open(self.metadata_log_file, 'r') as file:
                metadata = json.load(file)
        else:
            metadata = {}
        
        metadata[key] = value
        with open(self.metadata_log_file, 'w') as file:
            json.dump(metadata, file)


    def afc_logger(self, **kwargs):
        '''
        Append a row to the AFC log
        
        This log is intended to record the position of the FocusDrive
        before and after AFC is called, as well as whether any errors occur.
    
        '''

        # construct the row
        row = {'timestamp': utils.timestamp()}
        row.update(kwargs)

        # append the row to the CSV
        if os.path.isfile(self.afc_log_file):
            d = pd.read_csv(self.afc_log_file)
            d = d.append(row, ignore_index=True)
        else:
            d = pd.DataFrame([row])
        d.to_csv(self.afc_log_file, index=False)


    def acquisition_logger(self, channel_settings, **kwargs):
        '''
        Append a row to the acquisitions log
        
        This log is intended to contain the channel settings (laser power, exposure time, etc)
        and position identifiers (index, label, etc) associated with each acquired image/stack
        
        For now, kwargs are appended to the row without any validation, 
        and are intended to be used for position identifiers 
        (e.g., well_id, site_num, position_ind)

        *** 
        Note that this method must be called _manually_ 
        after each call to operations.acquire_stack 
        ***
    
        '''

        # construct the row
        row = {'timestamp': utils.timestamp()}
        row.update(channel_settings.__dict__)
        row.update(kwargs)

        # append the row to the CSV
        if os.path.isfile(self.acquisition_log_file):
            d = pd.read_csv(self.acquisition_log_file)
            d = d.append(row, ignore_index=True)
        else:
            d = pd.DataFrame([row])
        d.to_csv(self.acquisition_log_file, index=False)


    def _initialize_datastore(self):
        '''
        Initialize a datastore object
        '''

        # the datastore can only be initialized if the data directory does not exist
        if os.path.isdir(self.data_dir):
            raise ValueError('Data directory already exists at %s' % self.data_dir)

        # these arguments for createMultipageTIFFDatastore are copied from Nathan's script
        should_split_positions = True
        should_generate_separate_metadata = True

        self.event_logger('ACQUISITION INFO: Creating datastore at %s' % self.data_dir)
        self.datastore = self.mm_studio.data().createMultipageTIFFDatastore(
            self.data_dir,
            should_generate_separate_metadata, 
            should_split_positions
        )

        self.mm_studio.displays().createDisplay(self.datastore)
        
    
    def setup(self):
        '''
        Commands to execute before the acquisition begins
        e.g., setting the autofocus mode, camera mode, various synchronization commands
        '''
        self.acquisition_metadata_logger('setup_timestamp', utils.timestamp())

        # create the datastore
        self._initialize_datastore()


    def run(self):
        '''
        The main acquisition workflow
        '''
        raise NotImplementedError


    def cleanup(self):
        '''
        Commands that should be executed after the acquisition is complete
        (that is, after self.run)

        TODO: are there close/shutdown methods that should be called on the py4j objects?

        '''
        # freeze the datastore
        if self.datastore:
            self.datastore.freeze()

        # log the time
        self.acquisition_metadata_logger('cleanup_timestamp', utils.timestamp())

    
class PipelinePlateAcquisition(Acquisition):
    '''
    This is a re-implementation of Nathan's pipeline plate acquisition script

    It acquires hoechst and GFP z-stacks at some number of positions
    in some number of wells on a 96-well plate.

    See the comments in self.run for more details

    '''
    
    def __init__(
        self, 
        data_dir, 
        fov_scorer, 
        pml_id,
        plate_id,
        platemap_type, 
        env='dev', 
        verbose=True, 
        test_mode=None, 
        attempt_count=1,
        acquire_bf_stacks=True, 
        skip_fov_scoring=False
    ):

        # create the external metadata
        self.external_metadata = {'pml_id': pml_id, 'platemap_type': platemap_type}

        # if the platemap is canonical half-plate imaging, 
        # we hard-code the parental_line, electroporation_id and round_id
        # (note that the round_id of 'R02' corresponds to imaging a thawed plate for the first time)
        if platemap_type != 'custom':
            self.external_metadata['parental_line'] = 'czML0383'
            self.external_metadata['imaging_round_id'] = 'R02'
            self.external_metadata['plate_id'] = plate_id
        
        # construct the root directory for this acquisition
        root_dir = os.path.join(data_dir, '%s-%s' % (pml_id, attempt_count))
        super().__init__(root_dir, env=env, verbose=verbose, test_mode=test_mode)

        # save the external metadata
        with open(os.path.join(self.root_dir, 'metadata.json'), 'w') as file:
            json.dump(self.external_metadata, file)

        # whether to acquire a brightfield stack after the hoechst and GFP stacks
        self.acquire_bf_stacks = acquire_bf_stacks

        # whether to skip FOV scoring (only for manual redos)
        self.skip_fov_scoring = skip_fov_scoring

        # create the log_dir for the fov_scorer instance
        self.fov_scorer = fov_scorer
        self.fov_scorer.log_dir = os.path.join(self.log_dir, 'fov-scoring')

        # log the directory the fov_scorer was loaded from
        self.acquisition_metadata_logger('fov_scorer_save_dir', self.fov_scorer.save_dir)

        # log the name of the acquisition subclass
        self.acquisition_metadata_logger('acquisition_name', self.__class__.__name__)

        # log whether BF stacks will be acquired
        self.acquisition_metadata_logger('brightfield_stacks_acquired', self.acquire_bf_stacks)
        self.acquisition_metadata_logger('fov_scoring_skipped', self.skip_fov_scoring)

        # initialize channel managers
        self.bf_channel = ChannelSettingsManager(settings.bf_channel_settings)
        self.gfp_channel = ChannelSettingsManager(settings.gfp_channel_settings)
        self.hoechst_channel = ChannelSettingsManager(settings.hoechst_channel_settings)
        
        # FOV selection settings
        self.fov_selection_settings = settings.fov_selection_settings

        # copy the autoexposure settings
        self.autoexposure_settings = settings.autoexposure_settings

        # brightfield stack settings
        self.brightfield_stack_settings = settings.bf_stack_settings
        
        # fluorescence stack settings for dev and prod
        # ('dev' settings reduce the number of slices acquired in dev mode)
        if self.env == 'prod':
            self.flourescence_stack_settings = settings.prod_fl_stack_settings
        if self.env == 'dev':
            self.flourescence_stack_settings = settings.dev_fl_stack_settings
    
        # stage labels for convenience
        self.xystage_label = 'XYStage'
        self.zstage_label = self.flourescence_stack_settings.stage_label

        # log all of the settings
        settings_names = [
            'fov_selection_settings', 
            'autoexposure_settings', 
            'flourescence_stack_settings',
            'brightfield_stack_settings'
        ]
        for settings_name in settings_names:
            self.acquisition_metadata_logger(
                settings_name, dict(getattr(self, settings_name)._asdict())
            )

        # log the channel settings
        for channel_name in ['hoechst_channel', 'gfp_channel', 'bf_channel']:
            self.acquisition_metadata_logger(
                channel_name, getattr(self, channel_name).__dict__
            )


    def setup(self):
        self.event_logger('ACQUISITION INFO: Calling setup method')

        super().setup()

        # change the autofocus mode to AFC
        af_manager = self.mm_studio.getAutofocusManager()
        af_manager.setAutofocusMethodByName("Adaptive Focus Control")

        # these `assignImageSynchro` calls are copied directly from Nathan's script
        self.mm_core.assignImageSynchro(self.zstage_label)
        self.mm_core.assignImageSynchro(self.xystage_label)
        self.mm_core.assignImageSynchro(self.mm_core.getShutterDevice())
        self.mm_core.assignImageSynchro(self.mm_core.getCameraDevice())

        # turn on auto shutter mode 
        # (this means that the shutter automatically opens and closes when an image is acquired)
        self.mm_core.setAutoShutter(True)
        self.event_logger('ACQUISITION INFO: Exiting setup method')


    def cleanup(self):
        '''
        TODO: are there commands that should be executed here
        to ensure the microscope is returned to a 'safe' state?
        '''
        self.event_logger('ACQUISITION INFO: Calling cleanup method')
        super().cleanup()
        self.event_logger('ACQUISITION INFO: Exiting cleanup method')


    def parse_hcs_position_label(self, label):
        '''
        Parse a position label generated by MicroManager's HCS Site Generator plugin
        and extract the plate well_id and the site number

        These labels are of the form `{well_id}_Site-{site_num}`

        Examples: 'A1-Site_0', 'A1-Site_24', 'G10-Site_0'
        '''

        pattern = r'^([A-H][0-9]{1,2})-Site_([0-9]+)$'
        result = re.findall(pattern, label)
        if not result:
            self.event_logger('ACQUISITION ERROR: Unexpected site label %s' % label)
        
        well_id, site_num = result[0]
        site_num = int(site_num)
        return well_id, site_num


    def run(self, mode='prod'):
        '''
        The main acquisition workflow

        Overview
        --------

        Parameters
        ----------
        mode : one of 'test' or 'prod'
            in test mode, only the first well is visted, and only one z-stack is acquired
            (at a position that is *not* among those selected by self.select_positions)

        Assumptions
        -----------
        The positions returned by mm_studio.getPositionList() must have been generated
        by the HCS Site Generator plugin. 

        In particular, they must have labels of the form `{well_id}-Site_{site_num}`.
        For example: 'A1_Site-0', 'A1_Site-1', 'A10_Site-10', etc.

        Here, the well_id identifies the well on the *imaging* plate, 
        and the site number is a per-well count, starting from zero, of positions in each well.

        '''

        # construct properties for each position
        # NOTE that the order of the positions is determined by the HCS Site Generator
        # and must be preserved to prevent dangerously long x-y stage movements
        all_plate_positions = []
        mm_position_list = self.mm_studio.getPositionList()
        for position_ind in range(mm_position_list.getNumberOfPositions()):
            
            mm_position = mm_position_list.getPosition(position_ind)
            position_label = mm_position.getLabel()
            well_id, site_num = self.parse_hcs_position_label(position_label)

            # construct a human-readable and unique name for the current position
            # (used in acquire_stack to determine the name of the TIFF file)
            position_name = f'{position_ind}-{well_id}-{site_num}'

            all_plate_positions.append({
                'ind': position_ind,
                'label': position_label,
                'name': position_name,
                'well_id': well_id,
                'site_num': site_num,
            })

        # list of *order-preserved* unique well_ids 
        # (assumes that all positions in each well appear together in a single contiguous block)
        unique_well_ids = [all_plate_positions[0]['well_id']]
        for position in all_plate_positions:
            well_id = position['well_id']
            if well_id != unique_well_ids[-1]:
                unique_well_ids.append(well_id)

        # only visit the first well in test mode
        if mode == 'test':
            unique_well_ids = unique_well_ids[:1]
            self.event_logger(
                'ACQUISITION WARNING: Acquisition is running in test mode, '
                'so only the first well will be visited',
                newline=True
            )

        for well_id in unique_well_ids:
            self.current_well_id = well_id
            
            # positions in this well
            all_well_positions = [p for p in all_plate_positions if p['well_id'] == well_id]

            # score and rank the positions
            if self.skip_fov_scoring:
                self.event_logger(
                    'ACQUISITION INFO: Skipping FOV scoring and acquiring all FOVs in well %s'
                    % well_id, 
                    newline=True
                )
                positions_to_acquire = all_well_positions
            else:
                self.event_logger(
                    'ACQUISITION INFO: Scoring all FOVs in well %s' % well_id, newline=True)
                positions_to_acquire = self.select_positions(all_well_positions)
    
            if not len(positions_to_acquire):
                self.event_logger('ACQUISITION WARNING: No FOVs will be imaged in well %s' % well_id)
                continue

            # prettify the scores for the event log
            scores = []
            if not self.skip_fov_scoring:
                scores = [
                    '%0.2f' % p['fov_score'] if p.get('fov_score') is not None else 'None' 
                    for p in positions_to_acquire
                ]

            self.event_logger(
                'ACQUISITION INFO: Imaging %d FOVs in well %s (scores: [%s])'
                 % (len(positions_to_acquire), well_id, ', '.join(scores)),
                newline=True
            )

            # in test mode, acquire a single unselected FOV
            if mode == 'test':
                self.event_logger(
                    'ACQUISITION WARNING: running in test mode, so the selected positions '
                    'will be replaced with one unselected position',
                    newline=True
                )
                names = [position['name'] for position in positions_to_acquire]
                for position in all_well_positions:
                    if position['name'] not in names:
                        positions_to_acquire = [position]
                        break
            
            self.acquire_positions(positions_to_acquire)
        self.cleanup()
    

    def select_positions(self, positions):
        '''
        Visit and score each of the positions listed in `positions` 
        and select the highest-scoring subset of positions to acquire

        The positions should correspond to all positions in one well,
        but we do not explicitly assume that that is the case here. 

        *** Note about AFC ***
        Because AFC can take some time to run, we use a little trick/hack here to speed it up.

        First, before the loop over positions, we call AFC at the first position
        (in site-number order) and record the FocusDrive position (which is what AFC updates).
        Then, in the position loop, we move the FocusDrive to this AFC-adjusted position,
        right after moving to the new position and right *before* calling AFC.

        The logic of this little trick is that, because the positions we visit are all in one well, 
        the AFC-adjusted FocusDrive positions are likely to be very similiar to one another 
        (because the plate should not be tilted on the length scale of a single well).

        '''

        # the minimum number of positions to image in a well
        min_num_positions = self.fov_selection_settings.min_num_positions

        # the maximum number of positions to image in a well
        max_num_positions = self.fov_selection_settings.max_num_positions

        # the returned list of 'good' positions to be imaged
        positions_to_image = []

        # sort positions by site number
        positions = sorted(positions, key=lambda position: position['site_num'])

        # go to the first position
        position_ind = positions[0]['ind']
        self.operations.go_to_position(self.mm_studio, self.mm_core, position_ind)

        # call AFC
        afc_did_succeed = self.operations.call_afc(
            self.mm_studio, self.mm_core, self.event_logger, self.afc_logger, position_ind
        )

        # if AFC succeeded, get the AFC-updated FocusDrive position
        afc_updated_focusdrive_position = None
        if afc_did_succeed:
            afc_updated_focusdrive_position = self.mm_core.getPosition('FocusDrive')

        # change to the hoechst channel for FOV scoring
        self.operations.change_channel(self.mm_core, self.hoechst_channel)

        # score the FOV at each position
        for position in positions:
            position_ind = position['ind']

            # catch timeout errors when the position is too close
            # to the lower bound of the stage range
            try:
                self.operations.go_to_position(self.mm_studio, self.mm_core, position_ind)
            except py4j.protocol.Py4JJavaError:
                self.event_logger(
                    'SCORING ERROR: The XYStage timed out at position %s' % position['name']
                )
                continue

            # update the FocusDrive position (this should help AFC to focus faster)
            if afc_updated_focusdrive_position is not None:
                self.operations.move_z_stage(
                    self.mm_core, 
                    'FocusDrive', 
                    position=afc_updated_focusdrive_position, 
                    kind='absolute'
                )

            # attempt to call AFC
            afc_did_succeed = self.operations.call_afc(
                self.mm_studio, self.mm_core, self.event_logger, self.afc_logger, position_ind
            )

            # update the AFC-updated FocusDrive position and log it for later use
            # in self.acquire_positions
            if afc_did_succeed:
                afc_updated_focusdrive_position = self.mm_core.getPosition('FocusDrive')
                position['afc_updated_focusdrive_position'] = afc_updated_focusdrive_position

            # acquire an image of the hoechst signal
            image = self.operations.acquire_image(
                self.gate, self.mm_studio, self.mm_core, self.event_logger
            )

            # score the FOV
            # note that, given all of the error handling in PipelineFOVScorer, 
            # this try-catch is a last line of defense that should never be needed
            log_info = None
            try:
                log_info = self.fov_scorer.score_raw_fov(image, position=position)
            except Exception as error:
                self.event_logger(
                    "SCORING ERROR: an uncaught exception occurred during FOV scoring at positions '%s': %s"
                    % (position['name'], error)
                )

            # retrieve the score and note it in the event log
            if log_info is not None:
                score = log_info.get('score')
                position['fov_score'] = score

                # prettify the score for the event log
                score = '%0.2f' % score if score is not None else score
                self.event_logger(
                    "SCORING INFO: The FOV score at position '%s' was %s (comment: '%s')"
                    % (position['name'], score, log_info.get('comment'))
                )

        # drop positions without a score
        # (this will happen if log_info.get('score') is None or if there was an uncaught error above)
        positions_with_score = [p for p in positions if p.get('fov_score') is not None]

        # sort positions in descending order by score (from good to bad)
        positions_with_score = sorted(positions_with_score, key=lambda p: -p['fov_score'])

        # list of acceptable positions
        acceptable_positions = [
            p for p in positions_with_score if p['fov_score'] > self.fov_selection_settings.min_score
        ]
        self.event_logger('ACQUISITION INFO: Found %d acceptable FOVs' % len(acceptable_positions))

        # crop the list if there are more acceptable positions than needed
        positions_to_image = acceptable_positions[:max_num_positions]

        # remove positions from positions_with_score that are now in positions_to_image
        positions_with_score = positions_with_score[len(positions_to_image):]

        # if there were too few acceptable positions found, 
        # append the next-highest-scoring positions
        num_positions_short = min_num_positions - len(positions_to_image)
        if num_positions_short > 0:
            additional_positions = positions_with_score[:num_positions_short]
            positions_to_image.extend(additional_positions)

            if len(additional_positions):
                self.event_logger(
                    'ACQUISITION INFO: Fewer than the minimum of %s acceptable FOVs were found '
                    'so %s additional scored FOVs will be imaged'
                    % (min_num_positions, len(additional_positions))
                )
            else:
                self.event_logger(
                    'ACQUISITION INFO: Fewer than the minimum of %s acceptable FOVs were found '
                    'and there are no additional scored FOVs to image'
                    % (min_num_positions,)
                )

        # if there are still too few FOVs, there's nothing we can do        
        if len(positions_to_image) > 0 and len(positions_to_image) < min_num_positions:
            self.event_logger(
                'ACQUISITION WARNING: All %s scored FOVs will be imaged '
                'but this is fewer than the required minimum of %s FOVs'
                % (len(positions_to_image), min_num_positions)
            )

        return positions_to_image


    def go_to_position(self, position):
        '''
        Convenience method used in `acquire_positions`
        to move to a new position and update the FocusDrive position
        '''
        self.operations.go_to_position(self.mm_studio, self.mm_core, position['ind'])

        # update the FocusDrive if there is an AFC-updated FocusDrive position 
        # associated with this position (these are created in `select_positions`)
        afc_updated_focusdrive_position = position.get('afc_updated_focusdrive_position')
        if afc_updated_focusdrive_position is not None:
            current_position = self.mm_core.getPosition('FocusDrive')
            self.event_logger(
                'ACQUISITION INFO: Updating the interpolated FocusDrive position (%s) '
                'with the AFC-updated position (%s)'
                % (current_position, afc_updated_focusdrive_position)
            )
            self.operations.move_z_stage(
                self.mm_core, 
                'FocusDrive', 
                position=afc_updated_focusdrive_position, 
                kind='absolute'
            )

            # delay to help AFC 'adjust' to the new position
            time.sleep(1.0)


    def acquire_positions(self, positions):
        '''
        Acquire z-stacks at the positions listed in positions

        ** We assume that all of these positions correspond to one well **

        '''

        # this should never happen
        if not len(positions):
            print('Warning: acquire_positions received an empty list of positions')
            return

        # sort positions by site number (to minimize stage movement)
        positions = sorted(positions, key=lambda position: position['site_num'])

        self.event_logger(
            "ACQUISITION INFO: Autoexposing at the first acceptable FOV of well %s (position '%s')"
            % (self.current_well_id, positions[0]['name'])
        )
    
        # go to the first position, where autoexposure will be run
        position = positions[0]
        self.go_to_position(position)

        # attempt to call AFC
        afc_did_succeed = self.operations.call_afc(
            self.mm_studio, self.mm_core, self.event_logger, self.afc_logger, position['ind']
        )

        # for now, ignore AFC errors
        # TODO: think about how to handle this situation, which is very serious,
        # since autoexposure won't work if we are out of focus
        # (perhaps try moving to a different position?)
        if not afc_did_succeed:
            pass

        # reset the GFP channel settings
        self.gfp_channel.reset()

        # change the channel to GFP
        self.operations.change_channel(self.mm_core, self.gfp_channel)

        # run the autoexposure algorithm at the first position 
        # (note that laser power and exposure time are modified in-place)
        autoexposure_did_succeed = self.operations.autoexposure(
            gate=self.gate,
            mm_studio=self.mm_studio,
            mm_core=self.mm_core,
            stack_settings=self.flourescence_stack_settings,
            autoexposure_settings=self.autoexposure_settings,
            channel_settings=self.gfp_channel,
            event_logger=self.event_logger
        )

        if not autoexposure_did_succeed:
            self.event_logger(
                "ACQUISITION ERROR: autoexposure failed in well %s but attempting to continue" 
                % self.current_well_id
            )

        # acquire z-stacks at each position
        for ind, position in enumerate(positions):
            self.event_logger(
                "ACQUISITION INFO: Acquiring stacks at position %d of %d in well %s (position '%s')" 
                % (ind + 1, len(positions), position['well_id'], position['name']),
                newline=True
            )

            # sanity check 
            if self.current_well_id != position['well_id']:
                self.event_logger(
                    'ACQUISITION ERROR: The well_id of the position being acquired '
                    'does not match self.current_well_id'
                )

            # go to the position
            position_ind = position['ind']
            self.go_to_position(position)

            # attempt to call AFC
            afc_did_succeed = self.operations.call_afc(
                self.mm_studio, self.mm_core, self.event_logger, self.afc_logger, position_ind
            )

            # for now, ignore AFC errors
            # TODO: consider skipping the position if AFC has failed
            if not afc_did_succeed:
                pass
            
            # settings for the two fluorescence channels
            all_settings = [
                {
                    'channel': self.hoechst_channel,
                    'stack': self.flourescence_stack_settings,
                },{
                    'channel': self.gfp_channel,
                    'stack': self.flourescence_stack_settings,
                }
            ]

            # settings for the brightfield channel (which has its own z-stack settings)
            if self.acquire_bf_stacks:
                all_settings.append({
                    'channel': self.bf_channel,
                    'stack': self.brightfield_stack_settings,
                })

            # acquire a z-stack for each channel
            for channel_ind, settings in enumerate(all_settings):
                self.event_logger(
                    "ACQUISITION INFO: Acquiring channel '%s'" % settings['channel'].config_name
                )

                # change the channel
                self.operations.change_channel(self.mm_core, settings['channel'])

                # acquire the stack
                self.operations.acquire_stack(
                    mm_studio=self.mm_studio,
                    mm_core=self.mm_core, 
                    datastore=self.datastore, 
                    stack_settings=settings['stack'],
                    channel_ind=channel_ind,
                    position_ind=position_ind,
                    position_name=position['name']
                )

                # log the acquisition
                self.acquisition_logger(
                    channel_settings=settings['channel'],
                    position_ind=position_ind,
                    well_id=position['well_id'],
                    site_num=position['site_num'],
                    afc_did_succeed=afc_did_succeed
                )
        

