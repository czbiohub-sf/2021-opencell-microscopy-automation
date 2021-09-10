
import os
import re
import git
import json
import time
import py4j
import datetime
import dataclasses
import numpy as np
import pandas as pd

from dragonfly_automation import utils, microscope_operations
from dragonfly_automation.acquisitions import pipeline_plate_settings as settings


class PipelinePlateAcquisition:
    '''
    This script is the canonical OpenCell microscopy acquisition script
    It dynamically identifies, and acquires z-stacks at,
    a user-specified number of 'good' FOVs in each well of a 96-well plate

    root_dir : the imaging experiment directory 
        (usually ends with a directory of the form 'PML0123')
    mock_micromanager_api : whether to mock the micromanager API
    '''
    def __init__(
        self, 
        root_dir, 
        pml_id,
        plate_id,
        platemap_type, 
        micromanager_interface,
        mock_micromanager_api=False, 
        acquire_brightfield_stacks=True, 
        skip_fov_scoring=False,
        fov_scorer=None, 
    ):

        # strip trailing slashes
        root_dir = re.sub(f'{os.sep}+$', '', root_dir)
        self.root_dir = root_dir

        # subdirectory for raw data (where the datastore will be created)
        self.data_dir = os.path.join(self.root_dir, 'raw_data')

        # subdirectory for logfiles
        self.log_dir = os.path.join(self.root_dir, 'logs')
        if os.path.isdir(self.log_dir):
            raise ValueError('The experiment directory %s is not empty' % self.root_dir)
        os.makedirs(self.log_dir, exist_ok=True)

        self.all_events_log_file = os.path.join(self.log_dir, 'all-events.log')
        self.error_events_log_file = os.path.join(self.log_dir, 'error-events.log')
        self.important_events_log_file = os.path.join(self.log_dir, 'important-events.log')
        self.metadata_log_file = os.path.join(self.log_dir, 'experiment-metadata.json')
        self.acquisition_log_file = os.path.join(self.log_dir, 'acquired-images.csv')
        self.afc_log_file = os.path.join(self.log_dir, 'afc-calls.csv')

        # log the current commit
        try:
            repo = git.Repo('..')
            current_commit = repo.commit().hexsha
            self.acquisition_metadata_logger('git_commit', current_commit)
        except Exception:
            print('Warning: no git repo found and git commit hash will not be logged')

        # log experiment name, class name, root directory
        # (the name of the experiment is the name of the directory)
        self.acquisition_metadata_logger('experiment_name', os.path.split(self.root_dir)[-1])
        self.acquisition_metadata_logger('acquisition_name', self.__class__.__name__)
        self.acquisition_metadata_logger('root_directory', self.root_dir)

        # wrap the micromanager interface objects
        self.micromanager_interface = micromanager_interface
        self.micromanager_interface.wrap(self.event_logger)

        # create the operations instance (with logging enabled)
        self.operations = microscope_operations.MicroscopeOperations(self.event_logger)

        # whether to acquire a brightfield stack after the hoechst and GFP stacks
        self.acquire_brightfield_stacks = acquire_brightfield_stacks
        self.acquisition_metadata_logger(
            'brightfield_stacks_acquired', self.acquire_brightfield_stacks
        )

        # whether to skip FOV scoring (only for manual redos)
        self.skip_fov_scoring = skip_fov_scoring
        self.acquisition_metadata_logger('fov_scoring_skipped', self.skip_fov_scoring)

        # create the log_dir for the fov_scorer instance, 
        # and log the directory from which the fov_scorer instance was loaded
        if not self.skip_fov_scoring:
            self.fov_scorer = fov_scorer
            self.fov_scorer.log_dir = os.path.join(self.log_dir, 'fov-scoring')
            self.acquisition_metadata_logger('fov_scorer_save_dir', self.fov_scorer.save_dir)

        self.gfp_channel = settings.gfp_channel_settings
        self.hoechst_channel = settings.hoechst_channel_settings
        self.brightfield_channel = settings.brightfield_channel_settings
        self.fov_selection_settings = settings.fov_selection_settings
        self.autoexposure_settings = settings.autoexposure_settings
        self.brightfield_stack_settings = settings.brightfield_stack_settings

        # use dev stack settings when mocking the API to acquire only a few z-slices
        if mock_micromanager_api:
            self.flourescence_stack_settings = settings.dev_fluorescence_stack_settings
        else:
            self.flourescence_stack_settings = settings.prod_fluorescence_stack_settings
    
        # stage labels for convenience
        self.xystage_label = 'XYStage'
        self.zstage_label = self.flourescence_stack_settings.stage_label

        # log all of the settings
        settings_names = [
            'fov_selection_settings', 
            'autoexposure_settings', 
            'flourescence_stack_settings',
            'brightfield_stack_settings',
            'hoechst_channel',
            'gfp_channel',
            'brightfield_channel',
        ]
        for settings_name in settings_names:
            self.acquisition_metadata_logger(
                settings_name, dataclasses.asdict(getattr(self, settings_name))
            )

        # create the external metadata
        external_metadata = {'pml_id': pml_id, 'platemap_type': platemap_type}

        # if the platemap is canonical half-plate imaging, 
        # we hard-code the parental_line, electroporation_id and round_id
        # (note that the round_id of 'R02' corresponds to imaging a thawed plate for the first time)
        if platemap_type != 'custom':
            external_metadata['parental_line'] = 'czML0383'
            external_metadata['imaging_round_id'] = 'R02'
            external_metadata['plate_id'] = plate_id
        
        # save the external metadata
        with open(os.path.join(self.root_dir, 'metadata.json'), 'w') as file:
            json.dump(external_metadata, file)


    def event_logger(self, message, newline=False):
        '''
        Append a message to the event log

        Note that this method is also passed to, and called from, 
        - some methods in the microscope_operations module
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
        if message_is_important:
            print(message)


    def acquisition_metadata_logger(self, key, value):
        '''
        Append a key-value pair to the acquisition-level metadata (which is just a JSON object)

        This log is intended to capture metadata like the name of the acquisition subclass,
        the name of the imaging experiment, the start and end times of the acquisition itself,
        the git commit hash of the dragonfly-automation repo when the acquisition was run,
        and also all of the acquisition-level settings 
        (autoexposure, stack, and default channel settings)
        '''
        metadata = {}
        if os.path.isfile(self.metadata_log_file):
            with open(self.metadata_log_file, 'r') as file:
                metadata = json.load(file)
        
        metadata[key] = value
        with open(self.metadata_log_file, 'w') as file:
            json.dump(metadata, file)


    def afc_logger(self, **kwargs):
        '''
        Append a row to the AFC log
        
        This log is intended to record the position of the FocusDrive
        before and after AFC is called, as well as whether any errors occur.
        '''
        row = {'timestamp': utils.timestamp(), **kwargs}
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

        Note that this method must be called manually after each call to operations.acquire_stack     
        '''
        row = {'timestamp': utils.timestamp(), **channel_settings.__dict__, **kwargs}
        if os.path.isfile(self.acquisition_log_file):
            d = pd.read_csv(self.acquisition_log_file)
            d = d.append(row, ignore_index=True)
        else:
            d = pd.DataFrame([row])
        d.to_csv(self.acquisition_log_file, index=False)


    def setup(self):
        '''
        Commands to execute before the acquisition begins
        setting the autofocus mode, camera mode, various synchronization commands
        '''
        self.event_logger('ACQUISITION INFO: Calling setup method')
        self.acquisition_metadata_logger('setup_timestamp', utils.timestamp())

        # create the datastore
        self.event_logger('ACQUISITION INFO: Creating datastore at %s' % self.data_dir)
        self.micromanager_interface.create_datastore(self.data_dir)

        # change the autofocus mode to AFC
        af_manager = self.micromanager_interface.mm_studio.getAutofocusManager()
        af_manager.setAutofocusMethodByName("Adaptive Focus Control")

        # these `assignImageSynchro` calls are copied directly from Nathan's script
        self.micromanager_interface.mm_core.assignImageSynchro(self.zstage_label)
        self.micromanager_interface.mm_core.assignImageSynchro(self.xystage_label)

        self.micromanager_interface.mm_core.assignImageSynchro(
            self.micromanager_interface.mm_core.getShutterDevice()
        )
        self.micromanager_interface.mm_core.assignImageSynchro(
            self.micromanager_interface.mm_core.getCameraDevice()
        )

        # turn on auto shutter mode 
        # (this means that the shutter automatically opens and closes when an image is acquired)
        self.micromanager_interface.mm_core.setAutoShutter(True)
        self.event_logger('ACQUISITION INFO: Exiting setup method')


    def cleanup(self):
        '''
        Post-acquisition cleanup - close ('freeze') the datastore
        '''
        self.event_logger('ACQUISITION INFO: Calling cleanup method')

        # freeze the datastore (this takes 20-30min for a full 96-well plate acquisition)
        self.micromanager_interface.freeze_datastore()

        # log the time
        self.acquisition_metadata_logger('cleanup_timestamp', utils.timestamp())
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


    def run(self, mode='prod', test_mode_well_id=None):
        '''
        The main acquisition workflow
        mode : 'test' or 'prod'
            in test mode, only the first well is visited, and only one z-stack is acquired
            (at a position that is *not* among those selected by self.select_positions)
        test_mode_well_id : the well to image in test mode (if None, the first well is used)

        Note that the positions returned by mm_studio.getPositionList() must have been generated
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
        mm_position_list = self.micromanager_interface.mm_studio.getPositionList()
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

        # in test mode, only visit a user-specified test well or else the first well
        if mode == 'prod':
            well_ids_to_visit = unique_well_ids

        elif mode == 'test':
            self.event_logger(
                'ACQUISITION WARNING: Acquisition is running in test mode', newline=True
            )
            if test_mode_well_id is not None:
                if test_mode_well_id in unique_well_ids:
                    well_ids_to_visit = [test_mode_well_id]
                else:
                    well_ids_to_visit = [unique_well_ids[0]]
                    self.event_logger(
                        'ACQUISITION WARNING: '
                        'The specified test-mode well_id %s is not in the position list, '
                        'so the first well in the position list will be used for the test'
                        % test_mode_well_id,
                    )
            else:
                well_ids_to_visit = [unique_well_ids[0]]
                self.event_logger(
                    'ACQUISITION WARNING: No test-mode well_id was specified, '
                    'so the first well in the position list will be used for the test',
                )

        last_afc_updated_focusdrive_position = None
        for well_id in well_ids_to_visit:
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
                selected_positions = all_well_positions
            else:
                self.event_logger(
                    'ACQUISITION INFO: Scoring all FOVs in well %s' % well_id, newline=True
                )
                selected_positions, last_afc_updated_focusdrive_position = self.select_positions(
                    all_well_positions, last_afc_updated_focusdrive_position
                )

            if not len(selected_positions):
                self.event_logger(
                    'ACQUISITION WARNING: No FOVs will be imaged in well %s' % well_id
                )
                continue

            # prettify the scores for the event log
            scores = []
            if not self.skip_fov_scoring:
                scores = [
                    '%0.2f' % p['fov_score'] if p.get('fov_score') is not None else 'None' 
                    for p in selected_positions
                ]

            self.event_logger(
                'ACQUISITION INFO: Imaging %d FOVs in well %s (scores: [%s])'
                % (len(selected_positions), well_id, ', '.join(scores)),
                newline=True
            )

            # in test mode, acquire a single unselected FOV
            if mode == 'test':
                self.event_logger(
                    'ACQUISITION WARNING: running in test mode, so the selected positions '
                    'will be replaced with one unselected position',
                    newline=True
                )
                names = [position['name'] for position in selected_positions]
                for position in all_well_positions:
                    if position['name'] not in names:
                        selected_positions = [position]
                        break
            
            self.acquire_positions(selected_positions)
        self.cleanup()
    

    def select_positions(self, positions, last_afc_updated_focusdrive_position=None):
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

        # whether to never call AFC (except at the first position, before the for loop)
        never_call_afc = False

        # sort positions by site number
        positions = sorted(positions, key=lambda position: position['site_num'])

        # go to the first position in the well
        self.operations.go_to_position(self.micromanager_interface, positions[0]['ind'])

        # attempt to call AFC
        afc_did_succeed = self.operations.call_afc(
            self.micromanager_interface, 
            self.event_logger, 
            self.afc_logger, 
            positions[0]['ind']
        )

        # if AFC failed, move the focusdrive to the last AFC-updated position and try AFC again
        if not afc_did_succeed and last_afc_updated_focusdrive_position is not None:
            self.event_logger(
                'SCORING WARNING: The first attempt to call AFC failed, '
                'so a second attempt will be made at the last AFC-updated FocusDrive position of %s'
                % last_afc_updated_focusdrive_position
            )
            self.operations.move_z_stage(
                self.micromanager_interface, 
                stage_label='FocusDrive', 
                position=last_afc_updated_focusdrive_position, 
                kind='absolute'
            )
            afc_did_succeed = self.operations.call_afc(
                self.micromanager_interface, 
                self.event_logger, 
                self.afc_logger, 
                positions[0]['ind']
            )

        # if AFC still failed, we cannot continue
        # (empirically, if AFC has failed at the first position in the well, 
        # it is unlikely to succeed at other positions in the same well)
        if not afc_did_succeed:
            self.event_logger(
                'SCORING ERROR: Both attempts to call AFC at the first site in well %s failed, '
                'so FOVs cannot be scored' % self.current_well_id
            )
            selected_positions = []
            return selected_positions, last_afc_updated_focusdrive_position

        # if AFC succeeded, update the last good focusdrive position
        afc_updated_focusdrive_position = (
            self.micromanager_interface.mm_core.getPosition('FocusDrive')
        )
        # change to the hoechst channel for FOV scoring
        self.operations.change_channel(self.micromanager_interface, self.hoechst_channel)

        # score the FOV at each position
        for ind, position in enumerate(positions):

            # catch xy-stage timeout errors 
            # (happens when the position is too close to the edge of the stage range)
            try:
                self.operations.go_to_position(self.micromanager_interface, position['ind'])
            except py4j.protocol.Py4JJavaError:
                self.event_logger(
                    'SCORING ERROR: The XYStage timed out at position %s' % position['name']
                )
                continue

            # update the FocusDrive position
            self.operations.move_z_stage(
                self.micromanager_interface, 
                stage_label='FocusDrive', 
                position=afc_updated_focusdrive_position, 
                kind='absolute'
            )

            # call AFC (to save time, we call AFC at every nth position)
            call_afc_every_nth = self.fov_selection_settings.num_positions_between_afc_calls + 1
            if not never_call_afc and ind % call_afc_every_nth == 0:
                afc_did_succeed = self.operations.call_afc(
                    self.micromanager_interface, self.event_logger, self.afc_logger, position['ind']
                )
                if afc_did_succeed:
                    afc_updated_focusdrive_position = (
                        self.micromanager_interface.mm_core.getPosition('FocusDrive')
                    )

            position['afc_updated_focusdrive_position'] = afc_updated_focusdrive_position

            # acquire an image of the hoechst signal
            image = self.operations.acquire_image(self.micromanager_interface, self.event_logger)

            # score the FOV
            # note that, given all of the error handling in PipelineFOVScorer, 
            # this try-catch is a last line of defense that should never be needed
            raw_fov_props = None
            try:
                raw_fov_props = self.fov_scorer.score_raw_fov(
                    image, 
                    min_otsu_thresh=self.fov_selection_settings.absolute_intensity_threshold,
                    min_num_nuclei=self.fov_selection_settings.min_num_nuclei,
                    position_props=position, 
                )
            except Exception as error:
                self.event_logger(
                    "SCORING ERROR: an uncaught exception occurred during FOV scoring "
                    "at position '%s': %s" % (position['name'], error)
                )

            # retrieve the score and note it in the event log
            if raw_fov_props is not None:
                score = raw_fov_props.get('score')
                position['fov_score'] = score

                # prettify the score for the event log
                score = '%0.2f' % score if score is not None else score
                self.event_logger(
                    "SCORING INFO: The FOV score at position '%s' was %s (comment: '%s')"
                    % (position['name'], score, raw_fov_props.get('comment'))
                )

        # drop positions without a score
        # (this happens if raw_fov_props.get('score') is None or if there was an uncaught error above)
        positions_with_score = [p for p in positions if p.get('fov_score') is not None]

        # sort positions in descending order by score (from good to bad)
        positions_with_score = sorted(positions_with_score, key=lambda p: -p['fov_score'])

        # list of acceptable positions
        acceptable_positions = [
            p for p in positions_with_score 
            if p['fov_score'] > self.fov_selection_settings.min_score
        ]
        self.event_logger('ACQUISITION INFO: Found %d acceptable FOVs' % len(acceptable_positions))

        # the minimum number of positions to image in a well
        min_num_positions = self.fov_selection_settings.min_num_positions

        # the maximum number of positions to image in a well
        max_num_positions = self.fov_selection_settings.max_num_positions

        # crop the list if there are more acceptable positions than needed
        selected_positions = acceptable_positions[:max_num_positions]

        # remove positions from positions_with_score that are now in selected_positions
        positions_with_score = positions_with_score[len(selected_positions):]

        # if there were too few acceptable positions found, 
        # append the next-highest-scoring positions
        num_positions_short = min_num_positions - len(selected_positions)
        if num_positions_short > 0:
            additional_positions = positions_with_score[:num_positions_short]
            selected_positions.extend(additional_positions)

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
        if len(selected_positions) > 0 and len(selected_positions) < min_num_positions:
            self.event_logger(
                'ACQUISITION WARNING: All %s scored FOVs will be imaged '
                'but this is fewer than the required minimum of %s FOVs'
                % (len(selected_positions), min_num_positions)
            )

        return selected_positions, afc_updated_focusdrive_position


    def go_to_position(self, position):
        '''
        Convenience method used in `acquire_positions`
        to move to a new position and update the FocusDrive position
        '''
        self.operations.go_to_position(self.micromanager_interface, position['ind'])

        # update the FocusDrive if there is an AFC-updated FocusDrive position 
        # associated with this position (these are created in `select_positions`)
        afc_updated_focusdrive_position = position.get('afc_updated_focusdrive_position')
        if afc_updated_focusdrive_position is not None:
            current_position = self.micromanager_interface.mm_core.getPosition('FocusDrive')
            self.event_logger(
                'ACQUISITION INFO: Updating the interpolated FocusDrive position (%s) '
                'with the AFC-updated position (%s)'
                % (current_position, afc_updated_focusdrive_position)
            )
            self.operations.move_z_stage(
                self.micromanager_interface, 
                stage_label='FocusDrive', 
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
            self.micromanager_interface, self.event_logger, self.afc_logger, position['ind']
        )

        # if AFC fails, autoexposure won't work, so we cannot continue
        # (because go_to_position uses the afc_updated_focusdrive_position, this should be very rare)
        if not afc_did_succeed:
            self.event_logger(
                'ACQUISITION ERROR: AFC failed at the first acceptable FOV of well %s, '
                'so autoexposure cannot be run and stacks cannot be acquired'
                % self.current_well_id
            )
            return

        # reset the GFP channel settings
        self.gfp_channel.reset()

        # change the channel to GFP
        self.operations.change_channel(self.micromanager_interface, self.gfp_channel)

        # run the autoexposure algorithm at the first position 
        # (note that laser power and exposure time are modified in-place)
        autoexposure_did_succeed = self.operations.autoexposure(
            self.micromanager_interface,
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
            self.go_to_position(position)

            # attempt to call AFC
            afc_did_succeed = self.operations.call_afc(
                self.micromanager_interface, self.event_logger, self.afc_logger, position['ind']
            )

            # if AFC failed, it is pointless to acquire z-stacks
            if not afc_did_succeed:
                self.event_logger(
                    'ACQUISITION ERROR: stacks will not be acquired because AFC failed'
                )
                continue

            # channel settings for the two fluorescence channels
            all_channel_settings = [
                {
                    'channel': self.hoechst_channel,
                    'stack': self.flourescence_stack_settings,
                }, {
                    'channel': self.gfp_channel,
                    'stack': self.flourescence_stack_settings,
                }
            ]

            # channel settings for the brightfield channel (which has its own z-stack settings)
            if self.acquire_brightfield_stacks:
                all_channel_settings.append({
                    'channel': self.brightfield_channel,
                    'stack': self.brightfield_stack_settings,
                })

            # acquire a z-stack for each channel
            for channel_ind, channel_settings in enumerate(all_channel_settings):
                self.event_logger(
                    "ACQUISITION INFO: Acquiring channel '%s'"
                    % channel_settings['channel'].config_name
                )

                # change the channel
                self.operations.change_channel(
                    self.micromanager_interface, channel_settings['channel']
                )

                # acquire the stack
                self.operations.acquire_stack(
                    self.micromanager_interface,
                    stack_settings=channel_settings['stack'],
                    channel_ind=channel_ind,
                    position_ind=position['ind'],
                    position_name=position['name'],
                    event_logger=self.event_logger
                )

                # log the acquisition
                self.acquisition_logger(
                    channel_settings=channel_settings['channel'],
                    position_ind=position['ind'],
                    well_id=position['well_id'],
                    site_num=position['site_num'],
                    afc_did_succeed=afc_did_succeed
                )
        

