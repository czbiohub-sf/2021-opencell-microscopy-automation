
import os
import re
import git
import json
import shutil
import datetime
import numpy as np
import pandas as pd

from dragonfly_automation import utils
from dragonfly_automation import operations
from dragonfly_automation.gateway import gateway_utils
from dragonfly_automation.settings import ChannelSettingsManager
from dragonfly_automation.programs import pipeline_plate_settings as settings


class Program:
    '''
    Base class for programs

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

    def __init__(self, root_dir, fov_classifier, env='dev', verbose=True, test_mode='test-real'):
        '''
        Program instantiation

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
        self.fov_classifier = fov_classifier

        # the name of the experiment is the name of the directory
        self.experiment_name = os.path.split(self.root_dir)[-1]

        self.log_dir = os.path.join(self.root_dir, 'logs')
        self.data_dir = os.path.join(self.root_dir, 'raw_data')

        # assign the log_dir and event logger to the fov_classifier instance
        fov_classifier.log_dir = os.path.join(self.log_dir, 'fov-classification')
        fov_classifier.external_event_logger = self.event_logger

        # check whether data and/or logs already exist for the root_dir
        if os.path.isdir(self.log_dir):
            if env == 'prod':
                raise ValueError('The experiment directory %s is not empty' % self.root_dir)
            if env == 'dev':
                print('WARNING: Removing existing experiment directory')
                shutil.rmtree(self.root_dir)
        
        os.makedirs(self.log_dir, exist_ok=True)

        # event logs (plaintext)
        self.all_events_log_file = os.path.join(
            self.log_dir, '%s_all-events.log' % self.experiment_name)

        self.error_events_log_file = os.path.join(
            self.log_dir, '%s_error-events.log' % self.experiment_name)

        self.important_events_log_file = os.path.join(
            self.log_dir, '%s_important-events.log' % self.experiment_name)
        
        # program metadata log (JSON)
        self.metadata_log_file = os.path.join(
            self.log_dir, '%s_experiment-metadata.json' % self.experiment_name)
        
        # acquisition log (CSV)
        self.acquisition_log_file = os.path.join(
            self.log_dir, '%s_acquired-images.csv' % self.experiment_name)

        # log the current commit
        repo = git.Repo('..')
        if not repo:
            raise ValueError('This script cannot be run outside of a git repo')
        current_commit = repo.commit().hexsha
        self.program_metadata_logger('git_commit', current_commit)

        # log the experiment root directory
        self.program_metadata_logger('root_directory', self.root_dir)

        # create the wrapped py4j objects (with logging enabled)
        self.gate, self.mm_studio, self.mm_core = gateway_utils.get_gate(
            env=self.env, 
            wrap=True, 
            logger=self.event_logger,
            test_mode=test_mode)

        # create the operations instance (with logging enabled)
        self.operations = operations.Operations(self.event_logger)

        # create the datastore
        self._initialize_datastore()


    def event_logger(self, message, newline=False):
        '''
        Append a message to the event log

        Note that this method is also passed to, and called from, 
        - some methods in the operations module
        - the wrappers around the gate, mm_studio, and mm_core objects
        - the logging method of the FOVClassifier instance at self.fov_classifier

        For now, we rely on the correct manual hard-coding of log messages 
        to identify, in the logfile, which of these contexts this method was called from

        '''

        log_filepaths = [self.all_events_log_file]

        # prepend a timestamp to the message
        message = '%s %s' % (utils.timestamp(), message)
        if newline:
            message = '\n%s' % message
 
        # manually-defined 'important' events
        important_labels = [
            'PROGRAM', 'CLASSIFIER', 'AUTOEXPOSURE', 'ERROR', 'WARNING']
        
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


    def program_metadata_logger(self, key, value):
        '''
        Append a key-value pair to the program-level metadata (which is just a JSON object)

        This log is intended to capture metadata like the name of the program subclass,
        the name of the imaging experiment, the start and end times of the acquisition itself,
        the git commit hash of the dragonfly-automation repo when the program was run,
        and also all of the program-level settings (autoexposure, stack, and default channel settings)

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

        self.event_logger('PROGRAM INFO: Creating datastore at %s' % self.data_dir)
        self.datastore = self.mm_studio.data().createMultipageTIFFDatastore(
            self.data_dir,
            should_generate_separate_metadata, 
            should_split_positions)

        self.mm_studio.displays().createDisplay(self.datastore)
        
    
    def setup(self):
        '''
        Commands to execute before the acquisition begins
        e.g., setting the autofocus mode, camera mode, various synchronization commands
        '''
        self.program_metadata_logger('setup_timestamp', utils.timestamp())


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
        self.program_metadata_logger('cleanup_timestamp', utils.timestamp())

    
class PipelinePlateProgram(Program):
    '''
    This program is a re-implementation of Nathan's pipeline plate acquisition script

    It acquires DAPI and GFP z-stacks at some number of positions
    in some number of wells on a 96-well plate.

    See the comments in self.run for more details

    '''
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # log the name of the program subclass
        self.program_metadata_logger('program_name', self.__class__.__name__)

        # initialize channel managers for GFP and DAPI
        self.gfp_channel = ChannelSettingsManager(settings.gfp_channel_settings)
        self.dapi_channel = ChannelSettingsManager(settings.dapi_channel_settings)
        
        # copy the autoexposure settings
        self.autoexposure_settings = settings.autoexposure_settings

        # different stack settings for dev and prod
        # (just to reduce the number of slices acquired in dev mode)
        if self.env == 'prod':
            self.stack_settings = settings.prod_stack_settings
        if self.env == 'dev':
            self.stack_settings = settings.dev_stack_settings
    
        # the maximum number of FOVs (that is, z-stacks) to acquire per well
        # (note that if few FOVs are accepted during the FOV assessment, 
        # we may end up with fewer than this number of stacks for some wells)
        self.max_num_stacks_per_well = 6

        # stage labels for convenience
        self.xystage_label = 'XYStage'
        self.zstage_label = self.stack_settings.stage_label
    
        # manually log all of the settings
        self.program_metadata_logger(
            'autoexposure_settings', 
            dict(self.autoexposure_settings._asdict()))

        self.program_metadata_logger(
            'stack_settings', 
            dict(self.stack_settings._asdict()))
    
        self.program_metadata_logger(
            'dapi_channel',
            self.dapi_channel.__dict__)

        self.program_metadata_logger(
            'gfp_channel',
            self.gfp_channel.__dict__)
    

    def setup(self):
        self.event_logger('PROGRAM INFO: Calling setup method')

        super().setup()

        # change the autofocus mode to AFC
        af_manager = self.mm_studio.getAutofocusManager()
        af_manager.setAutofocusMethodByName("Adaptive Focus Control")

        # these `assignImageSynchro` calls are copied directly from Nathan's script
        # TODO: determine whether they are necessary
        self.mm_core.assignImageSynchro(self.zstage_label)
        self.mm_core.assignImageSynchro(self.xystage_label)
        self.mm_core.assignImageSynchro(self.mm_core.getShutterDevice())
        self.mm_core.assignImageSynchro(self.mm_core.getCameraDevice())

        # turn on auto shutter mode 
        # (this means that the shutter automatically opens and closes when an image is acquired)
        self.mm_core.setAutoShutter(True)
        self.event_logger('PROGRAM INFO: Exiting setup method')


    def cleanup(self):
        '''
        TODO: are there commands that should be executed here
        to ensure the microscope is returned to a 'safe' state?
        '''
        self.event_logger('PROGRAM INFO: Calling cleanup method')
        super().cleanup()
        self.event_logger('PROGRAM INFO: Exiting cleanup method')


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
            self.event_logger('PROGRAM ERROR: Unexpected site label %s' % label)
        
        well_id, site_num = result[0]
        site_num = int(site_num)
        return well_id, site_num


    def run(self):
        '''
        The main acquisition workflow

        Overview
        --------


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
        all_positions = []
        mm_position_list = self.mm_studio.getPositionList()
        for position_ind in range(mm_position_list.getNumberOfPositions()):
            
            mm_position = mm_position_list.getPosition(position_ind)
            position_label = mm_position.getLabel()
            well_id, site_num = self.parse_hcs_position_label(position_label)

            # construct a human-readable and unique name for the current position
            # (used in acquire_stack to determine the name of the TIFF file)
            position_name = f'{position_ind}-{well_id}-{site_num}'

            all_positions.append({
                'ind': position_ind,
                'label': position_label,
                'name': position_name,
                'well_id': well_id,
                'site_num': site_num,
            })

        # list of *order-preserved* unique well_ids 
        # (assumes that all positions in each well appear together in a single contiguous block)
        unique_well_ids = []
        well_id = all_positions[0]['well_id']
        for position in all_positions:
            if position['well_id'] != well_id:
                well_id = position['well_id']
                unique_well_ids.append(well_id)

        # loop over wells
        for well_id in unique_well_ids:
            self.event_logger('PROGRAM INFO: Scoring all FOVs in well %s' % well_id, newline=True)
            self.current_well_id = well_id
            
            # positions in this well
            positions = [p for p in all_positions if p['well_id'] == well_id]

            # score and rank the positions
            positions_to_acquire = self.select_positions(positions)
            if not len(positions_to_acquire):
                self.event_logger('PROGRAM WARNING: No acceptable FOVs were found in well %s' % well_id)
            else:
                # pretty array of scores for the event log
                scores = ', '.join(['%0.2f' % p['fov_score'] for p in positions_to_acquire])
                self.event_logger(
                    'PROGRAM INFO: Imaging %d FOVs in well %s (scores: [%s])' % \
                        (len(positions_to_acquire), well_id, scores),
                    newline=True)
    
                self.acquire_positions(positions_to_acquire)

        self.cleanup()
    

    def select_positions(self, positions):
        '''
        Score and rank the positions and select the subset of positions to acquire
        
        Usually, the positions correspond to all positions in one well,
        but we do not assume that this is the case here. 

        '''

        # threshold to define acceptable FOVs
        abs_min_score = 0.15

        # the min and max number of positions to image in a well
        min_num_positions = 1
        max_num_positions = 6

        positions_to_image = []

        # sort positions by site number
        positions = sorted(positions, key=lambda position: position['site_num'])

        # move to the first position in the well
        self.operations.go_to_position(self.mm_studio, self.mm_core, positions[0]['ind'])

        # attempt to call AFC
        # TODO: more robust handling of AFC timeout
        autofocus_did_succeed = self.operations.autofocus(self.mm_studio, self.mm_core)
        if not autofocus_did_succeed:
            self.event_logger('PROGRAM ERROR: AFC failed and stacks will not be acquired')
            return positions_to_image

        # get the AFC-updated FocusDrive z-position
        focusdrive_position = self.mm_core.getPosition('FocusDrive')

        # change to the DAPI channel before FOV scoring
        self.operations.change_channel(self.mm_core, self.dapi_channel)

        # generate the scores for all positions in the well
        for position in positions:

            position_ind = position['ind']
            operations.go_to_position(self.mm_studio, self.mm_core, position_ind)

            # update the FocusDrive position (mimics running AFC)
            self.mm_core.setPosition('FocusDrive', focusdrive_position)

            # acquire an image of the DAPI signal for the FOV assessment
            image = self.operations.acquire_image(self.gate, self.mm_studio, self.mm_core)

            # score the FOV (a score of 0 corresponds to 'terrible' and 1 to 'great')
            # note that, given all of the error handling in FOVClassifier, 
            # the try-catch is a last line of defense that should never be needed
            try:
                fov_score = self.fov_classifier.classify_raw_fov(image, position_ind=position_ind)
            except Exception as error:
                fov_score = -999
                self.event_logger(
                    'PROGRAM ERROR: an uncaught exception occured during FOV classification: %s' % error)

            position['fov_score'] = fov_score

        # sort positions in descending order by score (from good to bad)
        positions = sorted(positions, key=lambda p: -p['fov_score'])

        # list of acceptable positions
        acceptable_positions = [p for p in positions if p['fov_score'] > abs_min_score]

        self.event_logger('PROGRAM INFO: Found %d acceptable FOVs' % len(acceptable_positions))

        # select only the highest-scoring positions 
        # if there are more acceptable positions than we need
        if len(acceptable_positions) > max_num_positions:
            positions_to_image = positions[:max_num_positions]
        
        # select the highest-scoring positions if there are too few acceptable positions
        elif len(acceptable_positions) < min_num_positions:
            positions_to_image = positions[:min_num_positions]
            self.event_logger(
                'PROGRAM INFO: Too few acceptable FOVs were found but the best %d FOVs will be imaged anyway' % min_num_positions)
        else:
            positions_to_image = acceptable_positions

        return positions_to_image


    def acquire_positions(self, positions):
        '''
        Acquire z-stacks at the positions listed in positions

        We assume that all of these positions correspond to *one* well

        # TODO: check that all positions are in the same well
        # TODO: more event logging ('imaging position 1/6 in well A10')

        '''
        
        self.event_logger(
            'PROGRAM INFO: Running the autoexposure algorithm on the first acceptable FOV of well %s' % self.current_well_id)
    
        # go to the first position
        self.operations.go_to_position(self.mm_studio, self.mm_core, positions[0]['ind'])

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
            stack_settings=self.stack_settings,
            autoexposure_settings=self.autoexposure_settings,
            channel_settings=self.gfp_channel,
            event_logger=self.event_logger)

        if not autoexposure_did_succeed:
            # TODO: decide how to handle this situation
            self.event_logger("PROGRAM ERROR: autoexposure failed in well %s" % self.current_well_id)
            return

        for position in positions:
            self.event_logger('PROGRAM INFO: Acquiring stacks at position %s' % position['name'])

            # go to the position
            position_ind = position['ind']
            self.operations.go_to_position(self.mm_studio, self.mm_core, position_ind)

            # acquire the stacks
            channels = [self.dapi_channel, self.gfp_channel]
            for channel_ind, channel_settings in enumerate(channels):
                self.event_logger("PROGRAM INFO: Acquiring channel '%s'" % channel_settings.config_name)

                # change the channel
                self.operations.change_channel(self.mm_core, channel_settings)

                # acquire the stack
                self.operations.acquire_stack(
                    mm_studio=self.mm_studio,
                    mm_core=self.mm_core, 
                    datastore=self.datastore, 
                    stack_settings=self.stack_settings,
                    channel_ind=channel_ind,
                    position_ind=position_ind,
                    position_name=position['name'])
                
                # log the acquisition
                self.acquisition_logger(
                    channel_settings=channel_settings,
                    position_ind=position_ind,
                    well_id=self.current_well_id,
                    site_num=position['site_num'])

        return


