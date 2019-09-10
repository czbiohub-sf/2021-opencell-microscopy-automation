
import os
import re
import git
import json
import shutil
import datetime
import numpy as np
import pandas as pd

from dragonfly_automation import operations
from dragonfly_automation import confluency_assessments
from dragonfly_automation.gateway import gateway_utils
from dragonfly_automation.settings import ChannelSettingsManager
from dragonfly_automation.programs import pipeline_plate_settings as settings


class Program(object):
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

    def __init__(self, root_dir, confluency_classifier, env='dev', verbose=True):
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
        self.confluency_classifier = confluency_classifier

        # the name of the experiment is the name of the directory
        self.experiment_name = os.path.split(self.root_dir)[-1]

        self.log_dir = os.path.join(self.root_dir, 'logs')
        self.data_dir = os.path.join(self.root_dir, 'raw_data')

        # check whether data and/or logs already exist for the root_dir
        if os.path.isdir(self.log_dir):
            if env == 'prod':
                raise ValueError('The experiment directory %s is not empty' % self.root_dir)
            if env == 'dev':
                print('WARNING: Removing existing experiment directory')
                shutil.rmtree(self.root_dir)
        
        os.makedirs(self.log_dir, exist_ok=True)

        # events log (plaintext)
        self.event_log_file = os.path.join(
            self.log_dir, '%s_events-log.log' % self.experiment_name)
        
        # program metadata log (JSON)
        self.metadata_log_file = os.path.join(
            self.log_dir, '%s_metadata-log.json' % self.experiment_name)
        
        # acquisition log (CSV)
        self.acquisition_log_file = os.path.join(
            self.log_dir, '%s_acquisitions-log.csv' % self.experiment_name)

        # log the current commit
        repo = git.Repo('../')
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
            logger=self.event_logger)

        # create the operations instance (with logging enabled)
        self.operations = operations.Operations(self.event_logger)

        # create the datastore
        self._initialize_datastore()


    @staticmethod
    def timestamp():
        return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')


    def event_logger(self, message):
        '''
        Append a message to the event log

        Note that this method is also passed to, and called from, 
        - some methods in the operations module
        - the wrappers around the gate, mm_studio, and mm_core objects

        For now, we rely on the correct manual hard-coding of log messages 
        to identify, in the logfile, which of these contexts this method was called from

        '''

        # prepend a timestamp
        message = '%s %s' % (self.timestamp(), message)

        # write the message
        with open(self.event_log_file, 'a') as file:
            file.write('%s\n' % message)
        
        if self.verbose and 'MM2PYTHON' not in message:
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
        row = channel_settings.__dict__
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
        self.program_metadata_logger('setup_timestamp', self.timestamp())


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
        # TODO: figure out why freeze() throws a py4j error
        if self.datastore:
            self.datastore.freeze()

        # log the time
        self.program_metadata_logger('cleanup_timestamp', self.timestamp())

    
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
        # (note that if few FOVs pass the confluency test, 
        # we may end up with fewer than this number of stacks)
        self.max_num_stacks_per_well = 8

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
            self.event_logger('PROGRAM CRITICAL: Unexpected site label %s' % label)
        
        well_id, site_num = result[0]
        site_num = int(site_num)

        return well_id, site_num


    def run(self):
        '''
        The main acquisition workflow
        
        The outermost loop is over all of the positions loaded into MicroManager
        (that is, all positions returned by `mm_studio.getPositionList()`)
        
        *** We assume that these positions were generated by the HCS Site Generator plugin ***

        In particular, we assume that the list of positions corresponds 
        to some number of positions (or 'sites') in some number of distinct wells,
        and that all of the positions in each well appear sequentially together.

        At each position, the following steps are performed:

            1) move to the new position (using the xy-stage and the FocusDrive z-stage)
            3) check if the new position is the first position in a new well
               (if it is, we will need to run the autoexposure method)
            4) check if we have already acquired enough FOVs/stacks for the current well
               (if we do, we'll skip the position)
            5) run autofocus, run autoexposure, assess confluency, and acquire the stacks 
               (see self.maybe_acquire_stacks)

        '''

        # keep track of how many stacks have been acquired from the current well
        num_stacks_from_current_well = 0

        position_list = self.mm_studio.getPositionList()
        for position_ind in range(position_list.getNumberOfPositions()):

            position = position_list.getPosition(position_ind)
            position_label = position.getLabel()

            self.event_logger("PROGRAM INFO: Moving to a new position (index=%d, label='%s')" % \
                (position_ind, position_label))

            # determine the well_id (on the *imaging* plate) and the site number
            # from the position label generated by the HC plugin
            # (note that the site number is a per-well count of positions)
            well_id, site_num = self.parse_hcs_position_label(position_label)

            self.current_well_id = well_id
            self.current_site_num = site_num
            self.current_position_ind = position_ind
            self.current_position_label = position_label
            
            # use the site number to determine whether this is the first position in a new well
            is_new_well = self.current_site_num == 0

            # if the position is the first one in a new well...
            if is_new_well:
                self.event_logger(
                    'PROGRAM INFO: The current position is the first position in well %s' % well_id)

                # ...indicate that the autoexposure algorithm should be run...
                should_do_autoexposure = True

                # ...reset the stack counter...
                num_stacks_from_current_well = 0

                # ...and reset the GFP channel settings to their default values
                # (they will be adjusted later by the autoexposure algorithm)
                self.gfp_channel.reset()

            # skip the position if enough stacks have already been acquired from the current well
            if num_stacks_from_current_well >= self.max_num_stacks_per_well:
                self.event_logger(
                    'PROGRAM INFO: Position skipped because max_num_stacks_per_well was exceeded')
                continue

            # move the stage to the new position
            # note that `goToPosition` moves only the stages specified in the position list,
            # which should be the 'XYStage' and 'FocusDrive' devices and *not* the 'PiezoZ' stage
            position.goToPosition(position, self.mm_core)

            # autofocus, autoexpose if necessary, assess confluency, and acquire stacks
            did_acquire_stacks = self.maybe_acquire_stacks(should_do_autoexposure)

            # autoexposure should only be run on the first *imaged* position in each well
            if did_acquire_stacks:
                should_do_autoexposure = False
                num_stacks_from_current_well += 1

        self.cleanup()
    
    
    def maybe_acquire_stacks(self, should_do_autoexposure=False):
        '''
        Attempt to acquire z-stacks at the current position

        Performs the following steps:

        1) autofocus using the DAPI channel
        2) do the confluency test; if it fails, return
        3) call the autoexposure method using the GFP channel if should_do_autoexposure is true
        4) acquire a z-stack in DAPI and GFP channels and 'put' the stacks in self.datastore

        '''

        # whether we acquired stacks at this position
        did_acquire_stacks = False
    
        # attempt to autofocus (using AFC)
        autofocus_did_succeed = self.operations.autofocus(self.mm_studio, self.mm_core)

        # errors during autofocusing are usually due to AFC timing out
        # TODO: log the error message itself
        if not autofocus_did_succeed:
            self.event_logger(
                'PROGRAM WARNING: AFC failed and stacks will not be acquired')
            return did_acquire_stacks

        # confluency assessment (using the DAPI channel)
        self.operations.change_channel(self.mm_core, self.dapi_channel)
        snap = self.operations.acquire_snap(self.gate, self.mm_studio)

        confluency_is_good, assessment_did_succeed = confluency_assessments.assess_confluency(
            snap,
            self.confluency_classifier,
            log_dir=self.log_dir,
            position_ind=self.current_position_ind)

        # note that if there was an error during confluency assessment, 
        # confluency_is_good will still be False
        if not assessment_did_succeed:
            self.event_logger('PROGRAM WARNING: an error occurred during confluency assessment')
    
        # if the confluency is not good, we should not acquire the stacks
        if not confluency_is_good:
            self.event_logger("PROGRAM INFO: The confluency test failed")
            if self.env == 'dev':
                print("Warning: The confluency test failed but this is ignored in 'dev' mode")
            else:
                return did_acquire_stacks
        else:
            self.event_logger("PROGRAM INFO: The confluency test passed")


        # -----------------------------------------------------------------
        #
        # Autoexposure for the GFP channel 
        # *if* the current position is the first FOV of a new well
        # (Note that laser power and exposure time are modified in-place)
        #
        # -----------------------------------------------------------------
        if should_do_autoexposure:
            self.operations.change_channel(self.mm_core, self.gfp_channel)
            autoexposure_did_succeed = self.operations.autoexposure(
                self.gate,
                self.mm_studio,
                self.mm_core,
                self.stack_settings,
                self.autoexposure_settings,
                self.gfp_channel,
                self.event_logger)

            if not autoexposure_did_succeed:
                # TODO: decide how to handle this situation
                # (autoexposure fails when the GFP signal is so bright
                # that the stack is overexposed even at the minimum laser power)
                self.event_logger(
                    'PROGRAM WARNING: Autoexposure failed, but stacks will be acquired anyway')

        # -----------------------------------------------------------------
        #
        # Acquire the stacks
        #
        # -----------------------------------------------------------------
        channels = [self.dapi_channel, self.gfp_channel]
        for channel_ind, channel_settings in enumerate(channels):

            # change the channel
            self.operations.change_channel(self.mm_core, channel_settings)

            # acquire the stack
            self.operations.acquire_stack(
                self.mm_studio,
                self.mm_core, 
                self.datastore, 
                self.stack_settings,
                channel_ind=channel_ind,
                position_ind=self.current_position_ind)
            
            # log the acquisition
            self.acquisition_logger(
                channel_settings=channel_settings,
                position_ind=self.current_position_ind,
                well_id=self.current_well_id,
                site_num=self.current_site_num)

        # if we're still here, we assume that the stacks were acquired successfully
        did_acquire_stacks = True
        return did_acquire_stacks


