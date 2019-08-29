
import os
import re
import git
import json
import shutil
import datetime
import numpy as np

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

    def __init__(self, datastore, root_dir, env='dev', verbose=True):
        '''
        Program instantiation

        datastore : py4j datastore object
            passed explicitly here to avoid an unusual py4j error
        root_dir : the imaging experiment directory 
            (usually ends with a directory of the form 'ML0000_20190823')
        env : 'prod' or 'dev'
        verbose : whether to print log messages

        '''
        self.env = env
        self.root_dir = root_dir
        self.datastore = datastore
        self.verbose = verbose

        # create the log and data directories
        self.log_dir = os.path.join(self.root_dir, 'logs')
        self.data_dir = os.path.join(self.root_dir, 'data')
        
        # check whether data/logs already exist for the root_dir
        if os.path.isdir(self.log_dir):
            if env=='prod':
                raise ValueError('ERROR: data already exists for this experiment')
            if env=='dev':
                print('WARNING: Removing existing log directory')
                shutil.rmtree(self.log_dir)
                os.makedirs(self.log_dir)

        # log file for events
        self.event_log_file = os.path.join(self.log_dir, 'events.log')
        
        # JSON log for metadata
        self.metadata_log_file = os.path.join(self.log_dir, 'metadata.json')
        self.metadata_logger('root_directory', self.root_dir)
        
        # log the current commit
        repo = git.Repo('../')
        current_commit = 'unknown'
        if repo:
            current_commit = repo.commit().hexsha
        self.metadata_logger('git_commit', current_commit)

        # log the date and time
        self.metadata_logger('instantiation_timestamp', self.timestamp())

        # create the wrapped py4j objects (with logging enabled)
        self.gate, self.mm_studio, self.mm_core = gateway_utils.get_gate(
            env=self.env, 
            wrap=True, 
            logger=self.event_logger)

        # create the operations instance (with logging enabled)
        self.operations = operations.Operations(self.event_logger)


    @staticmethod
    def timestamp():
        return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')


    def event_logger(self, message):
        '''
        Write a message to the log

        Note that this method is also passed to, and called from, 
        - the operations module
        - the confluency assessment method
        - the wrappers around the gate, mm_studio, and mm_core objects

        For now, we rely on the correct manual hard-coding of log messages 
        to identify, in the logfile, which of these contexts this method was called from

        '''

        # prepend a timestamp
        message = '%s %s' % (self.timestamp(), message)

        # write the message
        with open(self.event_log_file, 'a') as file:
            file.write('%s\n' % message)
        
        if self.verbose:
            print(message)


    def metadata_logger(self, key, value):
        '''
        Append a field to the metadata JSON log
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


    def _initialize_datastore(self):
        '''
        Initialize a datastore object

        *** currently unused because of an unresolved Java error ***
        TODO: figure out why the call here to `createMultipageTIFFDatastore` fails
        '''

        os.makedirs(self.data_dir, exist_ok=True)

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
        raise NotImplementedError

    def run(self):
        '''
        The main acquisition workflow
        '''
        raise NotImplementedError

    def cleanup(self):
        '''
        Commands that should be executed after the acquisition is complete
        (that is, after self.run)
        '''

        # freeze the datastore
        if self.datastore:
            self.datastore.freeze()

        # log the time
        self.metadata_logger('cleanup_timestamp', self.timestamp())

    
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
        self.metadata_logger('program_name', self.__class__.__name__)

        # initialize channel managers for GFP and DAPI
        self.gfp_channel = ChannelSettingsManager(settings.gfp_channel_settings)
        self.dapi_channel = ChannelSettingsManager(settings.dapi_channel_settings)
        
        # copy the autoexposure settings
        self.autoexposure_settings = settings.autoexposure_settings

        # different stack settings for dev and prod
        # (just to reduce the number of slices acquired in dev mode)
        if self.env=='prod':
            self.stack_settings = settings.prod_stack_settings
        if self.env=='dev':
            self.stack_settings = settings.dev_stack_settings
    
        # the maximum number of FOVs (that is, z-stacks) to acquire per well
        # (note that if few FOVs pass the confluency test, 
        # we may end up with fewer than this number of stacks)
        self.max_num_stacks_per_well = 8

        # stage labels for convenience
        self.xystage_label = 'XYStage'
        self.zstage_label = self.stack_settings.stage_label
    
        # manually log all of the settings
        self.metadata_logger(
            'autoexposure_settings', 
            dict(self.autoexposure_settings._asdict()))

        self.metadata_logger(
            'stack_settings', 
            dict(self.stack_settings._asdict()))
    
        self.metadata_logger(
            'dapi_channel',
            self.dapi_channel.__dict__)

        self.metadata_logger(
            'gfp_channel',
            self.gfp_channel.__dict__)
    

    def setup(self):

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


    def cleanup(self):
        '''
        TODO: are there other commands that should be executed here
        to ensure the microscope is returned to a 'safe' state?
        '''
        super().cleanup()


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
            # (the site number is a per-well count of positions)
            well_id, site_num = self.parse_hcs_position_label(position_label)
            
            # this is how we know whether this is the first position in a new well
            is_new_well = site_num==0

            # Here, note that `goToPosition` moves only the stages specified in the position list,
            # which should be the 'XYStage' and 'FocusDrive' devices and *not* the 'PiezoZ' stage

            # TODO: think about moving the goToPosition line after the num_stacks check;
            # this would prevent needlessly moving the stage to any 'extra' positions in a well,
            # but might introduce dangerously long stage movements when moving to a new well
            position.goToPosition(position, self.mm_core)

            # if the position is the first one in a new well
            if is_new_well:
                self.event_logger(
                    'PROGRAM INFO: The current position is the first position in well %s' % well_id)

                # only autoexpose on the first position of a new well
                run_autoexposure = True

                # reset the stack counter
                num_stacks_from_current_well = 0

                # reset the GFP channel settings to their default values
                # (they will be adjusted later by the autoexposure algorithm)
                self.gfp_channel.reset()

            # skip the position if enough stacks have already been acquired from the current well
            if num_stacks_from_current_well >= self.max_num_stacks_per_well:
                self.event_logger(
                    'PROGRAM INFO: Position skipped because max_num_stacks_per_well was exceeded')
                continue

            # autofocus, maybe autoexpose, assess confluency, and acquire stacks
            did_acquire_stacks = self.maybe_acquire_stacks(position_ind, run_autoexposure)

            # autoexposure should only be run on the first *imaged* position in each well
            if did_acquire_stacks:
                run_autoexposure = False
                num_stacks_from_current_well += 1

        self.cleanup()
    
    
    def maybe_acquire_stacks(self, position_ind, run_autoexposure=False):
        '''
        Attempt to acquire z-stacks at the current position

        Performs the following steps:

        1) autofocus using the DAPI channel
        2) do the confluency test; if it fails, return
        3) call the autoexposure method using the GFP channel if run_autoexposure is true
        4) acquire a z-stack in DAPI and GFP channels and 'put' the stacks in self.datastore

        '''

        # whether we acquired stacks at this position
        did_acquire_stacks = False
    
        # autofocus using DAPI 
        self.operations.change_channel(self.mm_core, self.dapi_channel)
        self.operations.autofocus(self.mm_studio, self.mm_core)    

        # confluency assessment (also using DAPI)
        snap = self.operations.acquire_snap(self.gate, self.mm_studio)
        confluency_is_good, confluency_label = confluency_assessments.assess_confluency(
            snap,
            log_dir=self.log_dir,
            position_ind=position_ind)

        if not confluency_is_good:
            self.event_logger("PROGRAM WARNING: The confluency test failed (label='%s')" % \
                confluency_label)
        
            if self.env=='dev':
                print("Warning: confluency test results are ignored in 'dev' mode")
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
        if run_autoexposure:
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
                self.event_logger(
                    'PROGRAM ERROR: Autoexposure failed; attempting to continue anyway')

        # -----------------------------------------------------------------
        #
        # Acquire the stacks
        #
        # -----------------------------------------------------------------
        channels = [self.dapi_channel, self.gfp_channel]
        for channel_ind, channel in enumerate(channels):

            # change the channel
            self.operations.change_channel(self.mm_core, channel)

            # acquire the stack
            self.operations.acquire_stack(
                self.mm_studio,
                self.mm_core, 
                self.datastore, 
                self.stack_settings,
                position_ind=position_ind,
                channel_ind=channel_ind)
    
        # if we're still here, we assume that the stacks were acquired successfully
        did_acquire_stacks = True
        return did_acquire_stacks


