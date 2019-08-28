
import os
import shutil
import datetime
import numpy as np

from dragonfly_automation import operations
from dragonfly_automation import confluency_assessments
from dragonfly_automation.gateway import gateway_utils
from dragonfly_automation.settings import ChannelSettingsManager
from dragonfly_automation.programs import pipeline_plate_settings as settings


class PipelinePlateProgram(object):


    def __init__(self, datastore, root_dir, env='dev', verbose=True):

        self.env = env
        self.root_dir = root_dir
        self.datastore = datastore
        self.verbose = verbose

        # create the log and data directories
        self.log_dir = os.path.join(self.root_dir, 'logs')
        self.data_dir = os.path.join(self.root_dir, 'data')

        if os.path.isdir(self.log_dir):
            if env=='prod':
                raise ValueError('ERROR: data already exists for this experiment')
            print('WARNING: Removing existing log directory')
            shutil.rmtree(self.log_dir)
            os.makedirs(self.log_dir)

        # log file for events
        self.event_log_file = os.path.join(self.log_dir, 'events.log')

        # create the wrapped py4j objects (with logging enabled)
        self.gate, self.mm_studio, self.mm_core = gateway_utils.get_gate(
            env=self.env, 
            wrap=True, 
            logger=self.event_logger)

        # create the operations instance (with logging enabled)
        self.operations = operations.Operations(self.event_logger)

        # initialize channel managers
        self.gfp_channel = ChannelSettingsManager(settings.gfp_channel_settings)
        self.dapi_channel = ChannelSettingsManager(settings.dapi_channel_settings)
        
        # copy the autoexposure settings
        self.autoexposure_settings = settings.autoexposure_settings

        # different stack settings for dev and prod
        # (just to reduce the number of slices acquired in dev mode)
        if env=='prod':
            self.stack_settings = settings.prod_stack_settings
        if env=='dev':
            self.stack_settings = settings.dev_stack_settings
    
        # the maximum number of FOVs (that is, z-stacks) to acquire per well
        # (note that if few FOVs pass the confluency test, 
        # we may end up with fewer than this number of stacks)
        self.max_num_stacks_per_well = 8

        # stage labels for convenience
        self.xystage_label = 'XYStage'
        self.zstage_label = self.stack_settings.stage_label
    

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
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        message = '%s %s' % (timestamp, message)

        # write the message
        with open(self.event_log_file, 'a') as file:
            file.write('%s\n' % message)
        
        if self.verbose:
            print(message)


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
        Generic microscope setup
        set the autofocus mode and run the `mm_core.assignImageSynchro` calls

        '''

        # change the autofocus mode to AFC
        af_manager = self.mm_studio.getAutofocusManager()
        af_manager.setAutofocusMethodByName("Adaptive Focus Control")

        # these `assignImageSynchro` calls are copied directly from Nathan's script
        # TODO: check with Bryant if these are necessary
        self.mm_core.assignImageSynchro(self.zstage_label)
        self.mm_core.assignImageSynchro(self.xystage_label)
        self.mm_core.assignImageSynchro(self.mm_core.getShutterDevice())
        self.mm_core.assignImageSynchro(self.mm_core.getCameraDevice())

        # turn on auto shutter mode 
        # (this means that the shutter automatically opens and closes when an image is acquired)
        self.mm_core.setAutoShutter(True)


    def cleanup(self):
        '''
        Commands that should be run after the acquisition is complete
        (that is, at the very end of self.run)
        '''

        if self.datastore:
            self.datastore.freeze()


    @staticmethod
    def is_new_well(position_label):
        '''
        This is the logic Nathan used to determine 
        whether a position is the first position visited in a new well

        Note that the position_label is assumed to have been generated by 
        the 96-well-plate position plugin for MicroManager
        '''
        is_new_well = ('Site_0' in position_label) or ('Pos_000_000' in position_label)
        return is_new_well


    def run(self):
        '''
        The main acquisition workflow
        
        The outermost loop is over all of the positions loaded into MicroManager
        (that is, all positions returned by `mm_studio.getPositionList()`)
        
        *** We assume that these positions were generated by the 96-well-plate platemap/position plugin ***

        In particular, we assume that the list of positions corresponds 
        to some number of FOVs in some number of distinct wells,
        and that all of the FOVs in each well appear together.

        At each position, the following steps are performed:

            1) move to the new position (this moves the xy-stage and the FocusDrive z-stage)
            3) check if the new position is the first position of a new well
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

            # Here, note that `goToPosition` moves only the stages specified in the position list,
            # which should be the 'XYStage' and 'FocusDrive' devices and *not* the 'PiezoZ' stage

            # TODO: think about moving the goToPosition line after the num_stacks check;
            # this would prevent needlessly moving the stage to any 'extra' positions in a well,
            # but might introduce dangerously long stage movements when moving to a new well
            position.goToPosition(position, self.mm_core)

            # if the position is the first one in a new well
            if self.is_new_well(position_label):
                self.event_logger('PROGRAM INFO: The current position is the first in a new well')

                # only autoexpose on the first position of a new well
                run_autoexposure = True

                # reset the stack counter
                num_stacks_from_current_well = 0

                # reset the GFP channel settings to their default values
                # (they will be adjusted later by the autoexposure algorithm)
                self.gfp_channel.reset()

            # keep moving if enough stacks have already been acquired from the current well
            if num_stacks_from_current_well >= self.max_num_stacks_per_well:
                self.event_logger('PROGRAM INFO: Position skipped because max_num_stacks_per_well was exceeded')
                continue

            # autofocus, maybe autoexpose, assess confluency, and acquire stacks
            did_acquire_stacks = self.maybe_acquire_stacks(position_ind, run_autoexposure)

            if did_acquire_stacks:
                # autoexposure should only be run on the first position imaged in each well
                run_autoexposure = False
                num_stacks_from_current_well += 1

        self.cleanup()
    
    
    def maybe_acquire_stacks(self, position_ind, run_autoexposure=False):
        '''
        Attempt to acquire z-stacks at the current x-y position

        Performs the following steps:

        1) autofocus using the DAPI channel
        2) run the confluency test
        3) run the autoexposure method using the GFP channel if run_autoexposure is true
        4) acquire a z-stack in DAPI and GFP channels and 'put' the stacks in self.datastore

        TODO: implement explicit error handling

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
            position_label=position_ind)

        if not confluency_is_good:
            self.event_logger("PROGRAM WARNING: The confluency test failed (label='%s')" % confluency_label)
        
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
                self.event_logger('PROGRAM ERROR: Autoexposure failed; attempting to continue anyway')

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


