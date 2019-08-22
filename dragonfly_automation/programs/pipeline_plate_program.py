
import os
import numpy as np
from skimage import filters

from dragonfly_automation import (
    constants, operations, assessments
)

from dragonfly_automation.gateway import gateway_utils
from dragonfly_automation.programs import pipeline_plate_settings as settings


class PipelinePlateProgram(object):


    def __init__(self, data_dirpath=None, env='dev'):

        self.env = env
        self.data_dirpath = data_dirpath

        # program settings
        self.settings = settings

        # stage labels for convenience
        self.zstage_label = settings.stack_settings.stage_label
        self.xystage_label = constants.XY_STAGE

        # create the py4j objects
        self.gate, self.mm_studio, self.mm_core = gateway_utils.get_gate(env=env)

        if env=='prod':
            self.datastore = self._initialize_datastore()
        
        if env=='dev':
            # no mock yet for the datastore object
            self.datastore = None


    def _initialize_datastore(self):

        if self.data_dirpath is None:
            raise ValueError('A data directory must be provided')

        os.makedirs(self.data_dirpath, exist_ok=True)

        # these arguments for createMultipageTIFFDatastore are copied from Nathan's script
        should_generate_separate_metadata = True
        should_split_positions = True

        datastore = self.mm_studio.data().createMultipageTIFFDatastore(
            self.data_dirpath, 
            should_generate_separate_metadata, 
            should_split_positions)

        self.mm_studio.displays().createDisplay(datastore)
        return datastore


    def setup(self):
        '''
        Generic microscope setup
        set the autofocus mode and run the `mm_core.assignImageSynchro` calls

        '''

        # get the AutofocusManager and change the autofocus mode to AFC
        self.af_manager = self.mm_studio.getAutofocusManager()
        self.af_manager.setAutofocusMethodByName("Adaptive Focus Control")

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
    def _is_first_position_in_new_well(position_label):
        '''
        This is the logic Nathan used to determine 
        whether a position is the first position in a new well

        Note that the position_label is assumed to have been generated by 
        the 96-well-plate position plugin for MicroManager
        '''
        flag = ('Site_0' in position_label) or ('Pos_000_000' in position_label)
        return flag


    def confluency_test(self, im):
        return True


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

            1) reset the piezo z-stage to zero
            2) move to the new position (this moves the xy-stage and the FocusDrive z-stage)
            3) check if the new position is the first FOV of a new well
               (if it is, we will need to run the autoexposure routine)
            4) check if we already have enough FOVs for the current well
               (if we do, we'll skip the position)
            5) autofocus using the 405 ('DAPI') channel
            6) run the confluency test (and skip the position if it fails)
            7) run the autoexposure routine using the 488 ('GFP') channel,
               *if* the position is the first position of a new well
            8) acquire the z-stack in 405 and 488 and 'put' the stacks in self.datastore

        '''


        position_list = self.mm_studio.getPositionList()
        for position_ind in range(position_list.getNumberOfPositions()):
            print('\n-------- Position %d --------' % position_ind)
            

            # -----------------------------------------------------------------
            #
            # Reset the Piezo z-stage and move the xy-stage to the next position
            #
            # -----------------------------------------------------------------
            operations.move_z_stage(self.mm_core, self.zstage_label, position=0.0, kind='absolute')

            # Here, note that `goToPosition` moves only the stage specified in the position list,
            # which should always be the 'XYStage', *not* the 'PiezoZ' stage
            # TODO: think about moving the goToPosition line to after the num FOV check;
            # this would prevent needlessly moving the stage to any 'extra' positions in a well,
            # but might introduce dangerously long stage movements when moving to a new well
            position = position_list.getPosition(position_ind)
            position.goToPosition(position, self.mm_core)


            # -----------------------------------------------------------------
            #
            # Determine position status, run autofocus, and test for confluency
            #
            # -----------------------------------------------------------------
            # check if the position is the first one in a new well
            new_well_flag = self._is_first_position_in_new_well(position.getLabel())
            if new_well_flag:
                self.num_stacks_from_current_well = 0

                # reset the GFP channel settings to their default values
                # (the DAPI settings never change)
                self.settings.gfp_channel.reset()


            # check if we have already acquired enough FOVs from the current well
            if self.num_stacks_from_current_well >= self.settings.NUM_STACKS_PER_WELL:
                continue

            # autofocus using DAPI 
            operations.change_channel(self.mm_core, self.settings.dapi_channel)
            operations.autofocus(self.af_manager, self.mm_core)    

            # confluency assessment (also using DAPI)
            im = operations.acquire_snap(self.gate, self.mm_studio)
            confluency_is_good, confluency_label = assessments.assess_confluency(im)
    
            if not confluency_is_good:
                print("Warning: confluency test failed (label='%s')" % confluency_label)
                if self.env=='dev':
                    print("Warning: confluency test results are ignored in 'dev' mode")
                else:
                    continue
            

            # -----------------------------------------------------------------
            #
            # Autoexposure for the GFP channel 
            # *if* the current position is the first FOV of a new well
            # (Note that laser power and exposure time are modified in-place)
            #
            # -----------------------------------------------------------------
            if new_well_flag:
                operations.change_channel(self.mm_core, self.settings.gfp_channel)
                autoexposure_did_succeed = operations.autoexposure(
                    self.gate,
                    self.mm_studio,
                    self.mm_core,
                    self.settings.stack_settings,
                    self.settings.autoexposure_settings,
                    self.settings.gfp_channel)

                if not autoexposure_did_succeed:
                    # TODO: decide how to handle this situation
                    print('Warning: autoexposure failure')


            # -----------------------------------------------------------------
            #
            # Acquire the stacks
            #
            # -----------------------------------------------------------------
            channels = [self.settings.dapi_channel, self.settings.gfp_channel]
            for channel_ind, channel_settings in enumerate(channels):

                # change the channel
                operations.change_channel(self.mm_core, channel_settings)

                # acquire the stack
                operations.acquire_stack(
                    self.mm_studio,
                    self.mm_core, 
                    self.datastore, 
                    self.settings.stack_settings,
                    position_ind=position_ind,
                    channel_ind=channel_ind)

    
            # update the FOV count
            self.num_stacks_from_current_well += 1


        self.cleanup()


