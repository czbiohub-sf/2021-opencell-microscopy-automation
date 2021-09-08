import pytest
import numpy as np

from dragonfly_automation import microscope_operations
from dragonfly_automation.settings_schemas import StackSettings


def test_call_afc():
    pass


def test_acquire_image():
    pass


def test_acquire_z_stack(mm_studio, mm_core, datastore, event_logger):
    
    # a z-stack with four z-slices
    stack_settings = StackSettings(
        stage_label='label', relative_bottom=0, relative_top=3, step_size=1
    )

    channel_ind = 0
    position_ind = 123
    microscope_operations.acquire_stack(
        mm_studio, 
        mm_core, 
        datastore, 
        stack_settings, 
        channel_ind=channel_ind,
        position_ind=position_ind,
        position_name='position_name',
        event_logger=event_logger
    )

    assert len(event_logger.messages) == 0
    assert len(datastore._images) == 4

    for image in datastore._images:
        image_coords = image.getCoords()
        assert image_coords.channel_ind == channel_ind
        assert image_coords.stage_position == position_ind


def test_acquire_z_stack_camera_error(mm_studio, mm_core, datastore, event_logger):
    '''
    Test that the acquire_stack method recovers from intermittent hardware errors
    thrown by the getTaggedImage method
    '''
    # a z-stack with four z-slices
    stack_settings = StackSettings(
        stage_label='label', relative_bottom=0, relative_top=3, step_size=1
    )

    # throw an error on the first call to getTaggedImage
    mm_core._throw_get_tagged_image_error = True

    channel_ind = 0
    position_ind = 123
    microscope_operations.acquire_stack(
        mm_studio, 
        mm_core, 
        datastore, 
        stack_settings, 
        channel_ind=channel_ind,
        position_ind=position_ind,
        position_name='position_name',
        event_logger=event_logger
    )

    # there should be one error message
    assert len(event_logger.messages) == 1
    assert 'Mocked getTaggedImage error' in event_logger.messages[0]

    # all z-slices should have been acquired
    assert len(datastore._images) == 4


def test_move_z_stage(mm_core):

    for valid_position in [0, -10, 10, '-10', '1.5']:
        microscope_operations.move_z_stage(
            mm_core, 'label', position=valid_position, kind='absolute'
        )

    for invalid_position in [None, np.nan, np.inf, '', 'hello']:
        with pytest.raises(TypeError):
            microscope_operations.move_z_stage(
                mm_core, 'label', position=invalid_position, kind='absolute'
            )


def test_autoexposure():
    pass

