import numpy as np
import pytest

from dragonfly_automation import microscope_operations
from dragonfly_automation.acquisitions import pipeline_plate_settings as settings
from dragonfly_automation.settings_schemas import StackSettings


def test_call_afc(get_mocked_interface, event_logger):
    '''
    Test that AFC works in the best case, without any AFC timeouts
    '''
    interface = get_mocked_interface()
    afc_did_succeed = microscope_operations.call_afc(interface, event_logger)
    assert afc_did_succeed


def test_call_afc_with_timeouts(get_mocked_interface, event_logger):
    '''
    Test that AFC recovers from AFC timeout errors at the first three offsets
    '''
    interface = get_mocked_interface(afc_fail_on_first_n_calls=3)
    afc_did_succeed = microscope_operations.call_afc(interface, event_logger)
    assert afc_did_succeed

    # four events - one for each timeout, plus one for the final successful AFC call
    assert len(event_logger.events) == 4

    # this line is dependent on the hard-coded list of offsets in call_afc
    assert 'AFC was called successfully at an offset of 40um' in event_logger.events[-1]

    # the final FocusDrive position should be the fourth AFC offset, at +40um
    assert interface.mm_core._current_z_position == 40


def test_call_afc_with_too_many_timeouts(get_mocked_interface, event_logger):
    '''
    Test that call_afc exits gracefully when AFC always times out
    '''
    interface = get_mocked_interface(afc_failure_rate=1)

    initial_focusdrive_position = 1234
    interface.mm_core._current_z_position = initial_focusdrive_position
    afc_did_succeed = microscope_operations.call_afc(interface, event_logger)
    assert not afc_did_succeed

    # seven events - one for each of the six offsets and a final error event
    assert len(event_logger.events) == 7

    assert 'AFC timed out at all offsets' in event_logger.events[-1]

    # important: when call_afc fails, it should reset the FocusDrive to its initial position
    assert interface.mm_core._current_z_position == initial_focusdrive_position


def test_acquire_z_stack(tmpdir, get_mocked_interface, event_logger):

    # a z-stack with four z-slices
    stack_settings = StackSettings(
        stage_label='label', relative_bottom=0, relative_top=3, step_size=1
    )

    interface = get_mocked_interface()
    interface.create_datastore(tmpdir / 'raw-data')

    channel_ind = 0
    position_ind = 123
    microscope_operations.acquire_stack(
        interface,
        stack_settings,
        channel_ind=channel_ind,
        position_ind=position_ind,
        position_name='position_name',
        event_logger=event_logger,
    )

    assert len(event_logger.events) == 0
    assert len(interface.datastore._images) == 4

    for image in interface.datastore._images:
        image_coords = image.getCoords()
        assert image_coords.channel_ind == channel_ind
        assert image_coords.stage_position == position_ind


def test_acquire_z_stack_camera_error(tmpdir, get_mocked_interface, event_logger):
    '''
    Test that the acquire_stack method recovers from intermittent hardware errors
    thrown by the getTaggedImage method
    '''
    # a z-stack with four z-slices
    stack_settings = StackSettings(
        stage_label='label', relative_bottom=0, relative_top=3, step_size=1
    )

    # throw an error on the first call to getTaggedImage
    interface = get_mocked_interface(raise_get_tagged_image_error_once=True)
    interface.create_datastore(tmpdir / 'raw-data')

    channel_ind = 0
    position_ind = 123
    microscope_operations.acquire_stack(
        interface,
        stack_settings,
        channel_ind=channel_ind,
        position_ind=position_ind,
        position_name='position_name',
        event_logger=event_logger,
    )

    # there should be one error message
    assert len(event_logger.events) == 1
    assert 'ERROR' in event_logger.events[0]

    # all z-slices should have been acquired
    assert len(interface.datastore._images) == 4


def test_move_z_stage(get_mocked_interface):

    interface = get_mocked_interface()

    for valid_position in [0, -10, 10, '-10', '1.5']:
        microscope_operations.move_z_stage(
            interface, 'label', position=valid_position, kind='absolute'
        )

    for invalid_position in [None, np.nan, np.inf, '', 'hello']:
        with pytest.raises(TypeError):
            microscope_operations.move_z_stage(
                interface, 'label', position=invalid_position, kind='absolute'
            )


def test_autoexposure_underexposed_fov(get_mocked_interface, event_logger):

    interface = get_mocked_interface(channel='488', exposure_state='under')
    initial_exposure_time = settings.gfp_channel_settings.exposure_time

    autoexposure_did_succeed = microscope_operations.autoexposure(
        interface,
        settings.fluorescence_stack_settings,
        settings.autoexposure_settings,
        settings.gfp_channel_settings,
        event_logger,
    )
    assert autoexposure_did_succeed

    # check that the exposure time has been increased
    assert settings.gfp_channel_settings.exposure_time > initial_exposure_time


def test_autoexposure_overexposed_fov(get_mocked_interface, event_logger):

    interface = get_mocked_interface(channel='488', exposure_state='over')
    initial_laser_power = settings.gfp_channel_settings.laser_power

    autoexposure_did_succeed = microscope_operations.autoexposure(
        interface,
        settings.fluorescence_stack_settings,
        settings.autoexposure_settings,
        settings.gfp_channel_settings,
        event_logger,
    )
    assert autoexposure_did_succeed

    # check that the exposure time has been increased
    assert settings.gfp_channel_settings.laser_power < initial_laser_power


def test_autoexposure_hopelessly_overexposed_fov(get_mocked_interface, event_logger):

    interface = get_mocked_interface(channel='488', exposure_state='way-over')
    initial_laser_power = settings.gfp_channel_settings.laser_power

    autoexposure_did_succeed = microscope_operations.autoexposure(
        interface,
        settings.fluorescence_stack_settings,
        settings.autoexposure_settings,
        settings.gfp_channel_settings,
        event_logger,
    )
    assert not autoexposure_did_succeed

    # check that the exposure time has been increased
    assert settings.gfp_channel_settings.laser_power < initial_laser_power
