import json
import os

import pandas as pd

from dragonfly_automation.acquisitions.pipeline_plate_acquisition import PipelinePlateAcquisition
from dragonfly_automation.acquisitions.pipeline_plate_settings import fov_selection_settings


def test_acquisition_without_problems(tmpdir, get_mocked_interface, trained_fov_scorer):
    '''
    Test the acquisition script without any mocked problems or errors
    Note that visiting three wells and two sites per well ensures that all six test FOV snaps
    will be scored (and four of the six are score-able, so the script should acquire four z-stacks)
    '''
    pml_id = 'PML0123'
    plate_id = 'P0001'
    platemap_type = 'none'

    micromanager_interface = get_mocked_interface(
        num_wells=3, num_sites_per_well=2, exposure_state='under'
    )
    acquisition = PipelinePlateAcquisition(
        root_dir=str(tmpdir),
        pml_id=pml_id,
        plate_id=plate_id,
        platemap_type=platemap_type,
        micromanager_interface=micromanager_interface,
        fov_scorer=trained_fov_scorer,
    )
    acquisition.setup()
    acquisition.run(mode='prod')

    # check that all of the logfiles were generated
    for filepath in [
        'experiment-metadata.json',
        'afc-calls.csv',
        'all-events.log',
        'important-events.log',
    ]:
        assert os.path.isfile(tmpdir / 'logs' / filepath)

    # there should have been no errors
    assert not os.path.isfile(tmpdir / 'logs' / 'error-events.log')

    # check that the acquisition script exited without crashing
    with open(tmpdir / 'logs' / 'experiment-metadata.json', 'r') as file:
        metadata = json.load(file)
    assert 'cleanup_timestamp' in metadata.keys()

    # check that the metadata is correct
    with open(tmpdir / 'metadata.json', 'r') as file:
        metadata = json.load(file)
    assert metadata.get('pml_id') == pml_id
    assert metadata.get('plate_id') == plate_id
    assert metadata.get('platemap_type') == platemap_type

    # there is only one good FOV among the test snaps loaded by the getLastMeta mock
    fovs = pd.read_csv(tmpdir / 'logs' / 'fov-scoring' / 'fov-score-log.csv')
    num_good_fovs = (fovs.score > fov_selection_settings.min_score).sum()
    assert num_good_fovs == 1

    # check that z-stacks were acquired at the expected number of FOVs
    # (this is four FOVs, because all six test snaps should have been scored,
    # four of which should have been score-able, and a minimum of two score-able FOVs
    # should have been acquired per well)
    df = pd.read_csv(tmpdir / 'logs' / 'acquired-images.csv')
    num_acquired_fovs = df.shape[0] / 2
    assert num_acquired_fovs == 4


def test_acquisition_with_problems(tmpdir, get_mocked_interface, trained_fov_scorer):
    '''
    Test the acquisition script with multiple common problems and errors:
    intermittent AFC timeouts, over-exposed FOV, and hardware errors
    on calls to getTaggedImage and goToPosition
    '''
    micromanager_interface = get_mocked_interface(
        num_wells=3,
        num_sites_per_well=2,
        exposure_state='over',
        afc_failure_rate=0.1,
        raise_go_to_position_error_once=True,
        raise_get_tagged_image_error_once=True,
    )
    acquisition = PipelinePlateAcquisition(
        root_dir=str(tmpdir),
        pml_id='PML0000',
        plate_id='P0000',
        platemap_type='none',
        micromanager_interface=micromanager_interface,
        fov_scorer=trained_fov_scorer,
    )
    acquisition.setup()
    acquisition.run(mode='prod')

    # there should have been several errors
    assert os.path.isfile(tmpdir / 'logs' / 'error-events.log')

    # check that the acquisition script exited without crashing
    with open(tmpdir / 'logs' / 'experiment-metadata.json', 'r') as file:
        metadata = json.load(file)
    assert 'cleanup_timestamp' in metadata.keys()

    fovs = pd.read_csv(tmpdir / 'logs' / 'fov-scoring' / 'fov-score-log.csv')
    num_good_fovs = (fovs.score > fov_selection_settings.min_score).sum()
    assert num_good_fovs == 1

    # check that the expected number of z-stacks were acquired
    df = pd.read_csv(tmpdir / 'logs' / 'acquired-images.csv')
    num_acquired_fovs = df.shape[0] / 2
    assert num_acquired_fovs == 4


def test_acquisition_with_total_afc_failure(tmpdir, get_mocked_interface, trained_fov_scorer):
    '''
    Test that the acquistion script recovers from AFC failure at all sites in multiple wells
    '''
    micromanager_interface = get_mocked_interface(
        num_wells=3,
        num_sites_per_well=2,
        exposure_state='over',
        afc_failure_rate=0.2,
        afc_always_fail_in_wells=['A1', 'A2'],
    )
    acquisition = PipelinePlateAcquisition(
        root_dir=str(tmpdir),
        pml_id='PML0000',
        plate_id='P0000',
        platemap_type='none',
        micromanager_interface=micromanager_interface,
        fov_scorer=trained_fov_scorer,
    )
    acquisition.setup()
    acquisition.run(mode='prod')

    # there should have been several errors
    assert os.path.isfile(tmpdir / 'logs' / 'error-events.log')

    # check that the acquisition script exited without crashing
    with open(tmpdir / 'logs' / 'experiment-metadata.json', 'r') as file:
        metadata = json.load(file)

    assert 'cleanup_timestamp' in metadata.keys()
