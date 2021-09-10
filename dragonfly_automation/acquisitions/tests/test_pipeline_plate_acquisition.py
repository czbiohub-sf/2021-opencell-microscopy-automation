
import os
import json
import pathlib
import pandas as pd

from dragonfly_automation.acquisitions.pipeline_plate_acquisition import PipelinePlateAcquisition
from dragonfly_automation.acquisitions.pipeline_plate_settings import fov_selection_settings


def test_acquisition_in_test_mode(tmpdir, get_mocked_interface, trained_fov_scorer):

    pml_id = 'PML0123'
    plate_id = 'P0001'
    platemap_type = 'none'

    micromanager_interface = get_mocked_interface()

    acquisition = PipelinePlateAcquisition(
        root_dir=str(tmpdir), 
        pml_id=pml_id,
        plate_id=plate_id,
        platemap_type=platemap_type,
        micromanager_interface=micromanager_interface,
        fov_scorer=trained_fov_scorer,
        skip_fov_scoring=False,
        acquire_brightfield_stacks=False,
    )
    acquisition.setup()

    # run the acquisition in test mode (only visit one well and take one z-stack)
    acquisition.run(mode='test', test_mode_well_id=None)

    # check that all of the logfiles were generated
    for filepath in [
        'experiment-metadata.json', 'afc-calls.csv', 'all-events.log', 'important-events.log'
    ]:
        assert os.path.isfile(tmpdir / 'logs' / filepath)
    
    # there should have been no errors
    assert not os.path.isfile(tmpdir / 'logs' / 'error-events.json')

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

    # get the number of good FOVs found (this varies because mocked FOVs are selected randomly)
    # TODO: what to do with this?
    fovs = pd.read_csv(tmpdir / 'logs' / 'fov-scoring' / 'fov-score-log.csv')
    num_good_fovs = (fovs.score > fov_selection_settings.min_score).sum()
    assert num_good_fovs > 0

    # check that the expected number of z-stacks were acquired 
    # (in test mode this is only two, for 405 and 488 channels at one FOV)
    df = pd.read_csv(tmpdir / 'logs' / 'acquired-images.csv')
    assert df.shape[0] == 2
