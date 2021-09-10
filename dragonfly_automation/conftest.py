from dragonfly_automation.tests.mocks import mm2python_mocks
import os
import pytest
import imageio
import pathlib

from dragonfly_automation.fov_models import PipelineFOVScorer


@pytest.fixture(scope='session')
def project_dir():
    return pathlib.Path(__file__).parent.parent


@pytest.fixture(scope='session')
def snaps_dir(project_dir):
    return project_dir / 'dragonfly_automation' / 'tests' / 'artifacts' / 'snaps'


@pytest.fixture(scope='session')
def fov_snap_and_score(snaps_dir):
    snap = imageio.imread(snaps_dir / 'good-1.tif')
    expected_score = 0.85
    return snap, expected_score


@pytest.fixture(scope='session')
def fov_snaps_high_score(snaps_dir):
    return [
        imageio.imread(snaps_dir / 'good-1.tif'),
        imageio.imread(snaps_dir / 'good-2.tif'),
        imageio.imread(snaps_dir / 'good-3.tif'),
    ]


@pytest.fixture(scope='session')
def fov_snaps_low_score(snaps_dir):
    snaps = []
    for kind in ['clumpy', 'overconfluent', 'sparse']:
        snaps.extend([
            imageio.imread(snaps_dir / f'{kind}-1.tif'),
            imageio.imread(snaps_dir / f'{kind}-2.tif'),
            imageio.imread(snaps_dir / f'{kind}-3.tif'),
        ])
    return snaps


@pytest.fixture(scope='session')
def fov_snaps_too_few(snaps_dir):
    return [
        imageio.imread(snaps_dir / 'too-few-1.tif'),
        imageio.imread(snaps_dir / 'too-few-2.tif'),
        imageio.imread(snaps_dir / 'too-few-3.tif'),
    ]


@pytest.fixture(scope='session')
def fov_snap_no_nuclei(snaps_dir):
    return imageio.imread(snaps_dir / 'no-nuclei-1.tif')


@pytest.fixture(scope='session')
def trained_fov_scorer(project_dir):

    fov_model_dir = str(project_dir / 'models' / '2019-10-08')
    fov_scorer = PipelineFOVScorer(save_dir=fov_model_dir, mode='prediction', random_state=42)
    fov_scorer.load()
    fov_scorer.train()
    fov_scorer.validate()
    return fov_scorer


@pytest.fixture(scope='function')
def event_logger():
    class EventLogger:
        def __init__(self):
            self.events = []

        def __call__(self, event):
            self.events.append(event)

    return EventLogger()
