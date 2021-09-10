import os
import py4j
import glob
import tifffile
import tempfile
import numpy as np
import py4j.protocol
import pathlib

from dragonfly_automation import utils
from dragonfly_automation.acquisitions import pipeline_plate_settings as settings
from dragonfly_automation.micromanager_interface import MicromanagerInterface


ALL_WELL_IDS = [
    'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12',
    'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10', 'B11', 'B12',
    'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 
    'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'D11', 'D12', 
    'E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8', 'E9', 'E10', 'E11', 'E12', 
    'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'F11', 'F12', 
    'G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7', 'G8', 'G9', 'G10', 'G11', 'G12', 
    'H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'H7', 'H8', 'H9', 'H10', 'H11', 'H12',
]


def get_mocked_interface(
    num_wells=2,
    num_sites_per_well=6,
    mock_overexposure=True,
    afc_failure_rate=0,
    afc_fail_on_first_n_calls=0,
    raise_go_to_position_error_once=False,
    raise_get_tagged_image_error_once=False,
    get_tagged_image_error_rate=0,
):

    gate = Gate()
    mm_studio = gate.getStudio()
    mm_core = gate.getCMMCore()

    mm_studio.position_list._construct_position_list(ALL_WELL_IDS[:num_wells], num_sites_per_well)

    gate._mock_overexposure = mock_overexposure
    mm_core._get_tagged_image_error_rate = get_tagged_image_error_rate
    mm_core._raise_get_tagged_image_error_once = raise_get_tagged_image_error_once
    mm_core._raise_go_to_position_error_once = raise_go_to_position_error_once

    mm_studio.af_manager.af_plugin._num_full_focus_calls = 0
    mm_studio.af_manager.af_plugin._afc_failure_rate = afc_failure_rate
    mm_studio.af_manager.af_plugin._afc_fails_on_first_n_calls = afc_fail_on_first_n_calls

    return MicromanagerInterface(gate, mm_studio, mm_core)


class MockJavaException:
    '''
    mock for py4j java_exception object expected by py4j.protocol.Py4JJavaError    
    '''
    _target_id = 'target_id'
    _gateway_client = '_gateway_client'


class MockPy4JJavaError(py4j.protocol.Py4JJavaError):

    def __init__(self):
        super().__init__('Mocked Py4JJavaError', MockJavaException())

    def __str__(self):
        return 'Mocked Py4JJavaError'


class BaseMockedPy4jObject:
    '''
    Generic mock for arbitrary instance attributes
    '''
    def __init__(self, name=None):
        self.name = name

    def __getattr__(self, name):
        def wrapper(*args):
            pass
        return wrapper


class Gate:

    def __init__(self):

        self._props = {}
        self._position_ind = None
        self._exposure_time = None

        # the name of the channel config
        self._config_name = None

        # whether to mock overexposure (instead of underexposure)
        self._mock_overexposure = False

        # filepaths to the test FOV snaps
        test_snap_filenames = [
            'no-nuclei-1.tif', 
            'too-few-1.tif',
            'sparse-1.tif', 
            'clumpy-1.tif', 
            'overconfluent-1.tif', 
            'good-1.tif', 
        ]
        snap_dir = pathlib.Path(__file__).parent.parent / 'artifacts' / 'snaps'
        self._snap_filepaths = [snap_dir / filepath for filepath in test_snap_filenames]

        def set_position_ind(position_ind):
            self._position_ind = position_ind

        def set_config_name(name):
            self._config_name = name

        def set_property(label, prop_name, prop_value):
            if self._props.get(label) is None:
                self._props[label] = {}
            self._props[label][prop_name] = prop_value

        def set_exposure_time(exposure_time):
            self._exposure_time = exposure_time

        self.mm_studio = MMStudio(set_position_ind=set_position_ind)

        self.mm_core = MMCore(
            set_property=set_property, 
            set_config_name=set_config_name,
            set_exposure_time=set_exposure_time
        )

    def getCMMCore(self):
        return self.mm_core

    def getStudio(self):
        return self.mm_studio

    def clearQueue(self):
        pass

    def getLastMeta(self):
        '''
        Returns a Meta object that provides access to the last image (or 'snap')
        taken by MicroManager (usually via live.snap()) as an numpy memmap
        '''
        meta = MockedMeta()

        im = tifffile.imread(
            self._snap_filepaths[self._position_ind % len(self._snap_filepaths)]
        )

        if self._config_name == settings.hoechst_channel_settings.config_name:
            channel = settings.hoechst_channel_settings
        elif self._config_name == settings.gfp_channel_settings.config_name:
            channel = settings.gfp_channel_settings

        # get the laser power of the current channel
        laser_power = self._props[channel.laser_line][channel.laser_name]

        # scale the intensities if the channel is 488 to simulate under- or over-exposure
        if '488' in channel.laser_name:
            relative_exposure = (
                (laser_power * self._exposure_time) / 
                (channel.default_laser_power * channel.default_exposure_time)
            )

            # emprically determined factor to yield an overexposed image
            if self._mock_overexposure:
                relative_exposure *= 100

            # empirically determined factor yield an image that is underexposed
            # but can be properly exposed by increasing exposure time
            # (without this, the image is underexposed even with the max exposure time)
            else:
                relative_exposure *= 10

            im = utils.multiply_and_clip_to_uint16(im, relative_exposure)

        meta._make_memmap(im)
        return meta


class MockedMeta:
    '''
    Mock for the objects returned by mm_studio.getLastMeta
    '''
    def _make_memmap(self, im):
        self.shape = im.shape
        self.filepath = os.path.join(tempfile.mkdtemp(), 'mock_snap.dat')
        im = im.astype('uint16')
        fp = np.memmap(self.filepath, dtype='uint16', mode='w+', shape=self.shape)
        fp[:] = im[:]
        del fp

    def getFilepath(self):
        return self.filepath

    def getxRange(self):
        return self.shape[0]

    def getyRange(self):
        return self.shape[1]


class AutofocusManager(BaseMockedPy4jObject):

    def __init__(self):
        self.af_plugin = AutofocusPlugin()

    def getAutofocusMethod(self):
        return self.af_plugin


class AutofocusPlugin(BaseMockedPy4jObject):

    def __init__(self):
        self._num_full_focus_calls = 0
        self._afc_failure_rate = 0.0
        self._afc_fails_on_first_n_calls = 0

    def fullFocus(self):
        if self._afc_failure_rate > 0 and np.random.rand() < self._afc_failure_rate:
            raise MockPy4JJavaError()

        elif self._num_full_focus_calls < self._afc_fails_on_first_n_calls:
            self._num_full_focus_calls += 1
            raise MockPy4JJavaError()

    def getPropertyNames(self):
        return 'Offset', 'LockThreshold'

    def getPropertyValue(self, name):
        return 0
    

class MMStudio(BaseMockedPy4jObject):
    '''
    Mock for MMStudio
    See https://valelab4.ucsf.edu/~MM/doc-2.0.0-beta/mmstudio/org/micromanager/Studio.html
    '''

    def __init__(self, set_position_ind):
        self.set_position_ind = set_position_ind
        self.af_manager = AutofocusManager()
        self.position_list = PositionList(self.set_position_ind)

    def getAutofocusManager(self):
        return self.af_manager

    def getPositionList(self):
        return self.position_list

    def live(self):
        return BaseMockedPy4jObject(name='SnapLiveManager')

    def data(self):
        return DataManager()
    
    def displays(self):
        return BaseMockedPy4jObject(name='DisplayManager')


class MMCore(BaseMockedPy4jObject):
    '''
    Mock for MMCore
    See https://valelab4.ucsf.edu/~MM/doc-2.0.0-beta/mmcorej/mmcorej/CMMCore.html
    '''

    def __init__(self, set_config_name, set_property, set_exposure_time):
        # callbacks to set a device property and the exposure time
        # (needed so that Meta objects can access the laser power and exposure time)
        self._set_property = set_property
        self._set_config_name = set_config_name
        self._set_exposure_time = set_exposure_time

        self._current_z_position = 0
        self._get_tagged_image_error_rate = 0.0
        self._raise_get_tagged_image_error_once = False
        self._raise_go_to_position_error_once = False

    def getPosition(self, *args):
        return self._current_z_position

    def setPosition(self, zdevice, zposition):
        self._current_z_position = zposition

    def setRelativePosition(self, zdevice, offset):
        self._current_z_position += offset

    def setConfig(self, group, name):
        self._set_config_name(name)

    def setExposure(self, exposure_time):
        self._set_exposure_time(exposure_time)
    
    def setProperty(self, label, prop_name, prop_value):
        self._set_property(label, prop_name, prop_value)

    def getTaggedImage(self):
        if self._raise_get_tagged_image_error_once:
            self._raise_get_tagged_image_error_once = False
            raise Exception('Mocked getTaggedImage error')
        elif np.random.rand() < self._get_tagged_image_error_rate:
            raise Exception('Mocked getTaggedImage error')


class DataManager:
    '''
    This object is returned by MMStudio.data()
    '''
    def createMultipageTIFFDatastore(self, *args):
        return MultipageTIFFDatastore()

    def convertTaggedImage(self, *args):
        return Image()        


class MultipageTIFFDatastore(BaseMockedPy4jObject):

    def __init__(self):
        super().__init__(name='Datastore')
        self._images = []

    def putImage(self, image):
        self._images.append(image)


class PositionList:

    def __init__(self, set_position_ind):
        self.set_position_ind = set_position_ind

    def _construct_position_list(self, well_ids, num_sites_per_well):
        '''
        construct the HCS-like list of position labels
        '''
        sites = ['Site_%d' % n for n in range(num_sites_per_well)]
        self._position_list = []
        for well_id in well_ids:
            self._position_list.extend(['%s-%s' % (well_id, site) for site in sites])

    def getNumberOfPositions(self):
        return len(self._position_list)

    def getPosition(self, ind):
        # set_position_ind is called here, instead of in Position.goToPosition,
        # because calls to position.goToPosition are always preceeded by a call to getPosition
        self.set_position_ind(ind)
        return Position(self._position_list[ind])


class Position:

    def __init__(self, label):
        self.label = label

    def __repr__(self):
        return 'Position(label=%s)' % self.label

    def getLabel(self):
        return self.label

    def goToPosition(self, position, mm_core):
        if mm_core._raise_go_to_position_error_once:
            mm_core._raise_go_to_position_error_once = False
            raise MockPy4JJavaError()


class Image:

    def __init__(self):
        self.coords = ImageCoords()
        self.metadata = ImageMetadata()

    def copyWith(self, coords, metadata):
        return self

    def getCoords(self):
        return self.coords

    def getMetadata(self):
        return self.metadata


class ImageCoords:

    def __init__(self):
        self.channel_ind, self.z_ind, self.stage_position = None, None, None
    
    def __repr__(self):
        return (
            'ImageCoords(channel_ind=%s, z_ind=%s, stage_position=%s)'
            % (self.channel_ind, self.z_ind, self.stage_position)
        )

    def build(self):
        return self

    def copy(self):
        return self

    def channel(self, value):
        self.channel_ind = value
        return self

    def z(self, value):
        self.z_ind = value
        return self

    def stagePosition(self, value):
        self.stage_position = value
        return self



class ImageMetadata:

    def __repr__(self):
        return 'ImageMetadata(position_name=%s)' % self.position_name

    def build(self):
        return self

    def copy(self):
        return self
    
    def positionName(self, value):
        self.position_name = value
        return self
