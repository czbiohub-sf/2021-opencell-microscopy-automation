import os
import py4j
import glob
import tifffile
import tempfile
import numpy as np

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

# for rapid testing
NUM_SITES_PER_WELL = 3
WELL_IDS = ['A1', 'B10']

# for simulating a real experiment
NUM_SITES_PER_WELL = 25
WELL_IDS = ALL_WELL_IDS[:48]

FOV_LOG_DIR = '/Users/keith.cheveralls/image-data/dragonfly-automation-tests/20190910/ML0000_20190910-3/logs/confluency-check/confluency-snaps'


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


class Base:

    def __init__(self, name=None):
        self.name = name

    def __getattr__(self, name):
        def wrapper(*args):
            pass
        return wrapper

    
class Gate:

    def __init__(self, test_mode):

        self.test_mode = test_mode
        self.simulate_under_exposure = True

        self.position_ind = None
        def set_position_ind(position_ind):
            self.position_ind = position_ind

            # alternate simulating under- and over-exposure at each new position
            self.simulate_under_exposure = not self.simulate_under_exposure

        self.laser_power = None
        def set_laser_power(laser_power):
            self.laser_power = laser_power

        self.exposure_time = None
        def set_exposure_time(exposure_time):
            self.exposure_time = exposure_time

        self.mm_studio = MMStudio(
            set_position_ind=set_position_ind)

        self.mm_core = MMCore(
            set_laser_power=set_laser_power,
            set_exposure_time=set_exposure_time)

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

        For an image of noise scaled by laser power and exposure time, use
        meta = OverexposureMeta(self.laser_power, self.exposure_time)

        For a 'real' image from the tests/test-snaps/ directory, use
        meta = RandomTestSnapMeta()

        For the real logged image corresponding to self.position_ind
        from a real experiment, use
        meta = LoggedImageMeta(log_dir=FOV_LOG_DIR, position_ind=self.position_ind)
        '''

        if self.test_mode == 'simulate-exposure':
            if self.simulate_under_exposure:
                meta = UnderexposureMeta(self.laser_power, self.exposure_time)
            else:
                meta = OverexposureMeta(self.laser_power, self.exposure_time)
        
        if self.test_mode == 'random-real':
            meta = RandomTestSnapMeta()

        if self.test_mode == 'logged-real':
            meta = LoggedImageMeta(
                fov_log_dir=FOV_LOG_DIR, 
                position_ind=self.position_ind)

        return meta


class Meta:
    '''
    Base class for Meta mocks
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


class LoggedImageMeta(Meta):

    def __init__(self, fov_log_dir, position_ind):
        
        filename = 'confluency_snap_pos%05d_RAW.tif' % position_ind
        im = tifffile.imread(os.path.join(fov_log_dir, filename))
        self._make_memmap(im)


class RandomTestSnapMeta(Meta):
    '''
    Mock for the Meta object that returns a random test snap
    from the tests/test-snaps/ directory
    (for testing confluency assessment)

    '''

    def __init__(self):

        # hack-ish way to find the directory of test snaps
        this_dir = os.path.dirname(__file__)
        package_dir = os.sep.join(this_dir.split(os.sep)[:-2])
        snap_dir = os.path.join(package_dir, 'tests', 'test-snaps', '*.tif')

        # randomly select a test snap
        snap_filepaths = glob.glob(snap_dir)
        ind = np.random.randint(0, len(snap_filepaths), 1)
        im = tifffile.imread(snap_filepaths[int(ind)])
        self._make_memmap(im)


class UnderexposureMeta(Meta):
    '''
    Mock for the Meta object that returns an image consisting of noise
    scaled by laser power and exposure time
    (for testing autoexposure algorithms)
    '''
    def __init__(self, laser_power, exposure_time):

        shape = (1024, 1024)

        # rel_max = 1 at default laser power and exposure time
        rel_max = (laser_power * exposure_time)/500

        minn = 0
        maxx = int(min(65535, 5000 * rel_max))
        im = np.random.randint(minn, maxx, shape, dtype='uint16')
        self._make_memmap(im)


class OverexposureMeta(Meta):
    '''
    Mock for the Meta object that returns an image consisting of noise
    scaled by laser power and exposure time
    (for testing autoexposure algorithms)
    '''
    def __init__(self, laser_power, exposure_time):

        shape = (1024, 1024)

        # rel_max = 2 at default laser power and exposure time
        rel_max = (laser_power * exposure_time)/250

        minn = int(min(65535 - 1, 40000 * rel_max))
        maxx = int(min(65535, 65535 * rel_max))
        im = np.random.randint(minn, maxx, shape, dtype='uint16')
        self._make_memmap(im)


class AutofocusManager(Base):
    def getAutofocusMethod(self):
        # this is the af_plugin object
        return AutofocusMethod()


class AutofocusMethod(Base):
    '''
    Mock for the af_plugin object
    '''
    def fullFocus(self):
        # TODO: programmatically specify this flag
        mock_afc_timeout = False
        if mock_afc_timeout:
            raise MockPy4JJavaError()
        else:
            return

    def getPropertyNames(self):
        return 'Offset', 'LockThreshold'

    def getPropertyValue(self, name):
        return 0
    

class MMStudio(Base):
    '''
    Mock for MMStudio
    See https://valelab4.ucsf.edu/~MM/doc-2.0.0-beta/mmstudio/org/micromanager/Studio.html
    '''

    def __init__(self, set_position_ind):
        self.set_position_ind = set_position_ind

    def getAutofocusManager(self):
        return AutofocusManager()

    def getPositionList(self):
        return PositionList(self.set_position_ind)
    
    def live(self):
        return Base(name='SnapLiveManager')

    def data(self):
        return DataManager()
    
    def displays(self):
        return Base(name='DisplayManager')


class MMCore(Base):
    '''
    Mock for MMCore
    See https://valelab4.ucsf.edu/~MM/doc-2.0.0-beta/mmcorej/mmcorej/CMMCore.html
    '''

    def __init__(self, set_laser_power, set_exposure_time):
        self._current_z_position = 0
        self.set_laser_power = set_laser_power
        self.set_exposure_time = set_exposure_time

    def getPosition(self, *args):
        return self._current_z_position

    def setPosition(self, zdevice, zposition):
        self._current_z_position = zposition

    def setRelativePosition(self, zdevice, offset):
        self._current_z_position += offset

    def setExposure(self, exposure_time):
        self.set_exposure_time(exposure_time)
    
    def setProperty(self, label, prop_name, prop_value):
        '''
        Explicitly mock setProperty in order to intercept laser power
        '''
        # hack-ish way to determine whether we're setting the laser power
        if prop_name.startswith('Laser'):
            self.set_laser_power(prop_value)



class DataManager:
    '''
    This object is returned by MMStudio.data()
    '''
    def createMultipageTIFFDatastore(self, *args):
        return Base(name='Datastore')

    def convertTaggedImage(self, *args):
        return Image()        


class PositionList:

    def __init__(self, set_position_ind):

        self.set_position_ind = set_position_ind

        # construct the HCS-like list of position labels
        sites = ['Site_%d' % n for n in range(NUM_SITES_PER_WELL)]
        self._position_list = []
        for well_id in WELL_IDS:
            self._position_list += ['%s-%s' % (well_id, site) for site in sites]

    def getNumberOfPositions(self):
        return len(self._position_list)

    def getPosition(self, ind):
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
        # print("Position.goToPosition(label='%s')" % position.label)
        pass


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
        return 'ImageCoords(channel_ind=%s, z_ind=%s, stage_position=%s)' % \
            (self.channel_ind, self.z_ind, self.stage_position)

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