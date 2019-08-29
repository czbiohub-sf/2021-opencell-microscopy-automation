import os
import glob
import tifffile
import tempfile
import numpy as np


class Base(object):

    def __init__(self, name=None):
        self.name = name

    def __getattr__(self, name):
        def wrapper(*args):
            pass
        return wrapper

    

class JavaGateway(object):

    def __init__(self):
        self.entry_point = Gate()



class Gate(object):

    def __init__(self):
        self.mm_studio = MMStudio()

        self.laser_power = None
        def set_laser_power(laser_power):
            self.laser_power = laser_power

        self.exposure_time = None
        def set_exposure_time(exposure_time):
            self.exposure_time = exposure_time

        self.mm_core = MMCore(
            set_laser_power=set_laser_power,
            set_exposure_time=set_exposure_time)

    def getCMMCore(self):
        return self.mm_core

    def getStudio(self):
        return self.mm_studio

    def getLastMeta(self):
        '''
        Returns a Meta object that provides access to the last image (or 'snap')
        taken by MicroManager (usually via live.snap()) as an numpy memmap

        For an image of noise scaled by laser power and exposure time, use
        meta = RandomMeta(self.laser_power, self.exposure_time)

        For a 'real' image from the tests/test-snaps/ directory, use
        meta = RealMeta()
        '''
        return RealMeta()


class Meta(object):
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


class RealMeta(Meta):
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


class RandomMeta(Meta):
    '''
    Mock for the Meta object that returns an image consisting of noise
    scaled by laser power and exposure time
    (for testing autoexposure algorithms)
    '''
    def __init__(self, laser_power, exposure_time):

        # over-exposed unless laser_power is below 10 and exposure_time is below 40
        if laser_power >= 10 or exposure_time > 40:
            rel_max = 1
        else:
            rel_max = exposure_time/40

        shape = (1024, 1024)
        maxx = int(65535 * min(1, rel_max))
        im = np.random.randint(0, maxx, shape, dtype='uint16')
        self._make_memmap(im)


class AutofocusManager(Base):

    def getAutofocusMethod(self):
        # this is the af_plugin object
        return Base(name='AutofocusMethod')



class MMStudio(Base):
    '''
    Mock for MMStudio
    See https://valelab4.ucsf.edu/~MM/doc-2.0.0-beta/mmstudio/org/micromanager/Studio.html
    '''

    def getAutofocusManager(self):
        return AutofocusManager()

    def getPositionList(self):
        return PositionList()
    
    def live(self):
        return Base(name='SnapLiveManager')

    def data(self):
        return DataManager()


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



class DataManager(object):
    def convertTaggedImage(self, *args):
        return Image()        


class PositionList(object):

    def __init__(self):
        sites = ['Site_%d' % n for n in range(5)]
        self._position_list = ['%s-%s' % ('A1', site) for site in sites]
        self._position_list += ['%s-%s' % ('B10', site) for site in sites]

    def getNumberOfPositions(self):
        return len(self._position_list)

    def getPosition(self, index):
        return Position(self._position_list[index])
    


class Position(object):

    def __init__(self, label):
        self.label = label

    def __repr__(self):
        return 'Position(label=%s)' % self.label

    def getLabel(self):
        return self.label

    def goToPosition(self, position, mm_core):
        # print("Position.goToPosition(label='%s')" % position.label)
        pass


class Image(object):

    def __init__(self):
        self.coords = ImageCoords()
        self.metadata = ImageMetadata()

    def copyWith(self, coords, metadata):
        return self

    def getCoords(self):
        return self.coords

    def getMetadata(self):
        return self.metadata


class ImageCoords(object):

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



class ImageMetadata(object):

    def __repr__(self):
        return 'ImageMetadata(position_name=%s)' % self.position_name

    def build(self):
        return self

    def copy(self):
        return self
    
    def positionName(self, value):
        self.position_name = value
        return self