import os
import tempfile
import numpy as np



class Base(object):

    def __init__(self, name=None):
        if name:
            self.name = name

    def __getattr__(self, name):
        def wrapper(*args):
            print('%s.%s%s' % (self.name, name, args))
        return wrapper

    

class JavaGateway(object):

    def __init__(self):
        self.entry_point = Gate()


class Gate(object):

    def __init__(self):

        self.laser_power = None
        def set_laser_power(laser_power):
            self.laser_power = laser_power

        self.exposure_time = None
        def set_exposure_time(exposure_time):
            self.exposure_time = exposure_time

        self.studio = CoreOrStudio(kind='studio')
        self.core = CoreOrStudio(
            kind='core', 
            set_laser_power=set_laser_power,
            set_exposure_time=set_exposure_time)

    def getCMMCore(self):
        return self.core

    def getStudio(self):
        return self.studio

    def getLastMeta(self):
        return Meta(self.laser_power, self.exposure_time)


class Meta(object):

    def __init__(self, laser_power, exposure_time):

        # over-exposed unless laser_power is below 10 and exposure_time is below 40
        if laser_power >= 10 or exposure_time > 40:
            rel_max = 1
        else:
            rel_max = exposure_time/40

        maxx = int(65535 * min(1, rel_max))

        self.shape = (1024, 1024)
        self.filepath = os.path.join(tempfile.mkdtemp(), 'mock_snap.dat')
        im = np.random.randint(0, maxx, self.shape, dtype='uint16')

        fp = np.memmap(self.filepath, dtype='uint16', mode='w+', shape=self.shape)
        fp[:] = im[:]
        del fp


    def getFilepath(self):
        return self.filepath

    def getxRange(self):
        return self.shape[0]

    def getyRange(self):
        return self.shape[1]


class AutofocusManager(Base):
    name = 'AutofocusManager'
    def getAutofocusMethod(self):
        # this is the af_plugin object
        return Base(name='AutofocusMethod')


class CoreOrStudio(object):
    '''
    Mock for mm_core and mm_studio
    '''

    def __init__(self, kind, set_laser_power=None, set_exposure_time=None):
        self.kind = kind
        self._current_z_position = 0
        self.set_laser_power = set_laser_power
        self.set_exposure_time = set_exposure_time


    def __getattr__(self, name):
        def wrapper(*args):
            print('%s.%s%s' % (self.kind, name, args))
        return wrapper


    def getAutofocusManager(self):
        return AutofocusManager()

    def getPosition(self, *args):
        return self._current_z_position

    def setPosition(self, zdevice, zposition):
        self.__getattr__('setPosition')(zdevice, zposition)
        self._current_z_position = zposition

    def setRelativePosition(self, zdevice, offset):
        self.__getattr__('setRelativePosition')(zdevice, offset)
        self._current_z_position += offset

    def getPositionList(self):
        return PositionList()

    def live(self):
        return Base(name='live')

    def setExposure(self, exposure_time):
        self.__getattr__('setExposure')(exposure_time)
        self.set_exposure_time(exposure_time)
    
    def setProperty(self, label, prop_name, prop_value):
        '''
        Explicitly mock setProperty in order to intercept laser power
        '''
        self.__getattr__('setProperty')(label, prop_name, prop_value)

        # hack-ish way to determine whether we're setting the laser power
        if prop_name.startswith('Laser'):
            self.set_laser_power(prop_value)
        
        
class PositionList(object):

    def __init__(self):
        self._position_list = ['Site_%d' % n for n in range(3)]

    def getNumberOfPositions(self):
        return len(self._position_list)

    
    def getPosition(self, index):
        return Position(self._position_list[index])
    


class Position(object):

    def __init__(self, label):
        self.label = label

    def getLabel(self):
        return self.label

    def goToPosition(self, position, mm_core):
        print("Position.goToPosition(label='%s')" % position.label)
    
