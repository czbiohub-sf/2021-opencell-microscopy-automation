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

        self._getLastMeta_count = 0

    def getCMMCore(self):
        return CoreOrStudio(kind='core')

    def getStudio(self):
        return CoreOrStudio(kind='studio')

    def getLastMeta(self):
        self._getLastMeta_count += 1
        return Meta(self._getLastMeta_count)


class Meta(object):

    def __init__(self, count):

        maxx = int(65535 * (.99**(count - 1)))

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

    def __init__(self, kind):
        self.kind = kind
        self._current_z_position = 0

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
    
