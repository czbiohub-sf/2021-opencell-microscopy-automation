
from dragonfly_automation.gateway import mock_gateway

try:
    from py4j.java_gateway import JavaGateway
except ImportError:
    print("Warning: py4j is not installed - 'prod' mode will not work")


class Py4jWrapper(object):
    '''
    Intercept and log calls to method attributes of a py4j object

    Note that we do not allow keyword arguments,
    because py4j objects seem to accept only positional arguments
    '''

    def __init__(self, obj, logger):
        self.wrapped_obj = obj
        self.logger = logger


    def __repr__(self):
        return self.wrapped_obj.__repr__()


    @staticmethod
    def is_class_instance(obj):
        # TODO: fix this hack-ish and possibly noncanonical logic
        return hasattr(obj, '__dict__')


    @classmethod
    def prettify_arg(cls, arg):
        if isinstance(arg, Py4jWrapper):
            return '<%s>' % arg.wrapped_obj.__class__.__name__
        elif cls.is_class_instance(arg):
            return '<%s>' % arg.__class__.__name__
        return arg


    def __getattr__(self, name):

        attr = getattr(self.wrapped_obj, name)

        # ignore calls to non-method attributes
        # (note that this built-in `callable` method seems to be specific to Python 3.3+)
        if not callable(attr):
            return attr

        def wrapper(*args):
            
            # construct the log message
            pretty_args = tuple([self.prettify_arg(arg) for arg in args])
            message = f'''MM2PYTHON: {self.wrapped_obj.__class__.__name__}.{name}{pretty_args}'''
            
            # log the message
            self.logger(message)

            # make the method call and handle the result
            result = attr(*args)
            if result == self.wrapped_obj:
                return self

            elif self.is_class_instance(result):
                return Py4jWrapper(result, self.logger)

            else:
                return result

        return wrapper 

    
def get_gate(env='dev', wrap=False, logger=None):

    if env=='dev' or env=='test':
        gate = mock_gateway.Gate()

    elif env=='prod':
        gateway = JavaGateway()
        gate = gateway.entry_point
    else:
        raise ValueError("env must be one of 'dev', 'test', or 'prod'")

    mm_core = gate.getCMMCore()
    mm_studio = gate.getStudio()
    
    # wrap the py4j objects
    if wrap:
        if not logger:
            raise ValueError('A logger method is required when wrap=True')

        gate = Py4jWrapper(gate, logger)
        mm_core = Py4jWrapper(mm_core, logger)
        mm_studio = Py4jWrapper(mm_studio, logger)

    return gate, mm_studio, mm_core
