import datetime

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

    def __init__(self, obj, log_file=None, verbose=True):
        self.wrapped_obj = obj
        self.log_file = log_file
        self.verbose = verbose


    def __repr__(self):
        return self.wrapped_obj.__repr__()
    

    @staticmethod
    def is_class_instance(obj):
        # TODO: fix this hack-ish and possibly noncanonical logic
        return hasattr(obj, '__dict__')
        

    @classmethod
    def prettify_arg(cls, arg):
        if isinstance(arg, Py4jWrapper):
            return arg.wrapped_obj.__class__.__name__
        elif cls.is_class_instance(arg):
            return arg.__class__.__name__
        return arg


    def __getattr__(self, name):

        attr = getattr(self.wrapped_obj, name)

        # do nothing if the attribute is not a method
        # note that the built-in `callable` method seems to be specific to Python 3.3+
        if not callable(attr):
            return attr

        def wrapper(*args):
            
            pretty_args = tuple([self.prettify_arg(arg) for arg in args])
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            record = f'''{timestamp} PY4J: {self.wrapped_obj.__class__.__name__}.{name}{pretty_args}'''

            if self.log_file:
                with open(self.log_file, 'a') as file:
                    file.write('%s\n' % record)

            if self.verbose:
                print(record)

            result = attr(*args)
            if result == self.wrapped_obj:
                return self

            elif self.is_class_instance(result):
                return Py4jWrapper(result, self.log_file, self.verbose)

            else:
                return result

        return wrapper 
    
    
def get_gate(env='dev', wrap=False, verbose=False, log_file=None):

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
        gate = Py4jWrapper(gate, log_file, verbose)
        mm_core = Py4jWrapper(mm_core, log_file, verbose)
        mm_studio = Py4jWrapper(mm_studio, log_file, verbose)

    return gate, mm_studio, mm_core
