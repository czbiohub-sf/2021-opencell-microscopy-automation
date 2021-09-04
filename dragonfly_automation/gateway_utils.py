
import time

from dragonfly_automation.tests.mocks import mock_gateway

try:
    from py4j.java_gateway import JavaGateway
except ImportError:
    print("Warning: py4j is not installed - 'prod' mode will not work")


class Py4jWrapper:
    '''
    Intercept and log calls to method attributes of a py4j object

    Note that the wrapper does not accept keyword arguments
    because py4j objects themselves do not accept them
    (see the definition of JavaMember.__call__ in py4j.java_gateway)
    '''

    def __init__(self, obj, event_logger):
        self.wrapped_obj = obj
        self.event_logger = event_logger

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
            
            # construct and log the message
            pretty_args = tuple([self.prettify_arg(arg) for arg in args])
            message = f'''MM2PYTHON: {self.wrapped_obj.__class__.__name__}.{name}{pretty_args}'''
            self.event_logger(message)

            # make the method call and handle the result
            num_tries = 10
            wait_time = 10
            call_succeeded = False
            for _ in range(num_tries):
                try:
                    result = attr(*args)
                    call_succeeded = True
                    break
                except Exception as error:
                    # HACK: do not intercept AFC errors (these are handled in call_afc)
                    # or getTaggedImage errors (these are handled in acquire_stack 
                    # by re-calling both snapImage and getTaggedImage)
                    if name in ['fullFocus', 'getTaggedImage']:
                        raise
                    self.event_logger(
                        'ERROR: An error occurred calling method `%s`: %s' % (name, str(error))
                    )
                    time.sleep(wait_time)

            # this indicates a persistent MicroManager error,
            # and is a situation from which we cannot recover
            if not call_succeeded:
                message = 'Call to method `%s` failed after %s tries' % (name, num_tries)
                self.event_logger('FATAL ERROR: %s' % message)
                raise Exception(message)

            if result == self.wrapped_obj:
                return self
            elif self.is_class_instance(result):
                return Py4jWrapper(result, self.event_logger)
            else:
                return result

        return wrapper 


def get_gate(mock=True, mocked_mode=None, wrap=False, event_logger=None):

    if mock:
        gate = mock_gateway.Gate(mocked_mode)
    else:
        gateway = JavaGateway()
        gate = gateway.entry_point

    mm_core = gate.getCMMCore()
    mm_studio = gate.getStudio()
    
    if wrap:
        if not event_logger:
            raise ValueError('An event_logger method is required when wrap=True')

        gate = Py4jWrapper(gate, event_logger)
        mm_core = Py4jWrapper(mm_core, event_logger)
        mm_studio = Py4jWrapper(mm_studio, event_logger)

    return gate, mm_studio, mm_core
