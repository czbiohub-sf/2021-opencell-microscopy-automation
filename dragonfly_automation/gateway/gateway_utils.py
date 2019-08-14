
from dragonfly_automation.gateway import mock_gateway

try:
    from py4j.java_gateway import JavaGateway
except ImportError:
    print("Warning: py4j is not installed - 'prod' mode will not work")


def get_gate(env='dev'):

    if env=='dev' or env=='test':
        gate = mock_gateway.Gate()

    elif env=='prod':
        gateway = JavaGateway()
        gate = gateway.entry_point
    else:
        raise ValueError("env must be one of 'dev', 'test', or 'prod'")

    mm_core = gate.getCMMCore()
    mm_studio = gate.getStudio()
    return gate, mm_studio, mm_core
