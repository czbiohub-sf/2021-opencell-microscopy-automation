# dragonfly-automation
This repo contains a Python package, `dragonfly_automation`, that provides a framework for writing automation scripts written using the MicroManager API exposed by `mm2python` to control the spinning disk confocal microscope (nicknamed 'dragonfly'). 

## Example

```python
from dragonfly_automation.programs import PipelinePlateProgram

program = PipelinePlateProgram(data_dirpath='/local/path/to/data/', env='dev')

# setup datastore and initial microscope settings
program.setup()

# run the acquisition script itself
program.run()

# freeze the datastore and return the microscope to a safe state
program.cleanup()

```

## Requirements
py4j, numpy, scipy, skimage



## Development


