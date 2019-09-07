# dragonfly-automation
This is a Python package that organizes, and provides a framework for writing, automation scripts that use `mm2python` to control the spinning disk confocal microscope (nicknamed 'dragonfly'). 


## Usage example

```python
from dragonfly_automation.programs.pipeline_plate_program import PipelinePlateProgram

program = PipelinePlateProgram('path/to/experiment/directory/', env='prod', verbose=True)

# setup the datastore and apply initial/global microscope settings
program.setup()

# run the acquisition script itself
program.run()

# freeze the datastore and return the microscope to a safe state
program.cleanup()

```

## (Aspirational) features
* Modular and reusable methods for common tasks (e.g., autofocusing, adjusting exposures, acquiring z-stacks)
* Methods for less trivial tasks (e.g., dynamic field-of-view selection)
* Built-in metadata and event logging
* Flexible and robust error handling


## Requirements
py4j, mm2python, numpy, scipy, skimage


