# dragonfly-automation
This is a Python package that organizes, and provides a framework for writing, automation scripts that use `mm2python` to control the spinning disk confocal microscope (nicknamed 'dragonfly'). 


## Usage example

```python
from dragonfly_automation.acquisitions.pipeline_plate_acquisition import PipelinePlateAcquisition

aq = PipelinePlateAquisition('path/to/experiment/directory/', env='prod', verbose=True)

# setup the datastore and apply initial/global microscope settings
aq.setup()

# run the acquisition script itself
aq.run()

# freeze the datastore and return the microscope to a safe state
aq.cleanup()

```

## (Aspirational) features
* Modular and reusable methods for common tasks (e.g., autofocusing, adjusting exposures, acquiring z-stacks)
* Pre-trained machine-learning models for dynamic field-of-view scoring and/or classification
* Built-in and thorough logging of metadata, errors, and MicroManager API calls
* Mocks for the MicroManager APIs to facilitate development and enable testing


## Requirements
Python packages: py4j, pandas, numpy, scipy, skimage, sklearn

MicroManager: [mm2python plugin](https://github.com/czbiohub/mm2python)

## TODOs
- set up CI and write tests
- add an ability to 'interact' with a running script to change some critical settings:
	- max number of FOVs per well
	- wells to skip
	- whether to stop the script safely
- better autoexposure method (using threshold_multiotsu method?)
- image fluorescent beads to measure PSF and chromatic aberration

- develop script to acquire short timelapses (instead of z-stacks)

## FOV scoring
- optimize the dbscan epsilon (the clustering is very sensitive to this)
- organize/clarify all of the empirical parameters and thresholds in FOVScorer
- try a CNN on downsampled FOVs (requires augmentation to enforce rotational invariance)
