# dragonfly-automation
This is a Python package that organizes, and provides a framework for writing, automation scripts that use `mm2python` to control the spinning disk confocal microscope (nicknamed 'dragonfly').


## Usage examples
The script is called using a CLI. Some common examples are shown below. See `scripts/run_acquisition.py` for all CLI arguments.
```
// start an acquisition in test mode, using the mocked MicroManager API
// (this can be run in any environment and does not require py4j, mm2python, or MicroManager)
python ./scripts/run_acquisition.py \
--data-dir ./tests/output/ \
--pml-id PML0000 \
--plate-id P0000 \
--platemap-type none \
--mock-micromanager-api \
--mocked-mode random-real \
--mode test

// start an acquisition in test mode, using the real API and well A1 for the test
python ./scripts/run_acquisition.py \
--data-dir ./tests/output/ \
--pml-id PML0000 \
--plate-id P0000 \
--platemap-type none \
--mode test
--test-well A1

// start a full-plate acquisition, using the real API
python ./scripts/run_acquisition.py \
--pml-id PML0123 \
--plate-id P0021 \
--platemap-type none
```

## (Aspirational) features
* Modular and reusable methods for common tasks (e.g., autofocusing, adjusting exposures, acquiring z-stacks)
* Pre-trained machine-learning models for dynamic field-of-view scoring and/or classification
* Built-in and thorough logging of metadata, errors, and MicroManager API calls
* Mocks for the MicroManager APIs to facilitate development and enable testing


## Requirements
MicroManager (beta) + [mm2python](https://github.com/czbiohub/mm2python)

Python packages: py4j, pandas, numpy, scipy, skimage, sklearn

## Near-term TODOs
- set up CI and formalize tests
- define explicit event kinds (autofocus, scoring, autoexposure, etc) and levels (info, warning, error) for event logging

## Long-term TODOs
- better autoexposure method (using threshold_multiotsu method?)
- image fluorescent beads to measure PSF and chromatic aberration
- develop script to acquire short timelapses (instead of z-stacks)
- add an ability to 'interact' with a running script to change some critical settings:
	- max number of FOVs per well
	- wells to skip
	- whether to stop the script safely

## FOV scoring TODOs
- optimize the dbscan epsilon (the clustering is very sensitive to this)
- organize/clarify all of the empirical parameters and thresholds in FOVScorer
- try a CNN on downsampled FOVs (requires augmentation to enforce rotational invariance)
