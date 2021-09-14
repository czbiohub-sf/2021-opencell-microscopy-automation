# dragonfly-automation
This is a Python package that contains the microscope-automation scripts used to enable high-throughput microscopy for the OpenCell project. These scripts use [`mm2python`](https://github.com/czbiohub/mm2python) to control the spinning disk confocal microscope (nicknamed 'dragonfly'). They are written against the MicroManager 2.0 beta API.

#### Primary features
* Modular and reusable methods for common tasks (e.g., autofocusing, adjusting exposures, acquiring z-stacks)
* Pre-trained machine-learning models for dynamic field-of-view scoring and/or classification
* Extensive logging of metadata, errors, and MicroManager API calls
* Mocks for the MicroManager APIs to facilitate development and enable testing


## Usage examples
The script is called using a CLI. Some common examples are shown below. See `scripts/run_acquisition.py` for all CLI arguments.
```
// start an acquisition in test mode, using the mocked MicroManager API
// (this does not require mm2python or MicroManager)
python ./scripts/run_acquisition.py \
--data-dir ./tests/output/ \
--pml-id PML0000 \
--plate-id P0000 \
--platemap-type none \
--mock-micromanager-api \
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
python ./scripts/run_acquisition.py --pml-id PML0123 --plate-id P0021 --platemap-type none
```

## Development
We use black for formatting, flake8 and pylint for linting, pytest for testing, and Github Actions for CI. To setup a new dev env (starting from a clean virtualenv):

1) Clone the repo, install dependencies, and install the `dragonfly_automation` package using: `pip install -e .`
2) Install pre-commit hooks using `pre-commit install`
3) To run tests locally, use `make test`


## Near-term TODOs
- migrate to pycromanager and gamma
- add tests for the complex AFC-related logic in score_positions (e.g. mock multiple wells in a row failing)
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
