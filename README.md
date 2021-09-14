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
// start an acquisition in test mode and using the mocked MicroManager API
// (this does not require mm2python or MicroManager)
python ./scripts/run_acquisition.py \
--data-dir ./tests/output/ \
--pml-id PML0000 \
--plate-id P0000 \
--platemap-type none \
--mode test \
--mock-micromanager-api

// start a full-plate acquisition, using mm2python and the real MicroManager API
python ./scripts/run_acquisition.py \
--pml-id PML0123 --plate-id P0021 --platemap-type none
```

## Development
We use black for formatting, flake8 and pylint for linting, pytest for testing, and Github Actions for CI. To setup a new dev env:

1) Create a new Python 3.7 virtualenv: `conda create -n dragonflyenv python=3.7`
2) Clone this repo and install dependencies using pip: `pip install -r requirements.txt`
3) Install the `dragonfly_automation` package: `pip install -e .`
4) Install the pre-commit hooks: `pre-commit install`
5) To run tests locally, use `make test`
