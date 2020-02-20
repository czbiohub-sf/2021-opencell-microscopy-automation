import os
import re
import sys
import glob
import json
import shutil
import datetime
import tifffile
import argparse
import numpy as np
import pandas as pd

HERE = os.path.abspath(os.path.dirname(__file__))
REPO_ROOT = os.path.join(HERE, os.pardir)

sys.path.insert(0, REPO_ROOT)
from dragonfly_automation.fov_models import PipelineFOVScorer
from dragonfly_automation.acquisitions.pipeline_plate_acquisition import PipelinePlateAcquisition


def parse_args():
    '''
    '''
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--data-dir',
        dest='data_dir',
        type=str,
        required=False,
        default=os.path.join('D:', 'PipelineML', 'data'))

    parser.add_argument('--pml-id', dest='pml_id', type=str, required=True)
    parser.add_argument('--plate-id', dest='plate_id', type=str, required=True)
    parser.add_argument('--platemap-type', dest='platemap_type', type=str, required=True)

    # environment type - 'dev' or 'prod' - whether to mock the microscope
    parser.add_argument('--env', dest='env', type=str, default='prod', required=False)

    # test mode when env='dev' (how to mock the microscope)
    parser.add_argument('--test-mode', type=str, default=None, required=False)

    # run mode: 'test' or 'prod'
    parser.add_argument('--mode', dest='mode', type=str, default='prod', required=False)

    # CLI args whose presence in the command sets them to True
    action_arg_names = ['acquire_bf_stacks', 'skip_fov_scoring']

    for arg_name in action_arg_names:
        parser.add_argument(
            '--%s' % arg_name.replace('_', '-'), 
            dest=arg_name,
            action='store_true',
            required=False)

    for arg_name in action_arg_names:
        parser.set_defaults(**{arg_name: False})
    
    args = parser.parse_args()
    return args


def main():

    args = parse_args()

    # create a new directory for the acquisition
    dirpath = None
    attempt_count = 0
    while dirpath is None or os.path.isdir(dirpath):
        dirpath = os.path.join(args.data_dir, '%s-%s' % (args.pml_id, attempt_count))
        attempt_count += 1

    fov_scorer = PipelineFOVScorer(mode='prediction')
    fov_scorer.load(os.path.join(REPO_ROOT, 'models', '2019-10-08'))
    fov_scorer.train()
    fov_scorer.validate()

    aq = PipelinePlateAcquisition(
        args.data_dir, 
        fov_scorer, 
        env=args.env,
        test_mode=args.test_mode,
        pml_id=args.pml_id,
        plate_id=args.plate_id,
        platemap_type=args.platemap_type,
        acquire_bf_stacks=args.acquire_bf_stacks,
        skip_fov_scoring=args.skip_fov_scoring,
        attempt_count=attempt_count)

    aq.setup()
    aq.run(mode=args.mode)


if __name__ == '__main__':
    main()
