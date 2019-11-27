import os
import re
import sys
import glob
import json
import dask
import shutil
import pickle
import hashlib
import skimage
import datetime
import tifffile
import argparse
import numpy as np
import pandas as pd
import dask.diagnostics

try:
    sys.path.append('/Users/keith.cheveralls/projects/opencell-process')
    from pipeline_process.imaging import image
except ImportError:
    # if we're running in a docker container on ESS
    sys.path.append('/home/projects/opencell-process')
    from pipeline_process.imaging import image

from dragonfly_automation.qc.pipeline_plate_qc import PipelinePlateQC


def parse_args():
    '''
    '''
    parser = argparse.ArgumentParser()

    parser.add_argument('root_dir', type=str)

    parser.add_argument(
        '--inspect', 
        dest='inspect', 
        action='store_true',
        required=False)

    parser.add_argument(
        '--run-all', 
        dest='run_all', 
        action='store_true',
        required=False)

    parser.set_defaults(inspect=False)
    parser.set_defaults(run_all=False)
    args = parser.parse_args()
    return args


def main():

    args = parse_args()
    qc = PipelinePlateQC(args.root_dir)

    if args.inspect:
        qc.summarize()

    if args.run_all:
        qc.summarize()

        print('Plotting FOV counts and scores')
        qc.plot_counts_and_scores(save_plot=True)

        # TODO: file renaming using either half-plate platemap or custom platemap
        print('Generating z-projections')
        qc.generate_z_projections()

        print('Plotting acquired FOVs')
        qc.tile_acquired_fovs(channel_ind=0, save_plot=True)
        qc.tile_acquired_fovs(channel_ind=1, save_plot=True)


if __name__ == '__main__':
    main()
