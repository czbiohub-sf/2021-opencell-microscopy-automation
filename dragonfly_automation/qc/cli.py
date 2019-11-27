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

    # CLI args whose presence in the command sets them to True
    action_arg_names = ['inspect', 'project', 'plot', 'run_all']

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
    qc = PipelinePlateQC(args.root_dir)

    if args.inspect:
        qc.summarize()

    if args.project:
        qc.generate_z_projections()

    if args.plot:
        qc.plot_counts_and_scores(save_plot=True)
        qc.tile_acquired_fovs(channel_ind=0, save_plot=True)
        qc.tile_acquired_fovs(channel_ind=1, save_plot=True)

    if args.run_all:

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
