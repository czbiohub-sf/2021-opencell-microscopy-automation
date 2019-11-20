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

from dragonfly_automation.qc.pipeline_plate_qc import PipelinePlateQC


def parse_args():
    '''
    '''
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--exp-root', 
        dest='exp_root', 
        required=True)

    parser.add_argument(
        '--opencell-repo', 
        dest='opencell_repo', 
        required=False)

    parser.add_argument(
        '--inspect', 
        dest='inspect', 
        action='store_true',
        required=False)

    parser.add_argument(
        '--run', 
        dest='run_all', 
        action='store_true',
        required=False)


    parser.set_defaults(inspect=False)
    parser.set_defaults(run_all=False)
    args = parser.parse_args()
    return args


def main():

    args = parse_args()
    qc = PipelinePlateQC(args.exp_root)

    if args.inspect:
        qc.summarize()

    if args.run_all:
        
        print('Plotting FOV counts and scores')
        qc.plot_counts_and_scores(save_plot=True)

        print('Generating z-projections')
        qc.generate_z_projections(args.opencell_repo)

        print('Plotting top FOVs')
        qc.tile_fovs(channel_ind=0, save_plot=True)
        qc.tile_fovs(channel_ind=1, save_plot=True)


if __name__ == '__main__':
    main()
