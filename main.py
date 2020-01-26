# ------------------------------------------------------------------------------
#
# Phys 490--Winter 2020
# Assignment 2
# Alexander Cooper
#
# ------------------------------------------------------------------------------

import json, argparse
import numpy as np

if __name__ == "__main__":

    # Get Command Line Arguments
    parser = argparse.ArgumentParser(description="Multi-Label Classifier in PyTorc")
    parser.add_argument("params",metavar="params/param_file_name.json",type=str)
    parser.add_argument('-v', type=int, default=1, metavar='N',
                        help='verbosity (default: 1)')
    parser.add_argument('--res-path', metavar='results',
                        help='path of results')
    args = parser.parse_args()

    # Hyperparameters from json file
    with open(args.params) as paramfile:
        hyper = json.load(paramfile)
        

   