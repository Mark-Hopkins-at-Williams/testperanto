import argparse
from os import walk
import numpy as np
from collections import defaultdict


def all_switching_codes(k):
    if k == 1:
        return ["0", "1"]
    else:
        suffixes = all_switching_codes(k-1)
        return ["0" + suffix for suffix in suffixes] + ["1" + suffix for suffix in suffixes]


def main(config_file, num_trials, work_dir):
    print('#!/bin/bash')
    for code in all_switching_codes(6):
        for trial in range(1, num_trials+1):
            print("bash experiment.sh {} {} {}/{}.{}".format(config_file, code, work_dir, code, trial))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize the behavior of ngram streams.')
    parser.add_argument('-c', '--config', required=True,
                        help='configuration file')
    parser.add_argument('-n', '--num_trials', type=int, required=True,
                        help='number of experiment trials')
    parser.add_argument('-w', '--work_dir', required=True,
                        help='working directory')
    args = parser.parse_args()
    main(args.config, args.num_trials, args.work_dir)
