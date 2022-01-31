import argparse
from os import walk
import numpy as np
from collections import defaultdict

def get_perplexity(eval_file):
    with open(eval_file, 'r') as reader:
        for line in reader:
            if 'Perplexity: ' in line:
                fields = line.split("Perplexity: ")
                return float(fields[1])
    return None

def main(directory):
    from os import listdir
    from os.path import join, isdir, isfile
    subdirs = [f for f in listdir(directory) if isdir(join(directory, f))]
    eval_files = [(subdir, join(join(directory, subdir), 'eval.txt')) for subdir in subdirs
                  if isfile(join(join(directory, subdir), 'eval.txt'))]
    results = defaultdict(list)
    for (key, filename) in eval_files:
        print(key)
        print(filename)
        switches, _ = key.split(".")
        perplexity = get_perplexity(filename)
        if perplexity is not None:
            results[switches].append(perplexity)
    for key in results:
        print('{}: {:.2f}, {:.2f}'.format(key, np.mean(results[key]), np.var(results[key])))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize the behavior of ngram streams.')
    parser.add_argument('-d', '--dir', required=True,
                        help='experiment directory')
    args = parser.parse_args()
    main(args.dir)
