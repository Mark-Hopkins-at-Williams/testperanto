import argparse
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from os import walk, listdir
from os.path import join, isdir, isfile
import pandas as pd
import seaborn as sns

def get_perplexity(eval_file):
    with open(eval_file, 'r') as reader:
        for line in reader:
            if 'Perplexity: ' in line:
                fields = line.split("Perplexity: ")
                return float(fields[1])
    return None

def harvest_data(directory):
    subdirs = [f for f in listdir(directory) if isdir(join(directory, f))]
    eval_files = [(subdir, join(join(directory, subdir), 'eval.txt')) for subdir in subdirs
                  if isfile(join(join(directory, subdir), 'eval.txt'))]
    data = []
    columns = ['switches', 'trial', 'perplexity']
    for (key, filename) in eval_files:
        switches, trial = key.split(".")
        perplexity = get_perplexity(filename)
        if perplexity is not None:
            data.append([switches, int(trial), perplexity])
    df = pd.DataFrame(data, columns=columns)
    return df

def viz_data(df):
    sns.set_theme(style="whitegrid")
    sns.set_palette("Spectral")
    sns.pointplot(y="perplexity",
                  x="switches",
                  data=df, orient="v", join=False,
                  order=sorted(set(df['switches'])))
    plt.xticks(rotation=90)
    plt.gcf().subplots_adjust(bottom=0.35)
    plt.show()

def main(directory):
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
    parser = argparse.ArgumentParser(description='Visualize the perplexity of LMs for various typological switches.')
    parser.add_argument('-d', '--dir', required=True,
                        help='experiment directory')
    args = parser.parse_args()
    dataframe = harvest_data(args.dir)
    viz_data(dataframe)
