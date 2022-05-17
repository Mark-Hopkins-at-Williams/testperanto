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
    def convert_switch_value(switches):
        conversions = {'000000': 'jpn', '011101': 'eng'}
        return conversions[switches]
    subdirs = [f for f in listdir(directory) if isdir(join(directory, f))]
    eval_files = [(subdir, join(join(directory, subdir), 'eval.txt')) for subdir in subdirs
                  if isfile(join(join(directory, subdir), 'eval.txt'))]
    data = []
    columns = ['word order', 'trial', 'perplexity']
    for (key, filename) in eval_files:
        switches, trial = key.split(".")
        perplexity = get_perplexity(filename)
        if perplexity is not None:
            data.append([convert_switch_value(switches), int(trial), perplexity])
    df = pd.DataFrame(data, columns=columns)
    return df

def harvest_nested_data(directory):
    subdirs = [f for f in listdir(directory) if isdir(join(directory, f))]
    dfs = [harvest_data(join(directory, subdir)) for subdir in subdirs]
    for df in dfs:
        print(df)
    df = pd.concat(dfs, keys=subdirs)
    df = df.reset_index(level=0).rename(columns={'level_0': 'technique'})
    df = df.reset_index(drop=True)
    print(df)
    return df

def viz_data(df):
    sns.set_theme(style="darkgrid")
    sns.set_palette("pastel")
    sns.catplot(x="perplexity",
                hue="word order",
                kind="box",
                data=df, y="technique")
    plt.xticks(rotation=90)
    plt.gcf().subplots_adjust(left=0.2)
    plt.gcf().subplots_adjust(bottom=0.2)

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
    dataframe = harvest_nested_data(args.dir)
    #dataframe = harvest_data(args.dir)
    viz_data(dataframe)
