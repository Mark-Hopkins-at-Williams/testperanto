##
# analysis.py
# Generates corpus statistics that can be plotted using matplotlib.
##

from collections import defaultdict
from math import log
import matplotlib.pyplot as plt
import pyconll
import sys
from tqdm import tqdm
from testperanto.trees import TreeNode
import pandas as pd
import seaborn as sns

powers_of_2 = [2**x for x in range(7, 30)]
multiples_of_1000 = [1000*k for k in range(1000)]


def type_count_over_time(token_stream, x_values):
    """Tracks the number of types in a token stream.

    Parameters
    ----------
    token_stream : Generator[str]
        A stream of tokens
    x_values : list[int]
        Token counts at which to record the number of types

    Returns
    -------
    list[int], list[int]
        The x-values (token counts) and corresponding y-values (type counts)
    """

    token_set = set()
    token_counter = 0
    x_vals = []
    singleton_cts = []
    for token in token_stream:
        token_counter += 1
        token_set.add(token)
        if token_counter in x_values:
            x_vals.append(token_counter)
            singleton_cts.append(len(token_set))
    return x_vals, singleton_cts


def singleton_proportion(token_stream, x_values):
    """Tracks the fraction of types that appear only once in the stream (singleton proportion).

    Parameters
    ----------
    token_stream : Generator[str]
        A stream of tokens
    x_values : list[int]
        Token counts at which to record the singleton proportion

    Returns
    -------
    list[int], list[int]
        The x-values (token counts) and corresponding y-values (singleton proportion)
    """

    token_counts = dict()
    singleton_token_set = set()
    token_counter = 0
    x_points = []
    y_points = []
    for token in token_stream:
        token_counter += 1
        if token not in token_counts:
            token_counts[token] = 1
            singleton_token_set.add(token)
        else:
            token_counts[token] += 1
            if token in singleton_token_set:
                singleton_token_set.remove(token)
        if token_counter in x_values:
            x_points.append(token_counter)
            y_points.append(1.0 * len(singleton_token_set) / len(token_counts))
    return x_points, y_points


def plot_statistic(stat_fn, corpora, x_values, axes="semilogx",
                   corpus_labels=None, x_label="num tokens", y_label="y"):
    """Plots token statistics using seaborn.

    Parameters
    ----------
    stat_fn : function
        Function for generating the (x,y) values (using interface of singleton_proportion)
    corpora : list[generator[str]]
        A list of token streams, one stream per corpus
    x_values : list[int]
        Token counts at which to record the chosen statistic
    axes : str
        Scale of the axes, either "semilogx" or "loglog"
    corpus_labels : list[str]
        Legend labels for each corpus (should be the same length as corpora)
    x_label : str
        Label for the x-axis
    y_label : str
        Label for the y-axis
    """

    data = []
    if corpus_labels is None:
        corpus_labels = ['corpus{}'.format(i) for i in range(len(corpora))]
    for i, token_stream in tqdm(enumerate(corpora)):
        x_vals, y_vals = stat_fn(corpora, x_values)
        data += [[corpus_labels[i], x,y] for [x,y] in zip(x_vals, y_vals)]
    df = pd.DataFrame(data, columns=['corpus', x_label, y_label])
    sns.set_style("darkgrid")
    ax = sns.lineplot(data=df, x="num tokens", y=y_label, hue="corpus", style="corpus")
    if axes == "semilogx":
        ax.set(xscale='log')
    elif axes == "loglog":
        ax.set(xscale='log')
        ax.set(yscale='log')
    plt.show()


def plot_singleton_proportion(corpora, corpus_labels=None):
    """Plots singleton proportion versus overall token count.

    Parameters
    ----------
    corpora : list[generator[str]]
        A list of token streams, one stream per corpus
    corpus_labels : list[str]
        Legend labels for each corpus (should be the same length as corpora)
    """

    if corpus_labels is None:
        corpus_labels = ['corpus{}'.format(i) for i in range(1, len(corpora) + 1)]
    metric = singleton_proportion
    axes = "semilogx"
    y_label = "singleton proportion"
    plot_statistic(metric, corpora, powers_of_2, axes,
                   corpus_labels=corpus_labels, y_label=y_label)
