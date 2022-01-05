##
# plotting.py
# Generates corpus statistics that can be plotted using matplotlib.
# $Author: mhopkins $
# $Revision: 32698 $
# $Date: 2012-04-19 15:36:06 -0700 (Thu, 19 Apr 2012) $
##

import matplotlib.pyplot as plt

powers_of_2 = [2**x for x in range(7, 30)]

def type_counts(token_stream, x_values):
    """ Tracks the number of types in a token stream. """
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
    """ Tracks the fraction of types that appear only once in the stream. """
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


def plot_statistic(stat_fn, token_stream, x_values):
    x_vals, y_vals = stat_fn(token_stream, x_values)
    plt.semilogx(x_vals, y_vals, 'r')
    plt.savefig('foo.png')

