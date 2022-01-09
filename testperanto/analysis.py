##
# analysis.py
# Generates corpus statistics that can be plotted using matplotlib.
# $Author: mhopkins $
# $Revision: 32698 $
# $Date: 2012-04-19 15:36:06 -0700 (Thu, 19 Apr 2012) $
##

from collections import defaultdict
from math import log
import matplotlib.pyplot as plt
import pyconll
import sys
from tqdm import tqdm
from testperanto.trees import TreeNode

powers_of_2 = [2**x for x in range(7, 30)]
multiples_of_1000 = [1000*k for k in range(1000)]


def type_count_over_time(token_stream, x_values):
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


def plot_statistic(stat_fn, token_streams, x_values, axes="semilogx"):
    plot_args = []
    colors = ['red', 'blue', 'green', 'yellow', 'orange', 'violet']
    for i, token_stream in enumerate(token_streams):
        print(colors[i])
        x_vals, y_vals = stat_fn(token_stream, x_values)
        plot_args += [x_vals, y_vals, colors[i]]
    if axes == "semilogx":
        plt.semilogx(*plot_args)
    elif axes == "loglog":
        plt.loglog(*plot_args)
    plt.show()


def harvest_dependencies(conllu_file, out_file, desired_deprels):
    reader = pyconll.load_from_file(conllu_file)
    with open(out_file, 'w') as writer:
        for sentence in tqdm(reader):
            heads = [tok.head for tok in sentence]
            deprels = [tok.deprel for tok in sentence]
            words = ['-ROOT-'] + [tok.form for tok in sentence]
            for i in range(len(deprels)):
                if deprels[i] in desired_deprels:
                    dependent, head = words[i+1], words[int(heads[i])]
                    writer.write("{} {}\n".format(dependent.lower(), head.lower()))


def harvest_dependencies_ptb(ptb_file, out_file, desired_deprels):
    with open(out_file, 'w') as writer:
        with open(ptb_file, 'r') as reader:
            for line in tqdm(list(reader)):
                tree = TreeNode.construct_from_str(line)
                desired = [(x,z) for (x,deprel,z) in get_dependencies(tree) if deprel in desired_deprels]
                for (dependent, head) in desired:
                    writer.write("{} {}\n".format(dependent.lower(), head.lower()))


def get_head(tree):
    if tree.get_num_children() == 0:
        return ' '.join(tree.get_label())
    elif tree.get_num_children() == 1 and tree.get_child(0).get_num_children() == 0:
        return get_head(tree.get_child(0))
    else:
        for child in tree.get_children():
            if child.get_label()[0] == 'head':
                return get_head(child.get_child(0))
        raise Exception('head not found: {}'.format(tree))

def get_child_heads(tree):
    if tree.get_num_children() == 0:
        return ' '.join(tree.get_label())
    else:
        result = []
        for child in tree.get_children():
            deprel = child.get_label()[0]
            if deprel != 'head':
                result += [(deprel, get_head(child.get_child(0)))]
        return result

def get_dependencies(tree):
    if tree.get_num_children() == 0:
        return []
    elif tree.get_num_children() == 1 and tree.get_child(0).get_num_children() == 0:
        return []
    else:
        result = []
        for child in tree.get_children():
            result += get_dependencies(child.get_child(0))
        head = get_head(tree)
        deprels = get_child_heads(tree)
        return result + [(dependent, deprel, head) for deprel, dependent in deprels if dependent != 'NULL']

if __name__ == "__main__":
    harvest_dependencies_ptb(sys.argv[1], sys.argv[2], sys.argv[3].split(','))
