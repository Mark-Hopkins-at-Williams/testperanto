from testperanto.trees import TreeNode
from collections import defaultdict
import matplotlib.pyplot as plt

def amr_str(tree, indent=""):
    """Takes the root node of a tree and prints it out in amr format.
    
    Parameters
    ----------
    tree : testperanto.trees.TreeNode
        the input tree
    indent : str
        the base number of spaces to indent each line

    Returns
    -------
    str
        a Penman-style formatting of the tree    
    """
    if len(tree.get_children()) == 0:
        return indent + tree.get_simple_label()
    child0 = tree.get_child(0)
    assert child0.get_simple_label() == "inst", f"tree: {tree}"
    inst0 = child0.get_child(0).get_simple_label()
    res = f"({inst0}"
    for child in tree.get_children()[1:]:
        if child.get_simple_label() == "mods":
            if child.get_child(0).get_simple_label() != "-null-":
                for grandchild in child.get_children():
                    recursive = amr_str(grandchild.get_child(0), indent + " " * 3)
                    my_indent = indent + " " * 3
                    res += f"\n{my_indent}:{grandchild.get_simple_label()} {recursive}"
        else:
            recursive = amr_str(child.get_child(0), indent + " " * 3)
            my_indent = indent + " " * 3
            res += f"\n{my_indent}:{child.get_simple_label()} {recursive}"
    res += ")"
    return res

def amr_parse(s):
    """Creates a TreeNode for an AMR expressed in the Penman style."""
    tokens = cool_split(s)
    stack = [tok for tok in tokens][::-1]
    root = TreeNode()
    root.label = tuple(["ROOT"])
    node_stack = [root]
    while len(stack) > 0:
        next_tok = stack.pop()
        if next_tok == "(":
            for label in ['X', 'inst']:
                inst_node = TreeNode()
                inst_node.label = tuple([label])
                node_stack[-1].children.append(inst_node)
                node_stack.append(inst_node)
            node = TreeNode()
            label = stack.pop()
            if stack[-1] == "/":
                label = label + stack.pop() + stack.pop()
            node.label = tuple([label])
            node_stack[-1].children.append(node)

        elif next_tok == ")":
            node_stack.pop()
            node_stack.pop()

        elif next_tok[0] == ":":
            node_stack.pop()
            node = TreeNode()
            node.label = tuple([next_tok[1:]])
            node_stack[-1].children.append(node)
            node_stack.append(node)

    return root.get_child(0)


def cool_split(s):
    """Tokenizes the Penman style AMR format."""
    lines = s.split('\n')
    uncommented_lines = '\n'.join([line for line in lines if line[0] != '#'])
    chunks = uncommented_lines.split()
    for chunk in chunks:
        next_token = ""
        for char in chunk:
            if char == "(" or char == ")":
                if len(next_token) > 0:
                    yield next_token 
                    next_token = ""
                yield char
            else:
                next_token += char
        if len(next_token) > 0:
            yield next_token

def get_statistics(tree, statistics):
    """A helper method to extract statistics from a treeNode object"""
    for child in tree.get_children():
        if child.get_simple_label() == 'X' and child.get_num_children() > 0:
            grandchildren = [g.get_simple_label() for g in child.get_children() if g.get_simple_label() != 'inst']
            if grandchildren:
                statistics[tuple(grandchildren)] += 1
        elif child.get_num_children() > 0:
            get_statistics(child, statistics)
    return statistics


def text_stats(path):
    """Takes an input text file path and plots some statistics from the text's AMRS"""
    text_file = open(path, "r")
    data = text_file.read()
    text_file.close()
    strings = data.split("\n\n")

    treeNodes = []
    for s in strings:
        parse = amr_parse(s)
        if parse is not None:
            treeNodes.append(parse)
    
    statistics = defaultdict(int)
    for tree in treeNodes:
        get_statistics(tree, statistics)

    raw_counts = defaultdict(int)
    for key, value in statistics.items():
        for item in key:
            raw_counts[item] += value

    raw_counts = dict(sorted(raw_counts.items(), key=lambda x:x[1], reverse=True))
    raw_counts = {str(k):v for (k, v) in raw_counts.items() if v >= 5}

    stats_by_level = dict(sorted(statistics.items(), key=lambda x:x[1], reverse=True))
    stats_by_level = {str(k):v for (k, v) in stats_by_level.items() if v >= 5}

    fig, (ax1, ax2) = plt.subplots(2)
    fig.suptitle('Little Prince AMR Breakdown')
    ax1.bar(stats_by_level.keys(), stats_by_level.values(), width=.9, color='g')
    ax1.set(xlabel='Sentence Articles', ylabel='count') 
    ax1.set_xticklabels(stats_by_level.keys(), rotation=45, ha='right', size = "x-small", rotation_mode='anchor')

    ax2.set(xlabel='Articles', ylabel='count')
    ax2.bar(raw_counts.keys(), raw_counts.values(), width=.9, color='g')
    ax2.set_xticklabels(raw_counts.keys(), rotation=45, ha='right', size = "x-small", rotation_mode='anchor')
    fig.subplots_adjust(hspace=0.75)

    plt.show()


            

    


