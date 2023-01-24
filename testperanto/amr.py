from testperanto.trees import TreeNode
from collections import defaultdict

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
            grandchildren = [g.get_simple_label() for g in child.get_children()]
            statistics[tuple(grandchildren)] += 1
        elif child.get_num_children() > 0:
            get_statistics(child, statistics)
    return statistics


def text_stats(path):
    """Takes an input text file path and returns a list of treeNodeAMRS"""
    with open(path, "r", errors = "ignore") as text_file: 
        treeNodes = file_parse(text_file.read())
    
    #for tree in treeNodes[:5]:
    #    print(tree)

    statistics = defaultdict(int)
    #print(treeNodes)
    for tree in treeNodes:
        # print(tree)
        get_statistics(tree, statistics)
    # return treeNodes
    return statistics

def file_parse(data):
    """ Takes in data as a string and parses it into trees based on amr convention """
    strings = data.split("\n\n")

    treeNodes = []
    for s in strings:
        parse = amr_parse(s)
        if parse is not None:
            treeNodes.append(parse)

    return treeNodes

            

    


