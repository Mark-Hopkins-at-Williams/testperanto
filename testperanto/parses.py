##
# parses.py
# Functions for manipulating dependency parses.
##


from testperanto.globals import EMPTY_STR


def get_dependencies(tree):
    """Extracts the dependency relationships of a dependency tree.

    It is assumed that the dependency tree is structured as in the following example:
        (S  (nsubj (NN dogs))
            (head  (VP  (head (VB chased))
                        (dobj (NP   (amod (ADJ concerned))
                                    (head (NN cats)))))))

    Each nonterminal node has exactly one child labeled as the "head". The
    dependencies are the sibling-head relationships, and are represented as triples
    of the form (sibling, dependency_relation, head). For instance, the dependencies
    for the above dependency tree are:
        [('concerned', 'amod', 'cats'),
         ('cats', 'dobj', 'chased'),
         ('dogs', 'nsubj', 'chased')])

    Parameters
    ----------
    tree : testperanto.trees.TreeNode
        The dependency tree

    Returns
    -------
    list[tuple]
        The dependency triples in the input tree
    """

    def get_head(node):
        if node.get_num_children() == 0:
            return ' '.join(node.get_label())
        elif node.get_num_children() == 1 and node.get_child(0).get_num_children() == 0:
            return get_head(node.get_child(0))
        else:
            for c in node.get_children():
                if c.get_label()[0] == 'head':
                    return get_head(c.get_child(0))
            raise Exception('head not found: {}'.format(node))

    def get_child_heads(node):
        if node.get_num_children() == 0:
            return ' '.join(node.get_label())
        else:
            retval = []
            for c in node.get_children():
                deprel = c.get_label()[0]
                if deprel != 'head':
                    retval += [(deprel, get_head(c.get_child(0)))]
            return retval

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
        return result + [(dependent, deprel, head) for deprel, dependent in deprels if dependent != EMPTY_STR]

