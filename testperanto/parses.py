##
# parses.py
# Functions for manipulating dependency parses.
# $Author: mhopkins $
# $Revision: 32698 $
# $Date: 2012-04-19 15:36:06 -0700 (Thu, 19 Apr 2012) $
##


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

