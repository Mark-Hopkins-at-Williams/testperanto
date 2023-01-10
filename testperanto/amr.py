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
        return indent + tree.get_label()    
    child0 = tree.get_child(0)
    assert child0.get_simple_label() == "inst"
    inst0 = child0.get_child(0).get_simple_label()
    res = f"({inst0}"
    for child in tree.get_children()[1:]:
        res += amr_str(child, indent + " " * 3)
    res += ")"
    prefix = f"\n{indent}:{tree.get_simple_label()} " if tree.get_simple_label() != "ROOT" else ""  
    return prefix + res
