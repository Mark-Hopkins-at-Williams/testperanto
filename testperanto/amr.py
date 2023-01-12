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

def amr_parse(filepath):
    file = open(filepath, mode='r')
    lines = file.readlines()
    file.close()
    amrs = []
    curr = ""
    for line in lines:
        if line[0] == '#':
            if curr != "":
                amrs.append(curr)
            curr = ""
        else:
            line = line.replace(":", " ").strip()
            curr += " " + line
    return amrs[1:]


