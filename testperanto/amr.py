from testperanto.trees import PositionBasedTree, str_to_position_tree, dfs_sort, TreeNode
from testperanto.util import compound
import sys

def amr_str(tree, indent=""):
    """This function takes the root node of a tree and prints it out in amr format"""

    if len(tree.get_children()) == 0:
        return indent + tree.get_label()
    
    child0 = tree.get_child(0)
    assert child0.get_simple_label() == "instance"
    instance0 = child0.get_child(0).get_simple_label()
    res = f"({instance0}"
    for child in tree.get_children()[1:]:
        res += amr_str(child, indent + " " * 3)
    res += ")"

    prefix = f"\n{indent}:{tree.get_simple_label()} " if tree.get_simple_label() != "ROOT" else ""  
    return prefix + res

def main():
    tree_str = "(ROOT (instance want-01) (arg0 (instance boy)) (arg1 (instance go-01)))"
    tree = TreeNode.from_str(tree_str)
    print(amr_str(tree))
    #print(tree.children)
    
if __name__ == "__main__":
    main()