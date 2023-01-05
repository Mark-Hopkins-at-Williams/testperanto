from testperanto.trees import PositionBasedTree, str_to_position_tree, dfs_sort, TreeNode
from testperanto.util import compound
import sys

def amr_str(tree):
    """This function takes the root node of a tree and prints it out in amr format"""

    def dfs(tree, postorder, count):
        if tree.children:
            for child in tree.children[::-1]:
                if child.get_label()[0] == "VB":
                    postorder.append("(s / " + str(child.children[0]))
                elif child.get_label()[0] == "NN":
                    postorder.append(":arg{} ".format(count) + '(' + str(child.children[0]) + ')')
                    count += 1
                elif child.is_leaf():
                    postorder.append(":placeholder " + '(' + str(child.children[0]) + ')')
                else:
                    dfs(child, postorder, count)

    postorder = []
    dfs(tree, postorder, 0)
    res = ""
    tab = "   "
    tabs = 0
    for item in postorder:
        res += "\n" + tab * tabs + item
        if item[1] == 's':
            tabs += 1
        if item == postorder[-1]:
            res += ')'
    
    return res

def main():
    tree_str = "(S (NN man) (NN dog) (VB bark))"
    tree = TreeNode.from_str(tree_str)
    print(amr_str(tree))
    #print(tree.children)
    
if __name__ == "__main__":
    main()
