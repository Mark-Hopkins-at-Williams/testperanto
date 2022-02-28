##
# test_trees.py
# Unit tests for trees.py.
##

import unittest
from testperanto.trees import str_to_position_tree, dfs_sort
from testperanto.trees import TreeNode
from testperanto.util import compound


class TestTrees(unittest.TestCase):

    def test_str_to_position_tree1(self):
        s = '(TOP (S (NP (PRP a)) (VP (AUX b) (RB c) (VB d))))'
        ptree = str_to_position_tree(s)
        self.assertEqual(str(ptree), s)
        expected = {(0, 4): ['TOP', 'S'],
                    (0, 1): ['NP', 'PRP', 'a'],
                    (1, 4): ['VP'],
                    (1, 2): ['AUX', 'b'],
                    (2, 3): ['RB', 'c'],
                    (3, 4): ['VB', 'd']}
        self.assertEqual(ptree.to_spanmap(), expected)

    def test_str_to_position_tree2(self):
        s = '(TOP-1 (S.2 (NP~24 (PRP "a")) (VP (AUX *1) (RB "c") (VB "d"))))'
        ptree = str_to_position_tree(s)
        self.assertEqual(str(ptree), s)
        expected = {(0, 4): ['TOP-1', 'S.2'],
                    (0, 1): ['NP~24', 'PRP', '"a"'],
                    (1, 4): ['VP'],
                    (1, 2): ['AUX', '*1'],
                    (2, 3): ['RB', '"c"'],
                    (3, 4): ['VB', '"d"']}
        self.assertEqual(ptree.to_spanmap(), expected)

    def test_dfs_sort_postorder(self):
        ordered = dfs_sort([(1, 2), (2, 1, 1), (2, 1, 2), (), (1, 2, 1), (1,),
                            (2,), (1, 1), (3,), (1, 2, 2)])
        expected = [(1, 1), (1, 2, 1), (1, 2, 2), (1, 2), (1,), (2, 1, 1),
                    (2, 1, 2), (2,), (3,), ()]
        self.assertEqual(ordered, expected)

    def test_dfs_sort(self):
        ordered = dfs_sort([(1, 2), (2, 1, 1), (2, 1, 2), (), (1, 2, 1), (1,),
                            (2,), (1, 1), (3,), (1, 2, 2)], postorder=False)
        expected = [(), (1,), (1, 1), (1, 2), (1, 2, 1), (1, 2, 2), (2,), (2, 1, 1),
                    (2, 1, 2), (3,)]
        self.assertEqual(ordered, expected)

    def test_tree_node(self):
        tree_str = "(S ({} (DT the) (NN dog)) (VB barked))".format(compound(['NP', '$y1']))
        tree = TreeNode.from_str(tree_str)
        self.assertEqual(str(tree), tree_str)
        self.assertEqual(tree.get_label(), ('S',))
        self.assertEqual(tree.get_child(0).get_label(), ('NP', '$y1'))
        self.assertEqual(tree.get_child(0).get_child(0).get_label(), ('DT',))
        self.assertEqual(tree.get_child(0).get_child(0).get_child(0).get_label(), ('the',))
        self.assertEqual(tree.get_child(0).get_child(1).get_label(), ('NN',))
        self.assertEqual(tree.get_child(0).get_child(1).get_child(0).get_label(), ('dog',))
        self.assertEqual(tree.get_child(1).get_label(), ('VB',))
        self.assertEqual(tree.get_child(1).get_child(0).get_label(), ('barked',))
        self.assertEqual(tree.get_num_children(), 2)
        self.assertEqual(tree.get_child(0).get_num_children(), 2)
        self.assertEqual(tree.get_child(1).get_num_children(), 1)
        self.assertEqual(tree.get_child(1).get_child(0).get_num_children(), 0)
        self.assertEqual(tree.is_leaf(), False)
        self.assertEqual(tree.get_child(0).is_leaf(), False)
        self.assertEqual(tree.get_child(1).is_leaf(), False)
        self.assertEqual(tree.get_child(1).get_child(0).is_leaf(), True)
        tree_str2 = "(S ({} (DT the) (NN cat)) (VB barked))".format(compound(['NP', '$y1']))
        tree_str3 = "(S ({} (DT the) (NN dog)) (VB barked))".format(compound(['NP', '$y2']))
        self.assertEqual(TreeNode.from_str(tree_str)==TreeNode.from_str(tree_str), True)
        self.assertEqual(TreeNode.from_str(tree_str)==TreeNode.from_str(tree_str2), False)
        self.assertEqual(TreeNode.from_str(tree_str)==TreeNode.from_str(tree_str3), False)
        self.assertEqual(tree.get_leaves(),
                         [TreeNode.make("the"), TreeNode.make("dog"), TreeNode.make("barked")])


if __name__ == "__main__":
    unittest.main()   