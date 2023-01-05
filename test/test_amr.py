##
# test_trees.py
# Unit tests for trees.py.
##

import unittest
from testperanto.amr import amr_str
from testperanto.trees import TreeNode


class TestAmr(unittest.TestCase):



    def test_amr_str1(self):
        tree_str = "(ROOT (instance want-01) (arg0 (instance boy)) (arg1 (instance go-01)))"
        tree = TreeNode.from_str(tree_str)
        expected = "(want-01\n   :arg0 (boy)\n   :arg1 (go-01))"
        self.assertEqual(amr_str(tree), expected)

    def test_amr_str2(self):
        tree_str = "(ROOT (instance obligate-01) (arg1 (instance i)) (arg2 (instance grow-02) (arg1 (instance i)) (arg2 (instance old))))"
        tree = TreeNode.from_str(tree_str)
        expected = "(obligate-01\n   :arg1 (i)\n   :arg2 (grow-02\n      :arg1 (i)\n      :arg2 (old)))"
        self.assertEqual(amr_str(tree), expected)        


if __name__ == "__main__":
    unittest.main()   