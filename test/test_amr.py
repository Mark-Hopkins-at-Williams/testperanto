##
# test_trees.py
# Unit tests for trees.py.
##

import unittest
from testperanto.amr import amr_str
from testperanto.trees import TreeNode
from peranto_scripts.generate import main
import random

class TestAmr(unittest.TestCase):

    def test_amr_str1(self):
        tree_str = "(ROOT (inst want-01) (arg0 (inst boy)) (arg1 (inst go-01)))"
        tree = TreeNode.from_str(tree_str)
        expected = "(want-01\n   :arg0 (boy)\n   :arg1 (go-01))"
        self.assertEqual(amr_str(tree), expected)

    def test_amr_str2(self):
        tree_str = "(ROOT (inst obligate-01) (arg1 (inst i)) (arg2 (inst grow-02) (arg1 (inst i)) (arg2 (inst old))))"
        tree = TreeNode.from_str(tree_str)
        expected = "(obligate-01\n   :arg1 (i)\n   :arg2 (grow-02\n      :arg1 (i)\n      :arg2 (old)))"
        self.assertEqual(amr_str(tree), expected)        
    
    def test_amr_str3(self):
        tree_str = "(ROOT (inst look-01) (arg0 (inst i)) (arg2 (inst around) (arg0 (inst i)) (arg3 (inst all))) (arg4 (inst careful)))"
        tree = TreeNode.from_str(tree_str)
        expected = "(look-01\n   :arg0 (i)\n   :arg2 (around\n      :arg0 (i)\n      :arg3 (all))\n   :arg4 (careful))"
        self.assertEqual(amr_str(tree), expected)   


    def test_amr_str4(self):
        #use scripts/generate.py to generate a tree using examples/amr/amr.json rules such that amr_str works properyly
        random.seed(9347)
        tree = main(["examples/amr/amr.json"], any, 1, False, "inactive")
        expected = """(vb.2866597763
   :arg0 (vb.1976741014
      :arg0 (nn.40873566)
      :arg1 (nn.1249596819))
   :arg1 (nn.3483591668))"""
        self.assertEqual(amr_str(tree), expected)        


if __name__ == "__main__":
    unittest.main()   