##
# test_substitutions.py
# Unit tests for substitutions.py.
# $Revision: 32586 $
# $Date: 2012-04-17 14:26:33 -0700 (Tue, 17 Apr 2012) $
##

import unittest
import sys
from testperanto.substitutions import LeafSubstitution, SymbolSubstitution
from testperanto import trees


class TestSubstitution(unittest.TestCase):

    def test_leaf_substitution(self):
        treestr = '(S (N $x1 $x2) $x1)'
        vartree = trees.construct_node_based_tree_from_string(treestr)
        subtreestr1 = '(NP the dog)'
        subtree1 = trees.construct_node_based_tree_from_string(subtreestr1)
        subtreestr2 = '(VB wags)'
        subtree2 = trees.construct_node_based_tree_from_string(subtreestr2)
        sub = LeafSubstitution()
        sub.add_substitution('$x1', subtree1)
        sub.add_substitution('$x2', subtree2)
        self.assertEqual(str(sub.substitute(vartree)),
                         "(S (N (NP the dog) (VB wags)) (NP the dog))")

    def test_symbol_substitution(self):
        treestr = '(S~$y1 (N~$y2 $x1 $x2) $x1)'
        vartree = trees.construct_node_based_tree_from_string(treestr)
        sub = SymbolSubstitution()
        sub.add_substitution('$y1', '342')
        sub.add_substitution('$y2', '23')
        self.assertEqual(str(sub.substitute(vartree)), "(S~342 (N~23 $x1 $x2) $x1)")


if __name__ == "__main__":
    unittest.main()   