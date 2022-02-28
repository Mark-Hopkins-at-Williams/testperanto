##
# test_substitutions.py
# Unit tests for substitutions.py.
##

import unittest
import sys
from testperanto.substitutions import LeafSubstitution, SymbolSubstitution
from testperanto.trees import TreeNode
from testperanto.util import compound


class TestSubstitution(unittest.TestCase):

    def test_leaf_substitution(self):
        treestr = '(S (N $x1 $x2) $x1)'
        vartree = TreeNode.from_str(treestr)
        subtreestr1 = '(NP the dog)'
        subtree1 = TreeNode.from_str(subtreestr1)
        subtreestr2 = '(VB bit)'
        subtree2 = TreeNode.from_str(subtreestr2)
        sub = LeafSubstitution()
        sub.add_substitution('$x1', subtree1)
        sub.add_substitution('$x2', subtree2)
        self.assertEqual(str(sub.substitute(vartree)),
                         "(S (N (NP the dog) (VB bit)) (NP the dog))")

    def test_symbol_substitution(self):
        treestr = f"({compound(['S', '$y1'])} ({compound(['N', '$y2'])} $x1 $x2) $x1)"
        vartree = TreeNode.from_str(treestr)
        sub = SymbolSubstitution()
        sub.add_substitution('$y1', '342')
        sub.add_substitution('$y2', '23')
        self.assertEqual(str(sub.substitute(vartree)),
                         f"({compound(['S', '342'])} ({compound(['N', '23'])} $x1 $x2) $x1)")
        self.assertEqual(sub.substitute_into_compound_symbol(('N', '$y1', "3", '$y2')),
                         ('N', '342', "3", '23'))


if __name__ == "__main__":
    unittest.main()   