##
# test_matcher.py
# Unit tests for matcher.py.
# $Revision: 32586 $
# $Date: 2012-04-17 14:26:33 -0700 (Tue, 17 Apr 2012) $
##

import unittest
import sys
from testperanto.globals import DOT
from testperanto.matchers import LeafMatcher, SymbolMatcher
from testperanto.trees import TreeNode


class TestMatching(unittest.TestCase):

    def test_leaf_matcher(self):
        pattern_tree_str = '(S (N $x1 $x2) $x3)'
        pattern_tree = TreeNode.from_str(pattern_tree_str)
        in_tree_str = '(S (N (NP the dog) (VB wags)) (JJ quickly))'
        in_tree = TreeNode.from_str(in_tree_str)
        matcher = LeafMatcher(pattern_tree)
        sub = matcher.match(in_tree)
        self.assertEqual(str(sub.substitute(pattern_tree)),
                         "(S (N (NP the dog) (VB wags)) (JJ quickly))")
        in_tree_str = '(S (N (NP the dog) (VB wags)) (JJ quickly) (JJS again))'
        in_tree = TreeNode.from_str(in_tree_str)
        sub = matcher.match(in_tree)
        self.assertEqual(sub, None)

    def test_refinement_matcher(self):
        pattern_tree_str = f'(S{DOT}$y1 (N{DOT}$y2 $x1 $x2) $x3)'
        pattern_tree = TreeNode.from_str(pattern_tree_str)
        in_tree_str = f'(S{DOT}32 (N{DOT}123 (NP the dog) (VB wags)) (JJ quickly))'
        in_tree = TreeNode.from_str(in_tree_str)
        matcher = SymbolMatcher(pattern_tree)
        sub = matcher.match(in_tree)
        self.assertEqual(str(sub.substitute(pattern_tree)), f"(S{DOT}32 (N{DOT}123 $x1 $x2) $x3)")
        in_tree_str = f'(S{DOT}23 (N{DOT}231 (NP the dog) (VB wags)) (JJ quickly) (JJS again))'
        in_tree = TreeNode.from_str(in_tree_str)
        sub = matcher.match(in_tree)
        self.assertEqual(sub, None)
        in_tree_str = f'(S{DOT}32 (N{DOT}123{DOT}5 (NP the dog) (VB wags)) (JJ quickly))'
        in_tree = TreeNode.from_str(in_tree_str)
        sub = matcher.match(in_tree)
        self.assertEqual(sub, None)


if __name__ == "__main__":
    unittest.main()   