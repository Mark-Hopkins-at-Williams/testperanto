##
# test_matcher.py
# Unit tests for matcher.py.
# $Revision: 32586 $
# $Date: 2012-04-17 14:26:33 -0700 (Tue, 17 Apr 2012) $
##

import unittest
import sys
from testperanto.matchers import LeafVariableMatcher, RefinementVariableMatcher
from testperanto import trees


class TestMatching(unittest.TestCase):

    def test_leaf_matcher(self):
        pattern_tree_str = '(S (N $x1 $x2) $x3)'
        pattern_tree = trees.construct_node_based_tree_from_string(pattern_tree_str)
        in_tree_str = '(S (N (NP the dog) (VB wags)) (JJ quickly))'
        in_tree = trees.construct_node_based_tree_from_string(in_tree_str)
        matcher = LeafVariableMatcher(pattern_tree)
        sub = matcher.match(in_tree)
        self.assertEqual(str(sub.substitute(pattern_tree)),
                         "(S (N (NP the dog) (VB wags)) (JJ quickly))")
        in_tree_str = '(S (N (NP the dog) (VB wags)) (JJ quickly) (JJS right))'
        in_tree = trees.construct_node_based_tree_from_string(in_tree_str)
        sub = matcher.match(in_tree)
        self.assertEqual(sub, None)

    def test_refinement_matcher(self):
        pattern_tree_str = '(S~$y1 (N~$y2 $x1 $x2) $x3)'
        pattern_tree = trees.construct_node_based_tree_from_string(pattern_tree_str)
        in_tree_str = '(S~32 (N~123 (NP the dog) (VB wags)) (JJ quickly))'
        in_tree = trees.construct_node_based_tree_from_string(in_tree_str)
        matcher = RefinementVariableMatcher(pattern_tree)
        sub = matcher.match(in_tree)
        self.assertEqual(str(sub.substitute(pattern_tree)), "(S~32 (N~123 $x1 $x2) $x3)")
        in_tree_str = '(S~23 (N~231 (NP the dog) (VB wags)) (JJ quickly) (JJS right))'
        in_tree = trees.construct_node_based_tree_from_string(in_tree_str)
        sub = matcher.match(in_tree)
        self.assertEqual(sub, None)


if __name__ == "__main__":
    unittest.main()   