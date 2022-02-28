##
# test_parses.py
# Unit tests for parses.py.
##

import unittest
from testperanto.parses import get_dependencies
from testperanto.trees import TreeNode

class TestParses(unittest.TestCase):

    def test_get_dependencies1(self):
        tree = TreeNode.from_str("(NP (head (NN bottle)) (amod (ADJ blue)))")
        expected = set([('blue', 'amod', 'bottle')])
        self.assertEqual(set(get_dependencies(tree)), expected)

    def test_get_dependencies2(self):
        tree = TreeNode.from_str("(S (nsubj (NN dogs))" +
                                 "   (head (VP (head (VB chased)) " +
                                 "             (dobj (NP (amod (ADJ concerned)) " +
                                 "                       (head (NN cats)))))))")
        expected = set([('concerned', 'amod', 'cats'),
                        ('cats', 'dobj', 'chased'),
                        ('dogs', 'nsubj', 'chased')])
        self.assertEqual(set(get_dependencies(tree)), expected)



if __name__ == "__main__":
    unittest.main()   