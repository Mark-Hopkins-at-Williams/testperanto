##
# test_typology.py
# Unit tests for typology.py.
# $Revision: 32586 $
# $Date: 2012-04-17 14:26:33 -0700 (Tue, 17 Apr 2012) $
##

import unittest
import sys
from testperanto.typology import all_switching_codes, init_switched_grammar
from testperanto.trees import TreeNode, leaf_string


GRAMMAR1 = {
    "distributions": [
        {"name": "nn", "type": "pyor", "strength": 500, "discount": 0.5},
        {"name": "adj", "type": "pyor", "strength": 100, "discount": 0.8},
        {"name": "adj~$y1", "type": "pyor", "strength": 5, "discount": 0.5}
    ],
    "macros": [
        {"rule": "$qstart -> $qnp~$z1", "zdists": ["nn"]},
        {
            "rule": "$qnp~$y1 -> (NP (amod $qadj) (head $qnn~$y1))",
            "alt": "$qnp~$y1 -> (NP (head $qnn~$y1) (amod $qadj))",
            "switch": 0
        },
        {"rule": "$qnn~$y1 -> (NN bottle)"},
        {"rule": "$qadj -> (ADJ blue)"}
    ]
}

GRAMMAR2 = {
    "distributions": [],
    "macros": [
        {"rule": "$qstart -> $qs"},
        {"rule": "$qs -> (S (nsubj $qsubj) (head $qvp))"},
        {
            "rule": "$qvp -> (VP (head $qvb) (dobj $qnp))",
            "alt": "$qvp -> (VP (dobj $qnp) (head $qvb))",
            "switch": 0
        },
        {
            "rule": "$qnp -> (NP (amod $qadj) (head $qobj))",
            "alt": "$qnp -> (NP (head $qobj) (amod $qadj))",
            "switch": 1
        },
        {"rule": "$qvb -> (VB chased)"},
        {"rule": "$qsubj -> (NN dogs)"},
        {"rule": "$qobj -> (NN cats)"},
        {"rule": "$qadj -> (ADJ concerned)"}
    ]
}


class TestSubstitution(unittest.TestCase):

    def test_all_switching_codes(self):
        expected = ['0000', '0001', '0010', '0011', '0100', '0101', '0110', '0111',
                    '1000', '1001', '1010', '1011', '1100', '1101', '1110', '1111']
        self.assertEqual(all_switching_codes(4), expected)

    def test_switched_grammar1a(self):
        grammar = init_switched_grammar(GRAMMAR1, "1")
        output = grammar.run(TreeNode.construct_from_str("$qstart"))
        expected = "(NP (head (NN bottle)) (amod (ADJ blue)))"
        self.assertEqual(str(output), expected)

    def test_switched_grammar1b(self):
        grammar = init_switched_grammar(GRAMMAR1, "0")
        output = grammar.run(TreeNode.construct_from_str("$qstart"))
        expected = "(NP (amod (ADJ blue)) (head (NN bottle)))"
        self.assertEqual(str(output), expected)

    def test_switched_grammar2a(self):
        grammar = init_switched_grammar(GRAMMAR2, "00")
        output = grammar.run(TreeNode.construct_from_str("$qstart"))
        expected = "(S (nsubj (NN dogs)) (head (VP (head (VB chased)) (dobj (NP (amod (ADJ concerned)) (head (NN cats)))))))"
        self.assertEqual(str(output), expected)
        self.assertEqual(leaf_string(output), "dogs chased concerned cats")

    def test_switched_grammar2b(self):
        grammar = init_switched_grammar(GRAMMAR2, "10")
        output = grammar.run(TreeNode.construct_from_str("$qstart"))
        expected = "(S (nsubj (NN dogs)) (head (VP (dobj (NP (amod (ADJ concerned)) (head (NN cats)))) (head (VB chased)))))"
        self.assertEqual(str(output), expected)
        self.assertEqual(leaf_string(output), "dogs concerned cats chased")

    def test_switched_grammar2c(self):
        grammar = init_switched_grammar(GRAMMAR2, "01")
        output = grammar.run(TreeNode.construct_from_str("$qstart"))
        expected = "(S (nsubj (NN dogs)) (head (VP (head (VB chased)) (dobj (NP (head (NN cats)) (amod (ADJ concerned)))))))"
        self.assertEqual(str(output), expected)
        self.assertEqual(leaf_string(output), "dogs chased cats concerned")

    def test_switched_grammar2d(self):
        grammar = init_switched_grammar(GRAMMAR2, "11")
        output = grammar.run(TreeNode.construct_from_str("$qstart"))
        expected = "(S (nsubj (NN dogs)) (head (VP (dobj (NP (head (NN cats)) (amod (ADJ concerned)))) (head (VB chased)))))"
        self.assertEqual(str(output), expected)
        self.assertEqual(leaf_string(output), "dogs cats concerned chased")


if __name__ == "__main__":
    unittest.main()   