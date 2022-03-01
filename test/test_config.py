##
# test_config.py
# Unit tests for configuring tree transducer and grammar macros.
##


import unittest
import testperanto.examples
from testperanto.config import rewrite_gmacro_config, configure_transducer
from testperanto.config import generate_sentences, init_grammar_macro
from testperanto.distmanager import DistributionManager
from testperanto.globals import DOT, EMPTY_STR
from testperanto.rules import TreeTransducerRule, TreeTransducerRuleMacro
from testperanto.rules import RuleMacroSet
from testperanto.transducer import TreeTransducer
from testperanto.trees import TreeNode
from testperanto.util import compound


GRAMMAR1 = {
    "distributions": [
        {"name": "nn", "type": "pyor", "strength": 500, "discount": 0.5},
        {"name": "adj", "type": "pyor", "strength": 100, "discount": 0.8},
        {"name": f"adj{DOT}$y1", "type": "pyor", "strength": 5, "discount": 0.5}
    ],
    "macros": [
        {"rule": f"$qstart -> $qnp{DOT}$z1", "zdists": ["nn"]},
        {
            "rule": f"$qnp{DOT}$y1 -> (NP (amod $qadj) (head $qnn{DOT}$y1))",
            "alt": f"$qnp{DOT}$y1 -> (NP (head $qnn{DOT}$y1) (amod $qadj))",
            "switch": 0
        },
        {"rule": f"$qnn{DOT}$y1 -> (NN bottle)"},
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


def leaf_string(node):
    leaves = [compound(leaf.get_label()) for leaf in node.get_leaves()]
    leaves = [leaf for leaf in leaves if leaf != EMPTY_STR]
    output = ' '.join(leaves)
    return output


class TestConfig(unittest.TestCase):

    def test_config(self):
        rule1 = {'rule': f'$qtop{DOT}$y1 -> (TOP $qn{DOT}$z1)', 'zdists': ['a']}
        rule2 = {'rule': f'$qn{DOT}$y1 -> (N n{DOT}$y1 $qnprop{DOT}$z1)', 'zdists': ['a']}
        rule3 = {'rule': f'$qnprop{DOT}$y1 -> (NPROP def plu)'}
        rule4 = {'rule': f'$qnprop{DOT}$y1 -> (NPROP def sng)'}
        config = {"distributions": [{"name": "a", "type": 'uniform', "domain": [1, 2, 3]}],
                  "macros": [rule1, rule2, rule3, rule4]}
        transducer = configure_transducer(config)
        in_tree = TreeNode.from_str(f'$qtop{DOT}0')
        out_trees = set([str(transducer.run(in_tree)) for _ in range(500)])
        self.assertEqual(out_trees, {f'(TOP (N n{DOT}1 (NPROP def sng)))',
                                     f'(TOP (N n{DOT}1 (NPROP def plu)))',
                                     f'(TOP (N n{DOT}2 (NPROP def sng)))',
                                     f'(TOP (N n{DOT}2 (NPROP def plu)))',
                                     f'(TOP (N n{DOT}3 (NPROP def sng)))',
                                     f'(TOP (N n{DOT}3 (NPROP def plu)))'})


    def test_grammar_config(self):
        rule1 = {'rule': 'TOP -> S.$z1', 'zdists': ['vb']}
        rule2 = {'rule': 'S.$y1 -> NN.$z1 VB.$y1', 'zdists': ['nn.$y1']}
        rule3 = {'rule': 'NN.$y1 -> (@verbatim noun.$y1)'}
        rule4 = {'rule': 'VB.$y1 -> (@vb (STEM verb.$y1) (COUNT sng) (PERSON 3) (TENSE perfect))'}
        config = {"distributions": [{"name": "vb", "type": 'alternating'},
                                    {'name': 'nn', 'type': 'alternating'},
                                    {"name": f"nn{DOT}$y1", "type": 'averager'}],
                  "grammar": [rule1, rule2, rule3, rule4]}
        expected = {'distributions': [{'name': 'vb', 'type': 'alternating'},
                                      {'name': 'nn', 'type': 'alternating'},
                                      {'name': f'nn{DOT}$y1', 'type': 'averager'}],
                    'macros': [{'rule': f'$qtop -> (X $qs{DOT}$z1)', 'zdists': ['vb']},
                               {'rule': f'$qs{DOT}$y1 -> (X $qnn{DOT}$z1 $qvb{DOT}$y1)', 'zdists': [f'nn{DOT}$y1']},
                               {'rule': f'$qnn{DOT}$y1 -> (X (@verbatim noun{DOT}$y1))'},
                               {'rule': f'$qvb{DOT}$y1 -> (X (@vb (STEM verb{DOT}$y1) (COUNT sng) (PERSON 3) (TENSE perfect)))'}]}
        rewritten = rewrite_gmacro_config(config)
        self.assertEqual(expected, rewrite_gmacro_config(rewritten))
        transducer = configure_transducer(rewritten)
        sents = generate_sentences(transducer, start_state='TOP', num_to_generate=5)
        self.assertEqual(sents[0].split()[0], 'noun.0')
        self.assertEqual(sents[1].split()[0], 'noun.100')
        self.assertEqual(sents[2].split()[0], 'noun.40')
        self.assertEqual(sents[3].split()[0], 'noun.60')
        self.assertEqual(sents[4].split()[0], 'noun.0')
        for i in range(5):
            self.assertEqual(sents[i].split()[1][-4:], 'ized')


    def test_grammar_config2(self):
        rule1 = {'rule': 'TOP -> S.$z1', 'zdists': ['vb']}
        rule2 = {'rule': 'S.$y1 -> NN.$z1 VB.$y1', 'zdists': ['nn.$y1']}
        rule3 = {'rule': 'NN.$y1 -> (@verbatim noun.$y1)'}
        rule4 = {'rule': 'VB.$y1 -> (@verbatim verb.$y1)'}
        config = {"distributions": [{"name": "vb", "type": 'alternating'},
                                    {'name': 'nn', 'type': 'alternating'},
                                    {"name": "nn.$y1", "type": 'averager'}],
                  "grammar": [rule1, rule2, rule3, rule4]}
        transducer = init_grammar_macro(config)
        sents = generate_sentences(transducer, start_state='TOP', num_to_generate=5)
        self.assertEqual(sents[0], "noun.0 verb.0")
        self.assertEqual(sents[1], "noun.100 verb.100")
        self.assertEqual(sents[2], "noun.40 verb.0")
        self.assertEqual(sents[3], "noun.60 verb.100")
        self.assertEqual(sents[4], "noun.0 verb.0")

    def test_switched_grammar1a(self):
        grammar = configure_transducer(GRAMMAR1, "1")
        output = grammar.run(TreeNode.from_str("$qstart"))
        expected = "(NP (head (NN bottle)) (amod (ADJ blue)))"
        self.assertEqual(str(output), expected)

    def test_switched_grammar1b(self):
        grammar = configure_transducer(GRAMMAR1, "0")
        output = grammar.run(TreeNode.from_str("$qstart"))
        expected = "(NP (amod (ADJ blue)) (head (NN bottle)))"
        self.assertEqual(str(output), expected)

    def test_switched_grammar2a(self):
        grammar = configure_transducer(GRAMMAR2, "00")
        output = grammar.run(TreeNode.from_str("$qstart"))
        expected = "(S (nsubj (NN dogs)) (head (VP (head (VB chased)) (dobj (NP (amod (ADJ concerned)) (head (NN cats)))))))"
        self.assertEqual(str(output), expected)
        self.assertEqual(leaf_string(output), "dogs chased concerned cats")

    def test_switched_grammar2b(self):
        grammar = configure_transducer(GRAMMAR2, "10")
        output = grammar.run(TreeNode.from_str("$qstart"))
        expected = "(S (nsubj (NN dogs)) (head (VP (dobj (NP (amod (ADJ concerned)) (head (NN cats)))) (head (VB chased)))))"
        self.assertEqual(str(output), expected)
        self.assertEqual(leaf_string(output), "dogs concerned cats chased")

    def test_switched_grammar2c(self):
        grammar = configure_transducer(GRAMMAR2, "01")
        output = grammar.run(TreeNode.from_str("$qstart"))
        expected = "(S (nsubj (NN dogs)) (head (VP (head (VB chased)) (dobj (NP (head (NN cats)) (amod (ADJ concerned)))))))"
        self.assertEqual(str(output), expected)
        self.assertEqual(leaf_string(output), "dogs chased cats concerned")

    def test_switched_grammar2d(self):
        grammar = configure_transducer(GRAMMAR2, "11")
        output = grammar.run(TreeNode.from_str("$qstart"))
        expected = "(S (nsubj (NN dogs)) (head (VP (dobj (NP (head (NN cats)) (amod (ADJ concerned)))) (head (VB chased)))))"
        self.assertEqual(str(output), expected)
        self.assertEqual(leaf_string(output), "dogs cats concerned chased")


if __name__ == "__main__":
    unittest.main()   