##
# test_transducer.py
# Unit tests for testperanto.transducer.
##


import unittest
import sys

from testperanto.distmanager import DistributionManager
from testperanto.rules import TreeTransducerRule, TreeTransducerRuleMacro
from testperanto.rules import RuleMacroSet
from testperanto.transducer import TreeTransducer, run_transducer_cascade
from testperanto.trees import TreeNode


class TestTransducer(unittest.TestCase):

    def test_transducer(self):
        rule1_str = '($qs (S (NP $x1) (VP $x2))) -> (S (V ($qv $x2) ($qv $x2)) (N ($qn $x1)))'
        rule2_str = '($qn he) -> il'
        rule3_str = '($qv knows) -> sait'
        rules = [rule1_str, rule2_str, rule3_str]
        rules = [TreeTransducerRuleMacro(r) for r in rules]
        grammar = RuleMacroSet(rules)
        transducer = TreeTransducer(grammar)
        in_tree_str = '($qs (S (NP he) (VP knows)))'
        in_tree = TreeNode.from_str(in_tree_str)
        out_tree = transducer.run(in_tree)
        self.assertEqual(str(out_tree), "(S (V sait sait) (N il))")

    def test_transducer2(self):
        rule_strs = ['($qnp (NP $x1 $x2)) -> (NP ($qnn $x2) ($qadj $x1))',
                     '($qnn $x1) -> (NN $x1)',
                     '($qadj $x1) -> (ADJ $x1)']
        grammar = RuleMacroSet([TreeTransducerRuleMacro(r) for r in rule_strs])
        transducer = TreeTransducer(grammar)
        out_tree = transducer.run(TreeNode.from_str('($qnp (NP red rum))'))
        self.assertEqual(out_tree, TreeNode.from_str("(NP (NN rum) (ADJ red))"))

    def test_transducer_cascade(self):
        rule_strs0 = ['$qs -> (NP red rum)']
        grammar0 = RuleMacroSet([TreeTransducerRuleMacro(r) for r in rule_strs0])
        transducer0 = TreeTransducer(grammar0)
        rule_strs1 = ['($qs (NP $x1 $x2)) -> (NP ($qnn $x2) ($qadj $x1))',
                      '($qnn $x1) -> (NN $x1)',
                      '($qadj $x1) -> (ADJ $x1)']
        grammar1 = RuleMacroSet([TreeTransducerRuleMacro(r) for r in rule_strs1])
        transducer1 = TreeTransducer(grammar1)
        rule_strs2 = ['($qs (NP $x1 $x2)) -> (NP ($qtok $x1) ($qtok $x2))',
                      '($qtok (ADJ red)) -> (ADJ rouge)',
                      '($qtok (NN rum)) -> (NN rhum)']
        grammar2 = RuleMacroSet([TreeTransducerRuleMacro(r) for r in rule_strs2])
        transducer2 = TreeTransducer(grammar2)
        out_tree = run_transducer_cascade([transducer0, transducer1, transducer2],
                                          start_state="$qs")
        self.assertEqual(out_tree, TreeNode.from_str("(NP (NN rhum) (ADJ rouge))"))


if __name__ == "__main__":
    unittest.main()