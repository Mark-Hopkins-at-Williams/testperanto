##
# testmacros.py
# Unit tests for testperanto.macros.
# $Author: mhopkins $
# $Revision: 32586 $
# $Date: 2012-04-17 14:26:33 -0700 (Tue, 17 Apr 2012) $
##


import unittest
import sys

from testperanto.distmanager import DistributionManager, UniformDistributionFactory
from testperanto.macros import TreeTransducerRule, TreeTransducerRuleMacro
from testperanto.macros import MacroGrammar, TreeTransducer
from testperanto.trees import construct_node_based_tree_from_string, TreeNode
from testperanto.examples import AlternatingDistributionFactory, AveragerDistributionFactory

class TestMacros(unittest.TestCase):

    def setUp(self):
        self.manager = DistributionManager(generate_consecutive_ids=True)
        self.manager.add_factory(('nn',), AlternatingDistributionFactory(self.manager))
        self.manager.add_factory(('jj',), AlternatingDistributionFactory(self.manager))
        self.manager.add_factory(('nn', 'rule1',), AveragerDistributionFactory(self.manager, ('nn',)))
        self.manager.add_factory(('nn', 'rule1', '$y1'), AveragerDistributionFactory(self.manager, ('nn', 'rule1',)))
        self.manager.add_factory(('jj', '$z1'), AveragerDistributionFactory(self.manager, ('jj',)))
        self.manager.add_factory(('jj', '$z1', '$y1'), AveragerDistributionFactory(self.manager, ('jj', '$z1')))

    def test_rule_application(self):
        rule_str = '(S (N~23 $x1 $x2) $x3) -> (S (NP~23 $x2 $x3 $x1) $x1)'
        rule = TreeTransducerRule.construct_from_str(rule_str, weight=1.0)
        in_tree = TreeNode.construct_from_str('(S (N~23 (DT the) (NN dog)) (VBD jumped))')
        out_tree = rule.apply(in_tree)
        self.assertEqual(str(out_tree), '(S (NP~23 (NN dog) (VBD jumped) (DT the)) (DT the))')

    def test_rule_application_failure1(self):
        rule_str = '(S (N~24 $x1 $x2) $x3) -> (S (NP~23 $x2 $x3 $x1) $x1)'
        rule = TreeTransducerRule.construct_from_str(rule_str, weight=1.0)
        in_tree = TreeNode.construct_from_str('(S (N~23 (DT the) (NN dog)) (VBD jumped))')
        self.assertEqual(rule.apply(in_tree), None)

    def test_rule_application_failure2(self):
        rule_str = '(S (N~23 $x1 $x2 $x3)) -> (S (NP~23 $x2 $x3 $x1) $x1)'
        rule = TreeTransducerRule.construct_from_str(rule_str, weight=1.0)
        in_tree = TreeNode.construct_from_str('(S (N~23 (DT the) (NN dog)) (VBD jumped))')
        self.assertEqual(rule.apply(in_tree), None)

    def test_transducer(self):
        rule1_str = '($qs (S (NP $x1) (VP $x2))) -> (S (V ($qv $x2) ($qv $x2)) (N ($qn $x1)))'
        rule2_str = '($qn he) -> il'
        rule3_str = '($qv knows) -> sait'
        rules = [rule1_str, rule2_str, rule3_str]
        rules = [TreeTransducerRuleMacro(r) for r in rules]
        grammar = MacroGrammar(rules)
        transducer = TreeTransducer(grammar)
        in_tree_str = '($qs (S (NP he) (VP knows)))'
        in_tree = TreeNode.construct_from_str(in_tree_str)
        out_tree = transducer.transduce(in_tree)
        self.assertEqual(str(out_tree), "(S (V sait sait) (N il))")

    def test_example_macros(self):
        macro = TreeTransducerRuleMacro(rule='$qnp~$y1 -> (NP nn~$z1 jj~$z2)',
                                        base_weight=0.5,
                                        discount_factor=1.0,
                                        zdists=[('nn', 'rule1', '$y1'), ('jj','$z1', '$y1')],
                                        dist_manager=self.manager)
        state1 = construct_node_based_tree_from_string('$qnp~1')
        state2 = construct_node_based_tree_from_string('$qnp~2')
        state3 = construct_node_based_tree_from_string('$qnp~3')
        rule = macro.choose_rule(state1)
        self.assertEqual(str(rule), '$qnp~1 -> (NP nn~0 jj~0)')
        rule = macro.choose_rule(state1)
        self.assertEqual(str(rule), '$qnp~1 -> (NP nn~36 jj~100)')
        rule = macro.choose_rule(state1)
        self.assertEqual(str(rule), '$qnp~1 -> (NP nn~0 jj~64)')
        rule = macro.choose_rule(state2)
        self.assertEqual(str(rule), '$qnp~2 -> (NP nn~60 jj~100)')
        rule = macro.choose_rule(state3)
        self.assertEqual(str(rule), '$qnp~3 -> (NP nn~0 jj~0)')
        rule = macro.choose_rule(state1)
        self.assertEqual(str(rule), '$qnp~1 -> (NP nn~36 jj~36)')
        rule = macro.choose_rule(state1)
        self.assertEqual(str(rule), '$qnp~1 -> (NP nn~0 jj~40)')

    def test_macro_grammar(self):
        manager = DistributionManager()
        manager.add_factory(('a',), UniformDistributionFactory([1,2]))
        macros = list()
        macros.append(TreeTransducerRuleMacro('$qtop~$y1 -> (TOP $qn~$z1)', 1.0, 1.0, [('a',)], manager))
        macros.append(TreeTransducerRuleMacro('$qn~$y1 -> (N n~$y1 $qnprop~$z1 $qadjunct~$z2)', 1.0, 1.0, [('a',), ('a',)], manager))
        macros.append(TreeTransducerRuleMacro('$qnprop~$y1 -> (NPROP def plu)', 0.25, 1.0, [], manager))
        macros.append(TreeTransducerRuleMacro('$qnprop~$y1 -> (NPROP def sng)', 0.25, 1.0, [], manager))
        macros.append(TreeTransducerRuleMacro('$qnprop~$y1 -> (NPROP indef plu)', 0.25, 1.0, [], manager))
        macros.append(TreeTransducerRuleMacro('$qnprop~$y1 -> (NPROP indef sng)', 0.25, 1.0, [], manager))
        macros.append(TreeTransducerRuleMacro('$qadjunct~$y1 -> (ADJUNCT $qa~$z1)', 0.2, 1.0, [('a',)], manager))
        macros.append(TreeTransducerRuleMacro('$qadjunct~$y1 -> (ADJUNCT $qa~$z1 $qadjunct~$y1)', 0.8, 0.0, [('a',)], manager))
        macros.append(TreeTransducerRuleMacro('$qa~$y1 -> (A a~$z1)', 0.5, 1.0, [('a',)], manager))
        macros.append(TreeTransducerRuleMacro('$qa~$y1 -> (A a~$z1 $qn~$z2)', 0.5, 0.5, [('a',), ('a',)], manager))
        grammar = MacroGrammar(macros)
        in_tree = TreeNode.construct_from_str('$qnprop~1')
        nprop_rules = set([str(grammar.choose_rule(in_tree)) for _ in range(100)])
        self.assertEqual(nprop_rules, {'$qnprop~1 -> (NPROP def plu)',
                                       '$qnprop~1 -> (NPROP def sng)',
                                       '$qnprop~1 -> (NPROP indef plu)',
                                       '$qnprop~1 -> (NPROP indef sng)'})
        in_tree = TreeNode.construct_from_str('$qn~1')
        n1_rules = set([str(grammar.choose_rule(in_tree)) for _ in range(100)])
        self.assertEqual(n1_rules, {'$qn~1 -> (N n~1 $qnprop~1 $qadjunct~1)',
                                    '$qn~1 -> (N n~1 $qnprop~1 $qadjunct~2)',
                                    '$qn~1 -> (N n~1 $qnprop~2 $qadjunct~1)',
                                    '$qn~1 -> (N n~1 $qnprop~2 $qadjunct~2)'})
        in_tree = TreeNode.construct_from_str('$qn~2')
        n2_rules = set([str(grammar.choose_rule(in_tree)) for _ in range(100)])
        self.assertEqual(n2_rules, {'$qn~2 -> (N n~2 $qnprop~1 $qadjunct~1)',
                                    '$qn~2 -> (N n~2 $qnprop~1 $qadjunct~2)',
                                    '$qn~2 -> (N n~2 $qnprop~2 $qadjunct~1)',
                                    '$qn~2 -> (N n~2 $qnprop~2 $qadjunct~2)'})

    def test_config(self):
        rule1 = {'rule': '$qtop~$y1 -> (TOP $qn~$z1)', 'zdists': ['a']}
        rule2 = {'rule': '$qn~$y1 -> (N n~$y1 $qnprop~$z1)', 'zdists': ['a']}
        rule3 = {'rule': '$qnprop~$y1 -> (NPROP def plu)'}
        rule4 = {'rule': '$qnprop~$y1 -> (NPROP def sng)'}
        config = {"distributions": [{"name": "a", "type": 'uniform', "domain": [1, 2, 3]}],
                  "macros": [rule1, rule2, rule3, rule4]}
        transducer = TreeTransducer.from_config(config)
        in_tree = TreeNode.construct_from_str('$qtop~0')
        out_trees = set([str(transducer.transduce(in_tree)) for _ in range(500)])
        self.assertEqual(out_trees, {'(TOP (N n~1 (NPROP def sng)))',
                                     '(TOP (N n~1 (NPROP def plu)))',
                                     '(TOP (N n~2 (NPROP def sng)))',
                                     '(TOP (N n~2 (NPROP def plu)))',
                                     '(TOP (N n~3 (NPROP def sng)))',
                                     '(TOP (N n~3 (NPROP def plu)))'})

if __name__ == "__main__":
    unittest.main()   