##
# test_rules.py
# Unit tests for testperanto.rules.
##

import unittest
import testperanto.examples
from testperanto.distmanager import DistributionManager
from testperanto.rules import TreeTransducerRule, TreeTransducerRuleMacro
from testperanto.rules import RuleMacroSet
from testperanto.trees import TreeNode


def example_distribution_manager():
    manager = DistributionManager(generate_consecutive_ids=True)
    manager.add_config(('nn',), {'type': 'alternating'})
    manager.add_config(('jj',), {'type': 'alternating'})
    manager.add_config(('nn', 'rule1',), {'type': 'averager'})
    manager.add_config(('nn', 'rule1', '$y1'), {'type': 'averager'})
    manager.add_config(('jj', '$z1'), {'type': 'averager'})
    manager.add_config(('jj', '$z1', '$y1'), {'type': 'averager'})
    return manager


class TestRules(unittest.TestCase):

    def test_rule_lhs_rhs_weight(self):
        rule_str = '(S (N~23 $x1 $x2) $x3) -> (S (NP~23 $x2 $x3 $x1) $x1)'
        rule = TreeTransducerRule.from_str(rule_str, weight=5.0)
        self.assertEqual(rule.get_lhs(), TreeNode.from_str('(S (N~23 $x1 $x2) $x3)'))
        self.assertEqual(rule.get_rhs(), TreeNode.from_str('(S (NP~23 $x2 $x3 $x1) $x1)'))
        self.assertEqual(rule.get_weight(), 5.0)

    def test_rule_application(self):
        rule_str = '(S (N~23 $x1 $x2) $x3) -> (S (NP~23 $x2 $x3 $x1) $x1)'
        rule = TreeTransducerRule.from_str(rule_str, weight=1.0)
        in_tree = TreeNode.from_str('(S (N~23 (DT the) (NN dog)) (VBD jumped))')
        out_tree = rule.apply(in_tree)
        self.assertEqual(str(out_tree), '(S (NP~23 (NN dog) (VBD jumped) (DT the)) (DT the))')

    def test_rule_application_failure1(self):
        rule_str = '(S (N~24 $x1 $x2) $x3) -> (S (NP~23 $x2 $x3 $x1) $x1)'
        rule = TreeTransducerRule.from_str(rule_str, weight=1.0)
        in_tree = TreeNode.from_str('(S (N~23 (DT the) (NN dog)) (VBD jumped))')
        self.assertEqual(rule.apply(in_tree), None)

    def test_rule_application_failure2(self):
        rule_str = '(S (N~23 $x1 $x2 $x3)) -> (S (NP~23 $x2 $x3 $x1) $x1)'
        rule = TreeTransducerRule.from_str(rule_str, weight=1.0)
        in_tree = TreeNode.from_str('(S (N~23 (DT the) (NN dog)) (VBD jumped))')
        self.assertEqual(rule.apply(in_tree), None)

    def test_example_macro1(self):
        macro = TreeTransducerRuleMacro(rule='N~$y1 -> (NP nn~$z1 jj~$y1)',
                                        zdists=[('nn',)],
                                        dist_manager=example_distribution_manager())
        state1 = TreeNode.from_str('N~12')
        state2 = TreeNode.from_str('N~27')
        self.assertEqual(str(macro.choose_rule(state1)), 'N~12 -> (NP nn~0 jj~12)')
        self.assertEqual(str(macro.choose_rule(state2)), 'N~27 -> (NP nn~100 jj~27)')
        self.assertEqual(str(macro.choose_rule(state2)), 'N~27 -> (NP nn~0 jj~27)')
        self.assertEqual(str(macro.choose_rule(state1)), 'N~12 -> (NP nn~100 jj~12)')

    def test_example_macro2(self):
        macro = TreeTransducerRuleMacro(rule='$qnp~$y1 -> (NP nn~$z1 jj~$z2)',
                                        base_weight=0.5,
                                        discount_factor=1.0,
                                        zdists=[('nn', 'rule1', '$y1'), ('jj','$z1', '$y1')],
                                        dist_manager=example_distribution_manager())
        state1 = TreeNode.from_str('$qnp~1')
        state2 = TreeNode.from_str('$qnp~2')
        state3 = TreeNode.from_str('$qnp~3')
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

    def test_grammar_macro(self):
        manager = DistributionManager()
        manager.add_config(('a',), {'type': 'uniform', 'domain': [1,2]})
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
        grammar = RuleMacroSet(macros)
        in_tree = TreeNode.from_str('$qnprop~1')
        nprop_rules = set([str(grammar.choose_rule(in_tree)) for _ in range(100)])
        self.assertEqual(nprop_rules, {'$qnprop~1 -> (NPROP def plu)',
                                       '$qnprop~1 -> (NPROP def sng)',
                                       '$qnprop~1 -> (NPROP indef plu)',
                                       '$qnprop~1 -> (NPROP indef sng)'})
        in_tree = TreeNode.from_str('$qn~1')
        n1_rules = set([str(grammar.choose_rule(in_tree)) for _ in range(100)])
        self.assertEqual(n1_rules, {'$qn~1 -> (N n~1 $qnprop~1 $qadjunct~1)',
                                    '$qn~1 -> (N n~1 $qnprop~1 $qadjunct~2)',
                                    '$qn~1 -> (N n~1 $qnprop~2 $qadjunct~1)',
                                    '$qn~1 -> (N n~1 $qnprop~2 $qadjunct~2)'})
        in_tree = TreeNode.from_str('$qn~2')
        n2_rules = set([str(grammar.choose_rule(in_tree)) for _ in range(100)])
        self.assertEqual(n2_rules, {'$qn~2 -> (N n~2 $qnprop~1 $qadjunct~1)',
                                    '$qn~2 -> (N n~2 $qnprop~1 $qadjunct~2)',
                                    '$qn~2 -> (N n~2 $qnprop~2 $qadjunct~1)',
                                    '$qn~2 -> (N n~2 $qnprop~2 $qadjunct~2)'})

if __name__ == "__main__":
    unittest.main()   