##
# test_rules.py
# Unit tests for testperanto.rules.
##

import unittest
import testperanto.examples
from testperanto.distmanager import DistributionManager
from testperanto.globals import DOT
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
        rule_str = f'(S (N{DOT}23 $x1 $x2) $x3) -> (S (NP{DOT}23 $x2 $x3 $x1) $x1)'
        rule = TreeTransducerRule.from_str(rule_str, weight=5.0)
        self.assertEqual(rule.get_lhs(), TreeNode.from_str(f'(S (N{DOT}23 $x1 $x2) $x3)'))
        self.assertEqual(rule.get_rhs(), TreeNode.from_str(f'(S (NP{DOT}23 $x2 $x3 $x1) $x1)'))
        self.assertEqual(rule.get_weight(), 5.0)

    def test_rule_application(self):
        rule_str = f'(S (N{DOT}23 $x1 $x2) $x3) -> (S (NP{DOT}23 $x2 $x3 $x1) $x1)'
        rule = TreeTransducerRule.from_str(rule_str, weight=1.0)
        in_tree = TreeNode.from_str(f'(S (N{DOT}23 (DT the) (NN dog)) (VBD jumped))')
        out_tree = rule.apply(in_tree)
        self.assertEqual(str(out_tree), f'(S (NP{DOT}23 (NN dog) (VBD jumped) (DT the)) (DT the))')

    def test_rule_application_failure1(self):
        rule_str = f'(S (N{DOT}24 $x1 $x2) $x3) -> (S (NP{DOT}23 $x2 $x3 $x1) $x1)'
        rule = TreeTransducerRule.from_str(rule_str, weight=1.0)
        in_tree = TreeNode.from_str(f'(S (N{DOT}23 (DT the) (NN dog)) (VBD jumped))')
        self.assertEqual(rule.apply(in_tree), None)

    def test_rule_application_failure2(self):
        rule_str = f'(S (N{DOT}23 $x1 $x2 $x3)) -> (S (NP{DOT}23 $x2 $x3 $x1) $x1)'
        rule = TreeTransducerRule.from_str(rule_str, weight=1.0)
        in_tree = TreeNode.from_str(f'(S (N{DOT}23 (DT the) (NN dog)) (VBD jumped))')
        self.assertEqual(rule.apply(in_tree), None)

    def test_example_macro1(self):
        macro = TreeTransducerRuleMacro(rule=f'N{DOT}$y1 -> (NP nn{DOT}$z1 jj{DOT}$y1)',
                                        zdists=[('nn',)],
                                        dist_manager=example_distribution_manager())
        state1 = TreeNode.from_str(f'N{DOT}12')
        state2 = TreeNode.from_str(f'N{DOT}27')
        self.assertEqual(str(macro.choose_rule(state1)), f'N{DOT}12 -> (NP nn{DOT}0 jj{DOT}12)')
        self.assertEqual(str(macro.choose_rule(state2)), f'N{DOT}27 -> (NP nn{DOT}100 jj{DOT}27)')
        self.assertEqual(str(macro.choose_rule(state2)), f'N{DOT}27 -> (NP nn{DOT}0 jj{DOT}27)')
        self.assertEqual(str(macro.choose_rule(state1)), f'N{DOT}12 -> (NP nn{DOT}100 jj{DOT}12)')

    def test_example_macro2(self):
        macro = TreeTransducerRuleMacro(rule=f'$qnp{DOT}$y1 -> (NP nn{DOT}$z1 jj{DOT}$z2)',
                                        base_weight=0.5,
                                        discount_factor=1.0,
                                        zdists=[('nn', 'rule1', '$y1'), ('jj','$z1', '$y1')],
                                        dist_manager=example_distribution_manager())
        state1 = TreeNode.from_str(f'$qnp{DOT}1')
        state2 = TreeNode.from_str(f'$qnp{DOT}2')
        state3 = TreeNode.from_str(f'$qnp{DOT}3')
        rule = macro.choose_rule(state1)
        self.assertEqual(str(rule), f'$qnp{DOT}1 -> (NP nn{DOT}0 jj{DOT}0)')
        rule = macro.choose_rule(state1)
        self.assertEqual(str(rule), f'$qnp{DOT}1 -> (NP nn{DOT}36 jj{DOT}100)')
        rule = macro.choose_rule(state1)
        self.assertEqual(str(rule), f'$qnp{DOT}1 -> (NP nn{DOT}0 jj{DOT}64)')
        rule = macro.choose_rule(state2)
        self.assertEqual(str(rule), f'$qnp{DOT}2 -> (NP nn{DOT}60 jj{DOT}100)')
        rule = macro.choose_rule(state3)
        self.assertEqual(str(rule), f'$qnp{DOT}3 -> (NP nn{DOT}0 jj{DOT}0)')
        rule = macro.choose_rule(state1)
        self.assertEqual(str(rule), f'$qnp{DOT}1 -> (NP nn{DOT}36 jj{DOT}36)')
        rule = macro.choose_rule(state1)
        self.assertEqual(str(rule), f'$qnp{DOT}1 -> (NP nn{DOT}0 jj{DOT}40)')

    def test_grammar_macro(self):
        manager = DistributionManager()
        manager.add_config(('a',), {'type': 'uniform', 'domain': [1,2]})
        macros = list()
        macros.append(TreeTransducerRuleMacro(f'$qtop{DOT}$y1 -> (TOP $qn{DOT}$z1)', 1.0, 1.0, [('a',)], manager))
        macros.append(TreeTransducerRuleMacro(f'$qn{DOT}$y1 -> (N n{DOT}$y1 $qnprop{DOT}$z1 $qadjunct{DOT}$z2)', 1.0, 1.0, [('a',), ('a',)], manager))
        macros.append(TreeTransducerRuleMacro(f'$qnprop{DOT}$y1 -> (NPROP def plu)', 0.25, 1.0, [], manager))
        macros.append(TreeTransducerRuleMacro(f'$qnprop{DOT}$y1 -> (NPROP def sng)', 0.25, 1.0, [], manager))
        macros.append(TreeTransducerRuleMacro(f'$qnprop{DOT}$y1 -> (NPROP indef plu)', 0.25, 1.0, [], manager))
        macros.append(TreeTransducerRuleMacro(f'$qnprop{DOT}$y1 -> (NPROP indef sng)', 0.25, 1.0, [], manager))
        macros.append(TreeTransducerRuleMacro(f'$qadjunct{DOT}$y1 -> (ADJUNCT $qa{DOT}$z1)', 0.2, 1.0, [('a',)], manager))
        macros.append(TreeTransducerRuleMacro(f'$qadjunct{DOT}$y1 -> (ADJUNCT $qa{DOT}$z1 $qadjunct{DOT}$y1)', 0.8, 0.0, [('a',)], manager))
        macros.append(TreeTransducerRuleMacro(f'$qa{DOT}$y1 -> (A a{DOT}$z1)', 0.5, 1.0, [('a',)], manager))
        macros.append(TreeTransducerRuleMacro(f'$qa{DOT}$y1 -> (A a{DOT}$z1 $qn{DOT}$z2)', 0.5, 0.5, [('a',), ('a',)], manager))
        grammar = RuleMacroSet(macros)
        in_tree = TreeNode.from_str(f'$qnprop{DOT}1')
        nprop_rules = set([str(grammar.choose_rule(in_tree)) for _ in range(100)])
        self.assertEqual(nprop_rules, {f'$qnprop{DOT}1 -> (NPROP def plu)',
                                       f'$qnprop{DOT}1 -> (NPROP def sng)',
                                       f'$qnprop{DOT}1 -> (NPROP indef plu)',
                                       f'$qnprop{DOT}1 -> (NPROP indef sng)'})
        in_tree = TreeNode.from_str(f'$qn{DOT}1')
        n1_rules = set([str(grammar.choose_rule(in_tree)) for _ in range(100)])
        self.assertEqual(n1_rules, {f'$qn{DOT}1 -> (N n{DOT}1 $qnprop{DOT}1 $qadjunct{DOT}1)',
                                    f'$qn{DOT}1 -> (N n{DOT}1 $qnprop{DOT}1 $qadjunct{DOT}2)',
                                    f'$qn{DOT}1 -> (N n{DOT}1 $qnprop{DOT}2 $qadjunct{DOT}1)',
                                    f'$qn{DOT}1 -> (N n{DOT}1 $qnprop{DOT}2 $qadjunct{DOT}2)'})
        in_tree = TreeNode.from_str(f'$qn{DOT}2')
        n2_rules = set([str(grammar.choose_rule(in_tree)) for _ in range(100)])
        self.assertEqual(n2_rules, {f'$qn{DOT}2 -> (N n{DOT}2 $qnprop{DOT}1 $qadjunct{DOT}1)',
                                    f'$qn{DOT}2 -> (N n{DOT}2 $qnprop{DOT}1 $qadjunct{DOT}2)',
                                    f'$qn{DOT}2 -> (N n{DOT}2 $qnprop{DOT}2 $qadjunct{DOT}1)',
                                    f'$qn{DOT}2 -> (N n{DOT}2 $qnprop{DOT}2 $qadjunct{DOT}2)'})

if __name__ == "__main__":
    unittest.main()   