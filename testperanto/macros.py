##
# macros.py
# Representations and algorithms for tree transducer macros.
# $Author: mhopkins $
# $Revision: 34246 $
# $Date: 2012-05-23 18:26:36 -0700 (Wed, 23 May 2012) $
##

import json
from abc import ABC, abstractmethod

from testperanto import trees
from testperanto import distfactories
from testperanto.distfactories import DistributionManager
from testperanto.substitutions import SymbolSubstitution
from testperanto.matchers import LeafVariableMatcher
from testperanto.matchers import RefinementVariableMatcher
from testperanto.distributions import CategoricalDistribution
from testperanto.voicebox import VoiceboxFactory
from testperanto.trees import TreeNode


def rhs_refinement_var(i):
    return '$z{}'.format(i)


def is_state(label):
    try:
        return label[0][:2] == '$q'
    except Exception:
        return False


class TreeTransducerRule(object):
    def __init__(self, lhs, rhs, weight):
        self.lhs = lhs
        self.rhs = rhs
        self.weight = weight

    def get_lhs(self):
        return self.lhs

    def get_rhs(self):
        return self.rhs

    def get_weight(self):
        return self.weight

    def apply(self, in_tree):
        rule_lhs_matcher = LeafVariableMatcher(self.lhs)
        leaf_sub = rule_lhs_matcher.match(in_tree)
        if leaf_sub is not None:
            return leaf_sub.substitute(self.rhs)
        else:
            return None

    @staticmethod
    def construct_from_str(rule_str, weight):
        fields = rule_str.split('->')
        lhs_tree_str = fields[0].strip()
        rhs_tree_str = fields[1].strip()
        lhs = trees.construct_node_based_tree_from_string(lhs_tree_str)
        rhs = trees.construct_node_based_tree_from_string(rhs_tree_str)
        return TreeTransducerRule(lhs, rhs, weight)

    def __str__(self):
        return str(self.lhs) + ' -> ' + str(self.rhs)


class TreeTransducerRuleMacro(object):
    def __init__(self, rule, base_weight=1.0,
                 discount_factor=1.0, zdists=[],
                 dist_manager=DistributionManager()):
        self.rule = TreeTransducerRule.construct_from_str(rule, base_weight)
        self.lhs_matcher = RefinementVariableMatcher(self.rule.lhs)
        self.base_weight = base_weight
        self.zdist_keys = zdists
        self.discount_factor = discount_factor
        self.dist_manager = dist_manager

    def get_rule_weight(self, lhs_sub, recursion_depth):
        return (self.discount_factor ** recursion_depth) * self.base_weight

    def choose_rule(self, in_tree, recursion_depth=0):
        lhs_sub = self.lhs_matcher.match( in_tree )
        if lhs_sub is None:
            return None
        else:
            rule_weight = self.get_rule_weight(lhs_sub, recursion_depth)
            rhs_sub = SymbolSubstitution()
            for i in range(len(self.zdist_keys)):
                dist_key = self.zdist_keys[i]
                zdist = self.dist_manager.get(dist_key, lhs_sub)
                rhs_expansion = zdist.sample()
                lhs_sub.add_substitution(rhs_refinement_var(i + 1), rhs_expansion)  # note: start counting vars at 1
                rhs_sub.add_substitution(rhs_refinement_var(i + 1), rhs_expansion)  # note: start counting vars at 1
            rule_lhs = lhs_sub.substitute( self.rule.lhs )
            rule_rhs = rhs_sub.substitute( lhs_sub.substitute( self.rule.rhs ) )
            return TreeTransducerRule(rule_lhs, rule_rhs, rule_weight)

    def __str__(self):
        return str(self.rule.lhs) + ' -> ' + str(self.rule.rhs)

    
class MacroGrammar:
    def __init__(self, macros):
        self.macros = macros
    
    def add_macro(self, macro):
        self.macros.append(macro)

    def choose_rule(self, in_tree, recursion_depth=0):            
        successful_matches = []
        for macro in self.macros:
            rule = macro.choose_rule( in_tree, recursion_depth )
            if rule is not None:
                successful_matches.append( rule )
        if len(successful_matches) == 0:
            raise IndexError( 'no matches for input tree: ' + str(in_tree))
        weights = [rule.get_weight() for rule in successful_matches]
        weights_index = CategoricalDistribution(weights).sample()
        return successful_matches[weights_index]
        
    def apply_rule(self, in_tree, recursion_depth):
        rule = self.choose_rule(in_tree, recursion_depth)
        return rule.apply(in_tree)

    @staticmethod
    def from_config(config, manager):
        macro_configs = []
        if 'macros' in config:
            macro_configs = config['macros']
        macros = []
        for mconfig in macro_configs:
            mconfig['dist_manager'] = manager
            if 'zdists' in mconfig:
                mconfig['zdists'] = [tuple(zdist.split('~')) for zdist in mconfig['zdists']]
            macros.append(TreeTransducerRuleMacro(**mconfig))
        return MacroGrammar(macros)

    def __str__(self):
        return '\n'.join([str(m) for m in self.macros])


class TreeTransducer(object):
    def __init__(self, grammar):
        self.grammar = grammar

    def run(self, in_tree):
        return self.transduce(in_tree)

    def transduce(self, in_tree, recursion_depth=0):
        if is_state(in_tree.get_label()):
            retval = self.transduce(self.grammar.apply_rule(in_tree, recursion_depth), recursion_depth + 1)
        else:
            retval = trees.TreeNode()
            retval.label = in_tree.get_label()
            retval.children = []
            for i in range(in_tree.get_num_children()):
                retval.children.append(self.transduce(in_tree.get_child(i), recursion_depth))
        return retval

    @staticmethod
    def from_config(config):
        manager = DistributionManager.from_config(config)
        grammar = MacroGrammar.from_config(config, manager)
        return TreeTransducer(grammar)


def init_transducer_cascade(config_files, code=None):
    cascade = []
    for config_file in config_files:
        with open(config_file, 'r') as reader:
            config = json.load(reader)
            cascade.append(init_switched_grammar(config, code))
    vfactory = VoiceboxFactory()
    vbox = vfactory.create_voicebox("seuss")
    cascade.append(vbox)
    return cascade


def run_transducer_cascade(cascade, start_state='$qstart'):
    in_tree = TreeNode.construct_from_str(start_state)
    for transducer in cascade[:-1]:
        out_tree = transducer.run(in_tree)
        in_tree = TreeNode.construct_from_str('({} {})'.format(start_state, out_tree))
    output = cascade[-1].run(in_tree).get_child(0)
    return output


def init_switched_grammar(config, code):
    rules = []
    for macro in config['macros']:
        next_rule = {key: macro[key] for key in macro}
        if 'alt' in macro and 'switch' in macro and code[macro['switch']] == "1":
            next_rule['rule'] = next_rule['alt']
        next_rule = {key: next_rule[key] for key in next_rule if key not in ['alt', 'switch']}
        rules.append(next_rule)
    config = {"distributions": config["distributions"], "macros": rules}
    return TreeTransducer.from_config(config)

