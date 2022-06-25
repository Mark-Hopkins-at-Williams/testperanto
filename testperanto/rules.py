##
# rules.py
# Representations and algorithms for tree transducer rules.
##

import json
from abc import ABC, abstractmethod
from copy import deepcopy
from tqdm import tqdm
from testperanto.distributions import CategoricalDistribution
from testperanto.distmanager import DistributionManager
from testperanto.globals import DOT
from testperanto.substitutions import SymbolSubstitution
from testperanto.matchers import LeafMatcher, SymbolMatcher
from testperanto.trees import TreeNode
from testperanto.util import zvar, is_state


class TreeTransducerRule:
    """Implementation of a (extended LHS) tree transducer rule.

    Tree transducer rules have the form
        t1 -> t2
    where t1 and t2 are both testperanto.trees.TreeNode objects.

    The only variables that can be used in the labels are "x-variables",
    which can only be used to label the leaves of the trees. The x-variables
    are used to substitute matched subtrees, as in the following example.
    Suppose we have the tree transducer rule:
        (S (N.23 $x1 $x2) $x3) -> (S (NP.23 $x2 $x3 $x1) $x1)
    If we apply the rule to the input tree:
        (S (N.23 (DT the) (NN dog)) (VBD jumped))
    Then our output tree is:
        (S (NP.23 (NN dog) (VBD jumped) (DT the)) (DT the))

    Methods
    -------
    get_lhs()
        Returns the left-hand side tree.
    get_rhs()
        Returns the right-hand side tree.
    get_weight()
        Returns the weight associated with the rule.
    apply(in_tree):
        Applies the rule to an input tree if it matches the rule's lhs.
    """

    def __init__(self, lhs, rhs, weight):
        """
        Parameters
        ----------
        lhs : testperanto.trees.TreeNode
            t1, where the rule is t1 -> t2
        rhs : testperanto.trees.TreeNode
            t2, where the rule is t1 -> t2
        weight : float
            Weight to associate with the rule
        """

        self.lhs = lhs
        self.rhs = rhs
        self.weight = weight

    def get_lhs(self):
        """Returns the left-hand side tree.

        Returns
        -------
        testperanto.trees.TreeNode
            t1, where the rule is t1 -> t2
        """

        return self.lhs

    def get_rhs(self):
        """Returns the right-hand side tree.

        Returns
        -------
        testperanto.trees.TreeNode
            t2, where the rule is t1 -> t2
        """

        return self.rhs

    def get_weight(self):
        """Returns the weight associated with the rule.

        Returns
        -------
        float
            The rule weight
        """

        return self.weight

    def apply(self, in_tree):
        """Applies the rule to an input tree if it matches the rule's lhs.

        Suppose we have the tree transducer rule:
            (S (N.23 $x1 $x2) $x3) -> (S (NP.23 $x2 $x3 $x1) $x1)
        If we apply the rule to the input tree:
            (S (N.23 (DT the) (NN dog)) (VBD jumped))
        Then our output tree is:
            (S (NP.23 (NN dog) (VBD jumped) (DT the)) (DT the))

        If the rule's lhs does not match the input tree (i.e. we cannot make
        substitutions for the x-variables to produce the input tree), then
        None is returned.

        Parameters
        ----------
        in_tree : testperanto.trees.TreeNode
            The input tree

        Returns
        -------
        testperanto.trees.TreeNode
            A new tree, identical to the rhs tree after making the required x-variable
            substitutions. If no x-variable substitution can match the lhs tree, then
            None is returned.
        """

        rule_lhs_matcher = LeafMatcher(self.lhs)
        leaf_sub = rule_lhs_matcher.match(in_tree)
        return leaf_sub.substitute(self.rhs) if leaf_sub is not None else None

    @staticmethod
    def from_str(rule_str, weight):
        """Constructs a TreeTransducerRule from a string representation.

        The string representation should have the form "t1 -> t2", where t1 and t2
        are tree representations that can be parsed by testperanto.trees.TreeNode.from_str.

        For instance, the following is a valid string representation:
           "(S (N.23 $x1 $x2) $x3) -> (S (NP.23 $x2 $x3 $x1) $x1)"

        Parameters
        ----------
        rule_str : str
            String representation of the tree
        weight : float
            Weight to associate with the rule
        """

        fields = rule_str.split('->')
        lhs_tree_str = fields[0].strip()
        rhs_tree_str = fields[1].strip()
        lhs = TreeNode.from_str(lhs_tree_str)
        rhs = TreeNode.from_str(rhs_tree_str)
        return TreeTransducerRule(lhs, rhs, weight)

    def __str__(self):
        """Overrides the default string method."""
        return str(self.lhs) + ' -> ' + str(self.rhs)


class IndexedTreeTransducerRule:
    """Implementation of an indexed tree transducer rule.

    Indexed tree transducer rules have the form
        t1 -> t2
    where t1 and t2 are both testperanto.trees.TreeNode objects.

    An indexed tree transducer rule is a compact way of expressing a family of tree
    transducer rules, using y- and z-variables. Y-variables appear in the
    compound symbols of the rule lhs, and are used for matching input tree symbols
    and generating rules on the fly. Z-variables appear in the compound symbols of the
    rule rhs, and each is associated with a distribution that generates its value
    upon request.

    For instance, suppose we initialize the following indexed rule:
        irule = IndexedTreeTransducerRule(rule='N.$y1 -> (NP nn.$z1 jj.$y1)',
                                          zdists=[('nn',)],
                                          dist_manager=example_distribution_manager())
    where the example distribution manager associates 'nn' with a distribution
    that alternatively samples the numbers 0 and 100. Then, we can instantiate
    a rule from the indexed rule that matches input tree N.12 as follows:
        irule.choose_rule(TreeNode.from_str('N.12'))
    which returns the rule:
        N.12 -> (NP nn.0 jj.12)
    If we call the method again:
        irule.choose_rule(TreeNode.from_str('N.12'))
    It returns the rule:
        N.12 -> (NP nn.100 jj.12)
    (because the value sampled from distribution 'nn' this time was 100).

    Important:
    - a y-variable should never appear on the rhs but not the lhs
    - a z-variable should never appear on the lhs

    Methods
    -------
    choose_rule(in_tree, recursion_depth=0):
        Expands the indexed rule into a CFG rule by replacing the y- and z- variables.
    get_rule_weight(recursion_depth)
        Returns the indexed rule weight when applied at a particular recursion depth.
    """

    def __init__(self, rule, base_weight=1.0,
                 discount_factor=1.0, zdists=[],
                 dist_manager=DistributionManager()):
        """
        Parameters
        ----------
        rule : str
            String representation of the indexed rule (use the format specified by
            TreeTransducerRule.from_str)
        base_weight : float
            Base weight associated with any rule expanded from the indexed rule
        discount_factor : float
            Multiplicative penalty when the rule is applied at deeper recursion depths
        zdists : list[str]
            Keys associated with the distributions from which to sample the z-variable
            values (starting from $z1)
        dist_manager : testperanto.distmanager.DistributionManager
            Maps distribution keys to distributions.
        """

        self.rule = TreeTransducerRule.from_str(rule, base_weight)
        self.lhs_matcher = SymbolMatcher(self.rule.lhs)
        self.base_weight = base_weight
        self.zdist_keys = zdists
        self.discount_factor = discount_factor
        self.dist_manager = dist_manager

    def get_rule_weight(self, recursion_depth):
        """Returns the weight to associate with rules expanded from this indexed rule.

        Parameters
        ----------
        recursion_depth : int
            Indicates the depth of the derivation tree when this indexed rule is applied

        Returns
        -------
        float
            Base weight of the indexed rule, multiplied by the discount factor raised
            to the power of the recursion depth.
        """

        return (self.discount_factor ** recursion_depth) * self.base_weight

    def choose_rule(self, in_tree, recursion_depth=0):
        """Expands the indexed rule into a transducer rule by replacing the y- and z- variables.

        The y-variables are set by matching the input tree, whereas the z-variables
        are sampled from the provided z-distributions.

        For instance, suppose we initialize the following indexed rule:
            irule = IndexedTreeTransducerRule(rule='N.$y1 -> (NP nn.$z1 jj.$y1)',
                                              zdists=[('nn',)],
                                              dist_manager=example_distribution_manager())
        where the example distribution manager associates 'nn' with a distribution
        that alternatively samples the numbers 0 and 100. Then, we can instantiate
        a rule from the indexed rule that matches input tree N.12 as follows:
            irule.choose_rule(TreeNode.from_str('N.12'))
        which returns the rule:
            N.12 -> (NP nn.0 jj.12)
        If we call the method again:
            irule.choose_rule(TreeNode.from_str('N.12'))
        It returns the rule:
            N.12 -> (NP nn.100 jj.12)
        (because the value sampled from distribution 'nn' this time was 100).

        Parameters
        ----------
        in_tree : testperanto.trees.TreeNode
            Input tree
        recursion_depth : int
            Recursive depth of the derivation when this function is called

        Returns
        -------
        testperanto.rules.TreeTransducerRule
            The expanded rule
        """

        lhs_sub = self.lhs_matcher.match(in_tree)
        if lhs_sub is None:
            return None
        else:
            rule_weight = self.get_rule_weight(recursion_depth)
            rhs_sub = SymbolSubstitution()
            for i in range(len(self.zdist_keys)):
                dist_key = self.zdist_keys[i]
                zdist = self.dist_manager.get(dist_key, lhs_sub)
                rhs_expansion = zdist.sample()
                lhs_sub.add_substitution(zvar(i + 1), rhs_expansion)  # note: start counting vars at 1
                rhs_sub.add_substitution(zvar(i + 1), rhs_expansion)  # note: start counting vars at 1
            rule_lhs = lhs_sub.substitute(self.rule.lhs)
            rule_rhs = rhs_sub.substitute(lhs_sub.substitute(self.rule.rhs))
            return TreeTransducerRule(rule_lhs, rule_rhs, rule_weight)

    def __str__(self):
        return str(self.rule.lhs) + ' -> ' + str(self.rule.rhs)


class IndexedRuleSet:
    """A collection of IndexedTransducerRules.

    Methods
    -------
    get_lhs()
        Returns the left-hand side tree.
    get_rhs()
        Returns the right-hand side tree.
    get_weight()
        Returns the weight associated with the rule.
    apply(in_tree):
        Applies the rule to an input tree if it matches the rule's lhs.
    """

    def __init__(self, irules):
        """
        Parameters
        ----------
        irules : list[testperanto.rules.IndexedTreeTransducerRule]
            The indexed rules to include in the grammar
        """

        self.irules = irules

    def choose_rule(self, in_tree, recursion_depth=0):
        """Samples an applicable rule from the grammar.

        First, the grammar finds all indexed rules that can be applied to the input
        tree, i.e. any indexed rule that returns a non-None value for the call
        irule.choose_rule(in_tree).

        Then, rules are chosen from each applicable indexed rules.

        Finally, one of these rules is chosen in proportion to their associated weights.

        Parameters
        ----------
        in_tree : testperanto.trees.TreeNode
            The input tree
        recursion_depth : int
            Recursive depth of the derivation when this function is called

        Returns
        -------
        testperanto.rules.TreeTransducerRule
            The selected transducer rule to apply to the input tree

        Raises
        ------
        IndexError
            If no indexed rule in the grammar can be applied to the input tree
        """

        successful_matches = []
        for irule in self.irules:
            rule = irule.choose_rule(in_tree, recursion_depth)
            if rule is not None:
                successful_matches.append(rule)
        if len(successful_matches) == 0:
            raise IndexError('no matches for input tree: ' + str(in_tree))
        weights = [rule.get_weight() for rule in successful_matches]
        weights_index = CategoricalDistribution(weights).sample()
        return successful_matches[weights_index]

    @staticmethod
    def from_config(config, manager):
        """Constructs a IndexedRuleSet from a configuration dictionary.

        Parameters
        ----------
        config : dict
            The configuration dictionary
        manager : testperanto.distmanager.DistributionManager
            The distribution manager to associate with the indexed rules.
        """

        irule_configs = []
        if 'macros' in config:
            irule_configs = config['macros']
        irules = []
        for iconfig in irule_configs:
            iconfig['dist_manager'] = manager
            if 'zdists' in iconfig:
                iconfig['zdists'] = [tuple(zdist.split('.')) for zdist in iconfig['zdists']]
            irules.append(IndexedTreeTransducerRule(**iconfig))
        return IndexedRuleSet(irules)

    def __str__(self):
        """Overrides the string method."""
        return '\n'.join([str(m) for m in self.irules])
