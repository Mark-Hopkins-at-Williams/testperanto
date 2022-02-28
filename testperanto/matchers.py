##
# matchers.py
# Implementations of tree matchers.
##

from abc import ABC, abstractmethod
from testperanto.substitutions import LeafSubstitution, SymbolSubstitution

def is_leaf_var(symbol):
    return symbol[:2] == '$x'

def is_lhs_refinement_var(symbol):
    return symbol[:2] == "$y"

class Matcher(ABC):
    """Matches a pattern tree with a target tree. Abstract class.

    A pattern tree is a tree with variables in its labels. The role of the Matcher
    is twofold:
    1. It checks whether the target tree can be produced by making substitutions for
       the variables of the pattern tree.
    2. If so, create a dictionary that maps the variables to the required substitutions.

    Important: a variable should NOT appear more than once in a pattern tree.

    Methods
    -------
    match(target_tree)
        Returns a dictionary with the required variable substitutions to produce the
        target tree from the Matcher's pattern tree. If this is not possible, the method
        returns None.
    """

    def __init__(self, pattern_tree):
        """
        Parameters
        ----------
        pattern_tree : testperanto.trees.TreeNode
            The tree (with variables in its labels) that we want to match
        """

        self.pattern_tree = pattern_tree
        self.substitution = None
        self.match_success = False

    @abstractmethod
    def init_substitution(self):
        """Initializes a substitution, of the relevant type for the matcher.

        Returns
        -------
        testperanto.substitutions.Substitution
            An empty substitution
        """

        raise NotImplementedError("Implement me.")

    @abstractmethod
    def process_node(self, pattern_tree, target_tree):
        """Checks whether the root nodes of the two trees can be matched. Internal method.

        This is mostly a side-effecting operation that adds to the self.substitution
        dictionary if an appropriate match is found, and sets self.match_success to
        False if the match fails.

        Parameters
        ----------
        pattern_tree : testperanto.trees.TreeNode
            The pattern tree (with variables in its labels)
        target_tree : testperanto.trees.TreeNode
            The target tree that we want to match to the pattern tree

        Returns
        -------
        bool
            True iff we should continue to process the child nodes of these trees.
        """

    def match(self, target_tree):
        """Returns the required variable substitutions to produce the target tree.

        Parameters
        ----------
        target_tree : testperanto.trees.TreeNode
            The tree we want to match with the pattern tree

        Returns
        -------
        testperanto.substitutions.Substitution
            The required variable substitution to produce the target tree from
            the Matcher's pattern tree. If no such substitution exists, then
            the method returns None.
        """

        def match_helper(patt_tree, in_tree):
            if self.match_success:
                should_process_children = self.process_node(patt_tree, in_tree)
                if should_process_children:
                    if patt_tree.get_num_children() == in_tree.get_num_children():
                        for i in range(patt_tree.get_num_children()):
                            match_helper(patt_tree.get_child(i), in_tree.get_child(i))
                    else:
                        self.match_success = False

        self.substitution = self.init_substitution()
        self.match_success = True
        match_helper( self.pattern_tree, target_tree )
        return self.substitution if self.match_success else None


class LeafMatcher(Matcher):
    """Matches a pattern tree with a target tree, focusing on leaf variables.

    A pattern tree is a tree with variables in its labels. The role of the Matcher
    is twofold:
    1. It checks whether the target tree can be produced by making substitutions for
       the variables of the pattern tree.
    2. If so, create a dictionary that maps the variables to the required substitutions.

    The LeafMatcher focuses exclusively on leaves labeled with x-variables. For instance,
    if the pattern tree is
        (S (N $x1 $x2) $x3)
    then calling the .match method on target tree
        (S (N the dog) barked)
    should return the following substitution:
        {'$x1': "the", '$x2': "dog", '$x3': "barked"}

    Important: a variable should NOT appear more than once in a pattern tree.

    Methods
    -------
    match(target_tree)
        Returns the required Substitution to produce the target tree from the Matcher's
        pattern tree. If this is not possible, the method returns None.
    """

    def __init__(self, pattern_tree):
        super().__init__(pattern_tree)

    def init_substitution(self):
        """Initializes a LeafSubstitution."""
        return LeafSubstitution()

    def process_node(self, pattern_tree, target_tree):
        """Checks whether the root nodes of the two trees can be matched. Internal method.

        This is mostly a side-effecting operation that adds to the self.substitution
        dictionary if an appropriate match is found, and sets self.match_success to
        False if the match fails.

        Parameters
        ----------
        pattern_tree : testperanto.trees.TreeNode
            The pattern tree (with variables in its labels)
        target_tree : testperanto.trees.TreeNode
            The target tree that we want to match to the pattern tree

        Returns
        -------
        bool
            True iff we should continue to process the child nodes of these trees.
        """
        should_process_children = True
        patt_label = pattern_tree.get_label()
        if is_leaf_var(patt_label[0]):
            self.substitution.add_substitution(patt_label[0], target_tree)
            should_process_children = False
        elif patt_label != target_tree.get_label():
            self.match_success = False
        return should_process_children


class SymbolMatcher(Matcher):
    """Matches a pattern tree with a target tree, focusing on internal symbol variables.

    A pattern tree is a tree with variables in its labels. The role of the Matcher
    is twofold:
    1. It checks whether the target tree can be produced by making substitutions for
       the variables of the pattern tree.
    2. If so, create a dictionary that maps the variables to the required substitutions.

    The SymbolMatcher focuses exclusively on y-variables in compound symbol labels
    For instance, if the pattern tree is
        (S~$y1 (N~$y2 the dog) barked)
    then calling the .match method on target tree
        (S~52 (N~34 the dog) barked)
    should return the following substitution:
        {'$y1': "52", '$y2': "34"}

    Important: a variable should NOT appear more than once in a pattern tree.

    Methods
    -------
    match(target_tree)
        Returns the required Substitution to produce the target tree from the Matcher's
        pattern tree. If this is not possible, the method returns None.
    """

    def __init__(self, pattern_tree):
        """
        Parameters
        ----------
        pattern_tree : testperanto.trees.TreeNode
            The tree (with variables in its labels) that we want to match
        """

        super().__init__(pattern_tree)

    def init_substitution(self):
        """Initializes a SymbolSubstitution."""
        return SymbolSubstitution()

    def process_node(self, pattern_tree, target_tree):
        """Checks whether the root nodes of the two trees can be matched. Internal method.

        This is mostly a side-effecting operation that adds to the self.substitution
        dictionary if an appropriate match is found, and sets self.match_success to
        False if the match fails.

        Parameters
        ----------
        pattern_tree : testperanto.trees.TreeNode
            The pattern tree (with variables in its labels)
        target_tree : testperanto.trees.TreeNode
            The target tree that we want to match to the pattern tree

        Returns
        -------
        bool
            True iff we should continue to process the child nodes of these trees.
        """

        should_process_children = True
        patt_label = pattern_tree.get_label()
        in_tree_label = target_tree.get_label()
        if is_leaf_var(patt_label[0]):
            should_process_children = False
        elif len(patt_label) != len(in_tree_label):
            self.match_success = False
        else:
            for i in range(len(patt_label)):
                if is_lhs_refinement_var(patt_label[i]):
                    self.substitution.add_substitution(patt_label[i], in_tree_label[i])
                elif patt_label[i] != in_tree_label[i]:
                    self.match_success = False                    
        return should_process_children


