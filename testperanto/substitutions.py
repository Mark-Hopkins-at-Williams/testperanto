##
# substitutions.py
# Implementations of tree substitutions.
##


import copy
from testperanto import trees
from abc import ABC, abstractmethod


class Substitution(ABC):
    """Replaces the variables in a tree with custom substitutions.

    Methods
    -------
    add_substitution(variable, sub_value)
        Associates a variable with a replacement value.
    substitute(tree)
        Replaces the variables in the tree with their replacement values.
    """

    def add_substitution(self, variable, sub_value):
        """Associates a variable with a replacement value.

        Parameters
        ----------
        variable : str
            Substitution variable
        sub_value : testperanto.trees.TreeNode
            Desired replacement for the variable
        """

        self.substitution_dict[variable] = sub_value


    @abstractmethod
    def substitute(self, in_tree):
        """Replaces the variables in the tree with their replacement values."""


class LeafSubstitution(Substitution):
    """Replaces the leaves labeled with variables with custom substitutions.

    Custom substitutions are added using the .add_substitution method.
    For instance:
        from testperanto.trees import TreeNode
        sub = LeafSubstitution()
        sub.add_substitution('$x1', TreeNode.from_str('(NP the dog)'))
        sub.add_substitution('$x2', TreeNode.from_str('(VB bit)'))

    Then to apply the substitutions to a tree, we use the .substitute method.
    For instance:
        sub.substitute(TreeNode.from_str('(S (N $x1) (V $x2 $x1))'))

    This returns the following tree:
        (S (N (NP the dog)) (V (VB bit) (NP the dog)))

    Methods
    -------
    add_substitution(variable, sub_value)
        Associates a variable with a replacement value.
    substitute(tree)
        Replaces the leaves labeled with variables with custom substitutions.
    """

    def __init__(self):
        super().__init__()
        self.substitution_dict = {}

    def substitute(self, in_tree):
        """Replaces the variables in the tree with their replacement values.

        Parameters
        ----------
        in_tree : testperanto.trees.TreeNode
            Input tree containing the variables

        Returns
        -------
        testperanto.trees.TreeNode
            A copy of the input tree with the variables replaced by their
            replacement values
        """
        in_tree_first_id = in_tree.get_label()[0]
        if in_tree_first_id in self.substitution_dict:
            return copy.deepcopy( self.substitution_dict[in_tree_first_id] )
        else:
            retval = trees.TreeNode()
            label = in_tree.get_label()
            retval.label = label
            for i in range(in_tree.get_num_children()):
                retval.children.append( self.substitute(in_tree.get_child(i)) )
            return retval


class SymbolSubstitution(Substitution):
    """Replaces the variables in a tree's compound labels with string substitutions.

    Custom substitutions are added using the .add_substitution method.
    For instance:
        sub = SymbolSubstitution()
        sub.add_substitution('$y1', '342')
        sub.add_substitution('$y2', '23')

    Then to apply the substitutions to a tree, we use the .substitute method.
    For instance:
        sub.substitute(TreeNode.from_str('(S.$y1 (N.$y2 the dog) barked)'))

    This returns the following tree:
        (S.342 (N.23 the dog) barked)

    Methods
    -------
    add_substitution(variable, sub_value)
        Associates a variable with a replacement value.
    substitute(tree)
        Replaces the variables in all compound symbols of a tree with custom substitutions.
    substitute_into_compound_symbol(csymbol)
        Replaces the variables in a compound symbol with custom substitutions.
    """

    def __init__(self):
        Substitution.__init__(self)
        self.substitution_dict = {}

    def substitute(self, in_tree):
        """Replaces the variables in all compound symbols of a tree with custom substitutions.

        Parameters
        ----------
        in_tree : testperanto.trees.TreeNode
            Input tree containing the variables

        Returns
        -------
        testperanto.trees.TreeNode
            A copy of the input tree with the variables in compound symbols
            replaced by their replacement values
        """
        retval = trees.TreeNode()
        retval.label = self.substitute_into_compound_symbol(in_tree.get_label())
        for i in range(in_tree.get_num_children()):
            retval.children.append(self.substitute(in_tree.get_child(i)))
        return retval

    def substitute_into_compound_symbol(self, csymbol):
        """Replaces the variables in a compound symbol with custom substitutions.

        Parameters
        ----------
        csymbol : tuple
            Original compound symbol with variables

        Returns
        -------
        tuple
            The input symbol, with the variables replaced with their replacement values
        """
        out_seq = []
        for element in csymbol:
            if element in self.substitution_dict:
                out_seq.append(self.substitution_dict[element])
            else:
                out_seq.append(element)
        return tuple(out_seq)

    def __str__(self):
        return str(self.substitution_dict)
