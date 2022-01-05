##
# substitutions.py
# Implementations of tree substitutions.
# $Author: mhopkins $
# $Revision: 32586 $
# $Date: 2012-04-17 14:26:33 -0700 (Tue, 17 Apr 2012) $
##


import copy
from testperanto import trees
from abc import ABC, abstractmethod


class Substitution(ABC):
    @abstractmethod
    def substitute(self, in_tree):
        """Performs the tree substitution on the specified input tree."""


class LeafSubstitution(Substitution):
    
    def __init__(self):
        super().__init__()
        self.substitution_dict = {}

    def add_substitution(self, id, sub_value):
        self.substitution_dict[id] = sub_value
        
    def substitute(self, in_tree):
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
    def __init__(self):
        Substitution.__init__(self)
        self.substitution_dict = {}

    def add_substitution(self, id, sub_value):
        self.substitution_dict[id] = sub_value
                
    def substitute(self, in_tree):
        retval = trees.TreeNode()
        retval.label = self.substitute_in_sequence(in_tree.get_label())
        for i in range(in_tree.get_num_children()):
            retval.children.append( self.substitute(in_tree.get_child(i)) )
        return retval

    def substitute_in_sequence(self, seq):
        out_seq = []
        for element in seq:
            if element in self.substitution_dict:
                out_seq.append( self.substitution_dict[element] )
            else:
                out_seq.append( element )
        return tuple(out_seq)
    
    def to_tuple(self):
        substitution_dict_items = self.substitution_dict.items()
        substitution_dict_items.sort()
        return tuple(substitution_dict_items)

    def __str__(self):
        return str(self.substitution_dict)


class SubstitutionFactory:
    def __init__(self, substitution_type):
        self.substitution_type = substitution_type

    def construct(self):
        if self.substitution_type == 'leaf':
            return LeafSubstitution()
        elif self.substitution_type == 'symbol':
            return SymbolSubstitution()
        else:
            return None

