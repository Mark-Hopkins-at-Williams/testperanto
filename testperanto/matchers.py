##
# matchers.py
# Implementations of tree matchers.
# $Author: mhopkins $
# $Revision: 32586 $
# $Date: 2012-04-17 14:26:33 -0700 (Tue, 17 Apr 2012) $
##


from testperanto.substitutions import SubstitutionFactory

def is_leaf_var(symbol):
    return symbol[:2] == '$x'

def is_lhs_refinement_var(symbol):
    return symbol[:2] == "$y"

class Matcher(object):
    def __init__(self, pattern_tree, substitution_factory):
        self.pattern_tree = pattern_tree
        self.substitution_factory = substitution_factory

    def process_label(self, patt_label, in_label):
        raise NotImplementedError("Implement me.")        

    def match(self, in_tree):
        self.substitution = self.substitution_factory.construct()
        self.match_success = True
        self.match_helper( self.pattern_tree, in_tree )
        if self.match_success:
            return self.substitution
        else:
            return None

    def match_helper(self, patt_tree, in_tree):
        if self.match_success:
            should_process_children = self.process_label( patt_tree.get_label(), in_tree )
            if should_process_children:            
                if patt_tree.get_num_children() == in_tree.get_num_children():
                    for i in range(patt_tree.get_num_children()):
                        self.match_helper(patt_tree.get_child(i), in_tree.get_child(i))
                else:
                    self.match_success = False 
    
class LeafVariableMatcher(Matcher):
    def __init__(self, pattern_tree):
        super().__init__(pattern_tree, SubstitutionFactory('leaf'))

    def process_label(self, patt_label, in_tree):
        should_process_children = True
        if is_leaf_var( patt_label[0] ):
            self.substitution.add_substitution( patt_label[0], in_tree )
            should_process_children = False
        elif patt_label != in_tree.get_label():
            self.match_success = False
        return should_process_children

class RefinementVariableMatcher(Matcher):
    def __init__(self, pattern_tree):
        super().__init__(pattern_tree, SubstitutionFactory('symbol'))

    def process_label(self, patt_label, in_tree):
        should_process_children = True
        in_tree_label = in_tree.get_label()
        if is_leaf_var( patt_label[0] ):
            should_process_children = False
        elif len(patt_label) != len(in_tree_label):
            self.match_success = False
        else:
            for i in range(len(patt_label)):
                if is_lhs_refinement_var( patt_label[i] ):
                    self.substitution.add_substitution( patt_label[i], in_tree_label[i] )
                elif patt_label[i] != in_tree_label[i]:
                    self.match_success = False                    
        return should_process_children


