##
# trees.py
# Data structures and algorithms for trees.
# $Author: mhopkins $
# $Revision: 32586 $
# $Date: 2012-04-17 14:26:33 -0700 (Tue, 17 Apr 2012) $
##

import sys
import string
import math
import random

class PositionTreeReadError(Exception): pass                

class PositionBasedTree(object):
    """A representation of a tree based on positions.
    
    A position-representation of a tree describes nodes by the paths taken to reach them from the root.
    
    For instance, the tree (S (NP (PRP)) (VP (AUX) (RB) (VB) ) ) can be described by the following positions:
        (): the root
        (1): NP, the first child of the root
        (1,1): PRP, the first child of NP
        (2): VP, the second child of the root
        (2,1): AUX, the first child of VP
        (2,2): RB, the second child of VP
        (2,3): VB, the third child of VP

    Note that the positions can also be associated with labels. This root class is unlabeled, thus .get_label(pos) always returns None.
    Labeled trees can be created as subclasses of PositionBasedTree.
        
    """    

    def __init__(self):
        """Construct a trivial tree."""
        self.positions = set()
        self.labels = {}
        self.position_to_span = {}
        
    def get_root(self):
        """Return the root position (i.e. the empty tuple)."""
        return ()

    def get_parent(self, pos):
        """Return the parent position of the argument.

        Parameters
        ----------
        pos : tuple
            The position of interest.

        Returns
        -------
        tuple:
            The parent position of pos (or None if pos is the root or not in the tree).

        """
        if pos not in self.positions or len(pos)==0:
            return None        
        return pos[:-1]

    def get_children(self,pos):
        """Return a list of the child positions of the argument.

        There is no guarantee on the order of the children.

        Parameters
        ----------
        pos : tuple
            The position of interest.

        Returns
        -------
        list of tuples:
            A list of the child positions of pos (or an empty list if pos is not in the tree).

        """
        def is_child(x):
            return len(x) > 0 and x[:-1] == pos
        retval = filter( is_child, self.positions )
        return sorted(retval)

    def is_leaf(self, pos):
        """Return True iff the argument position is a leaf, i.e. a tree position that has no child positions.

        Parameters
        ----------
        pos : tuple
            The position of interest.

        Returns
        -------
        boolean:
            True if the argument position is a leaf (False if it has children or is not in the tree at all).

        """
        return (tuple(list(pos) + [1]) not in self.positions) and (pos in self.positions)


    def get_leaves(self):
        """Return a list of the leaf positions of the tree.

        There is no guarantee on the order of the leaves.

        Returns
        -------
        list of tuples:
            A list of the leaf positions of the tree.

        """
        return filter( self.is_leaf, self.positions )


    def get_spans(self):
        """Return the set of spans of the tree.
        
        Let [L1, L2, ..., Lk] be the list of leaf positions in DFS order.
        The span of position P is (i,j), where:
            - i+1 is the minimum index such that Li is a descendant position of P (i.e. P is a prefix of Li)
            - j is the maximum index such that Lj is a descendant position of P (i.e. P is a prefix of Lj)

        A span (i,j) is a span of the tree if there exists some tree position P such that (i,j) is the span of P.

        There is no guarantee on the order of the spans in the returned list.

        Returns
        -------
        set of tuples of length 2:
            A list of the spans of the tree.
                
        """
        return set(self.position_to_span.values())


    def compile_spans(self):
        """Compile a map from tree positions to their spans.
        
        Let [L1, L2, ..., Lk] be the list of leaf positions in DFS order.
        The span of position P is (i,j), where:
            - i+1 is the minimum index such that Li is a descendant position of P (i.e. P is a prefix of Li)
            - j is the maximum index such that Lj is a descendant position of P (i.e. P is a prefix of Lj)

        This function returns nothing, but populates the dictionary self.position_to_span.
                        
        """
        def compute_span(spanmap, pos):
            if pos in spanmap:
                return spanmap[pos]
            else:
                spanStart = 1000000
                spanEnd   = 0
                for p in self.get_children(pos):
                    childspan = compute_span(spanmap, p)
                    if childspan[0] < spanStart:
                        spanStart = childspan[0]
                    if childspan[1] > spanEnd:
                        spanEnd = childspan[1]
                spanmap[pos] = (spanStart, spanEnd)
                return (spanStart, spanEnd)    
        for spanpos, leafpos in enumerate(sorted(self.get_leaves())):
            self.position_to_span[leafpos] = (spanpos, spanpos+1)   #compute the leaf spans
        compute_span(self.position_to_span, tuple())  #compute all spans starting from root
 

    def to_spanmap(self):
        """Return a map from the tree spans to their labels.
        
        Let [L1, L2, ..., Lk] be the list of leaf positions in DFS order.
        The span of position P is (i,j), where:
            - i+1 is the minimum index such that Li is a descendant position of P (i.e. P is a prefix of Li)
            - j is the maximum index such that Lj is a descendant position of P (i.e. P is a prefix of Lj)

        A span (i,j) is a span of the tree if there exists some tree position P such that (i,j) is the span of P.

        This function returns a dictionary that maps every tree span (i,j) to a list of the labels of its associated positions (in root-to-leaf order).
                        
        """
        def to_spanmap_rec(spanmap, pos):
            current_span = self.position_to_span[ pos ]
            current_label = self.get_label( pos )
            if spanmap.has_key( current_span ):
                spanmap[current_span].append(current_label)
            else:
                spanmap[current_span] = [ current_label ]
                
            for p in self.get_children(pos):
                to_spanmap_rec(spanmap, p)
            
        spanmap = {}
        current_pos = self.get_root()
        if current_pos in self.positions:
            to_spanmap_rec(spanmap, current_pos)
        return spanmap      

    def get_label(self, pos):
        """Return the label associated with the specified position. Always returns None, since this tree is unlabeled."""
        if pos in self.labels:
            return self.labels[pos]
        else:
            return None

    def __str__(self):
        return self.to_string_rec( () )
    
    def to_string_rec(self, pos):
        retval = ''
        if self.is_leaf(pos):
            retval += str(self.get_label(pos))
        else:
            child_positions = dfs_sort(self.get_children(pos))                
            retval += '(' + str(self.get_label(pos)) 
            for childpos in child_positions:
                retval += ' ' + self.to_string_rec(childpos)
            retval += ')'
        return retval
                

def construct_position_based_tree_from_labels(labels):
    t = PositionBasedTree()
    t.positions = labels.keys()
    t.labels = labels
    t.compile_spans()    
    return t
    

def construct_position_based_tree_from_string(s, token_parser=lambda x: x, cls=PositionBasedTree):
    """Construct a PositionBasedTree based on a string.

    An example would be: '(TOP-1 (S-2 (NP-24 (PRP "a") ) (VP *0 (AUX *1) (RB "c") (VB "d") ) ) )'

    Helpful observations:
        - closing parens must have whitespace separating them

    Returns
    -------
    PositionBasedTree:
        The PositionBasedTree represented by the input string.
            
    Raises
    ------
    PositionTreeReadError:
        If the input string is not well-formed.

    """
    tokens = s.split()
    if len(tokens) == 0:
        raise PositionTreeReadError('Tree string contains too few tokens: ' + s )
    labels = {}
    num_children = []
    while len(tokens) > 0:
        next_token = tokens[0]
        tokens = tokens[1:]
        start_paren = False
        end_paren_count = 0
        if next_token.startswith('('):
            next_label = next_token[1:].strip()
            start_paren = True
        elif next_token.endswith(')'):
            while next_token.endswith(')'):
                next_token = next_token[:-1]
                end_paren_count += 1
            next_label = next_token
        else:
            next_label = next_token.strip()

        if len(num_children) > 0:
            num_children[-1] += 1
        if next_label != '':
            labels[ tuple(num_children) ] = token_parser(next_label)
        if start_paren:
            num_children.append(0)
        for i in range(end_paren_count):
            num_children.pop()
 
    if len(num_children) > 0:
        raise PositionTreeReadError('Reached end of tree string without balanced parens: ' + s )

    t = cls()
    t.positions = labels.keys()
    t.labels = labels
    t.compile_spans()    
    return t


def dfs_sort( positions, postorder=True ):
    """Sort a list of positions in depth-first search order.
    
    If the postorder flag is set to True, then visit children before parents; otherwise visit parents before children.

    Parameters
    ----------
    positions : list
        A list of positions.
    postorder : boolean
        Set True in order to visit children before parents in the DFS search.

    Returns
    -------
    list of tuples:
        The positions of the argument list, sorted in DFS order.

    >>> dfs_sort( [ (1,2), (2,1,1), (2,1,2), (), (1,2,1), (1,), (2,) , (1,1), (3,), (1,2,2) ] )
    [(1, 1), (1, 2, 1), (1, 2, 2), (1, 2), (1,), (2, 1, 1), (2, 1, 2), (2,), (3,), ()]
    >>> dfs_sort( [ (1,2), (2,1,1), (2,1,2), (), (1,2,1), (1,), (2,) , (1,1), (3,), (1,2,2) ], False )
    [(), (1,), (1, 1), (1, 2), (1, 2, 1), (1, 2, 2), (2,), (2, 1, 1), (2, 1, 2), (3,)]

    """
    def get_kth_element(x,k):
        post_constant = 0
        if( postorder ):
            post_constant = float('inf')
        if( k < len(x) ):
            return x[k]
        else:
            return post_constant
    if len(positions) == 0:
        return []
    treedepth = max( map( len, positions ))
    depths = range( treedepth )
    depths.reverse()
    for depth in depths:
        positions = sorted(positions, key=lambda x: get_kth_element(x, depth) )		
    return positions
	
def pretty_print_tree(tree,indent=0):
    if len(tree) == 0:
        return
    (sym,subsym) = tree[0]
    print (' ' * indent) + str(sym) + '-' + str(subsym)
    for subtree in tree[1:]:
        pretty_print_tree(subtree,indent+2)


def string_to_label(s):
    return tuple(s.split('~'))


def label_to_string(label):
    return '~'.join([str(component) for component in label])


class TreeNode(object):

    def __init__(self):
        self.label = None
        self.children = []

    def get_label(self):
        return self.label

    def get_children(self):
        return self.children

    def get_num_children(self):
        return len(self.children)

    def is_leaf(self):
        return len(self.children) == 0

    def get_child(self, index):
        return self.children[index]

    def __str__(self):
        if self.is_leaf():
            retval = label_to_string(self.label)
        else:
            retval = '(' + label_to_string(self.label)
            for child in self.children:
                retval += ' ' + str(child)
            retval += ')'
        return retval

    def __eq__(self, obj):
        if not (obj.get_label() == self.get_label()):
            return False
        if not (obj.get_num_children() == self.get_num_children()):
            return False
        for childindex in range(self.get_num_children()):
            if not (obj.get_child(childindex) == self.get_child(childindex)):
                return False
        return True

    @staticmethod
    def construct_from_str(s):
        return construct_node_based_tree_from_string(s)


def construct_node_based_tree_from_string(init_str):
    postree = construct_position_based_tree_from_string(init_str, string_to_label)
    pos = postree.get_root()
    return construct_node_based_tree_from_string_rec(postree, pos)


def construct_node_based_tree_from_string_rec(postree, pos):
    retval = TreeNode()
    retval.label = postree.get_label(pos)
    retval.children = []
    for childpos in postree.get_children(pos):
        retval.children.append(construct_node_based_tree_from_string_rec(postree, childpos))
    return retval


class TreeVisitor(object):
    def __init__(self):
        pass

    def execute(self, node):
        self.do_node_action(node)
        for i in range(node.get_num_children()):
            self.execute(node.get_child(i))

    def do_node_action(self, node):
        raise NotImplementedError("Implement me.")


class LeafLabelCollector(TreeVisitor):
    def __init__(self):
        super().__init__()
        self.leaf_labels = []

    def do_node_action(self, node):
        if node.is_leaf():
            self.leaf_labels.append(node.get_label())

    def get_leaf_labels(self):
        return self.leaf_labels

