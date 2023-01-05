##
# trees.py
# Data structures and algorithms for trees.
##

import sys
import string
import math
import random
from testperanto.globals import EMPTY_STR, COMPOUND_SEP
from testperanto.util import compound

class PositionTreeReadError(Exception):
    """Raised if there is an issue with initializing a PositionTree."""


class PositionBasedTree(object):
    """A representation of a tree based on positions.
    
    A position-representation of a tree describes nodes by the paths taken to reach them
    from the root.
    
    For instance, the tree (S (NP PRP) (VP AUX RB VB)) can be described by the
    following positions:
        (): the root
        (1): NP, the first child of the root
        (1,1): PRP, the first child of NP
        (2): VP, the second child of the root
        (2,1): AUX, the first child of VP
        (2,2): RB, the second child of VP
        (2,3): VB, the third child of VP

    Note that the positions can also be associated with labels. This root class is
    unlabeled, thus .get_label(pos) always returns None. Labeled trees can be created
    as subclasses of PositionBasedTree.

    Methods
    -------
    get_root()
        Returns the root position (i.e. the empty tuple).
    get_parent(pos):
        Returns the parent position of the specified position (or None if none exists).
    get_children(pos):
        Returns a list of the child positions of the specified position.
    get_label(pos):
        Returns the label associated with the specified position.
    is_leaf(pos):
        Returns whether the specified position is a leaf, i.e. a tree position
        that has no child positions.
    get_leaves():
        Returns a list of the leaf positions of the tree.
    get_spans():
        Returns the spans of the tree.
    to_spanmap():
        Returns a dictionary mapping the tree spans to their labels.
    """    

    def __init__(self):
        self.positions = set()
        self.labels = {}
        self.position_to_span = {}
        
    def get_root(self):
        """Returns the root position (i.e. the empty tuple).

        Returns
        -------
        tuple:
            The root position
        """

        return ()

    def get_parent(self, pos):
        """Returns the parent position of the argument.

        Parameters
        ----------
        pos : tuple
            The position of interest

        Returns
        -------
        tuple:
            The parent position of pos (or None if pos is the root or not in the tree)
        """

        if pos not in self.positions or len(pos)==0:
            return None        
        return pos[:-1]

    def get_children(self,pos):
        """Returns a list of the child positions of the argument.

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

        children = []
        counter = 1
        next_child = tuple(list(pos) + [counter])
        while next_child in self.positions:
            children.append(next_child)
            counter += 1
            next_child = tuple(list(pos) + [counter])
        return children


    def get_label(self, pos):
        """Returns the label associated with the specified position.

        Parameters
        ----------
        pos : tuple
            The position of interest.

        Returns
        -------
        object:
            The label associated with the specified position (None if there is no
            associated label or the position does not appear in the tree)
        """
        if pos in self.labels:
            return self.labels[pos]
        else:
            return None


    def is_leaf(self, pos):
        """Returns whether the argument position is a leaf.

        A leaf is a tree position that has no child positions.

        Parameters
        ----------
        pos : tuple
            The position of interest.

        Returns
        -------
        bool:
            True if the argument position is a leaf (and False if it has children or is
            not in the tree at all).
        """

        return (tuple(list(pos) + [1]) not in self.positions) and (pos in self.positions)


    def get_leaves(self):
        """Returns a list of the leaf positions of the tree.

        There is no guarantee on the order of the leaves.

        Returns
        -------
        list of tuples:
            A list of the leaf positions of the tree.
        """

        return filter( self.is_leaf, self.positions )


    def get_spans(self):
        """Returns the spans of the tree.
        
        Let [L1, L2, ..., Lk] be the list of leaf positions in DFS order.
        The span of position P is (i,j), where:
            - i+1 is the minimum index such that Li is a descendant position of P (i.e. P is a prefix of Li)
            - j is the maximum index such that Lj is a descendant position of P (i.e. P is a prefix of Lj)

        A span (i,j) is a span of the tree if there exists some tree position P such that (i,j) is the span of P.

        There is no guarantee on the order of the spans in the returned list.

        Returns
        -------
        set[tuple]:
            A set containing the spans of the tree.
                
        """
        return set(self.position_to_span.values())


    def compile_spans(self):
        """Compiles a map from tree positions to their spans.
        
        Let [L1, L2, ..., Lk] be the list of leaf positions in DFS order.
        The span of position P is (i,j), where:
            - i+1 is the minimum index such that Li is a descendant position of P
              (i.e. P is a prefix of Li)
            - j is the maximum index such that Lj is a descendant position of P
              (i.e. P is a prefix of Lj)

        This function returns nothing, but populates the dictionary self.position_to_span.
                        
        """
        def compute_span(spanmap, pos):
            if pos in spanmap:
                return spanmap[pos]
            else:
                span_start = math.inf
                span_end   = 0
                for p in self.get_children(pos):
                    childspan = compute_span(spanmap, p)
                    if childspan[0] < span_start:
                        span_start = childspan[0]
                    if childspan[1] > span_end:
                        span_end = childspan[1]
                spanmap[pos] = (span_start, span_end)
                return span_start, span_end
        for spanpos, leafpos in enumerate(sorted(self.get_leaves())):
            self.position_to_span[leafpos] = (spanpos, spanpos+1)  # compute the leaf spans
        compute_span(self.position_to_span, tuple())  # compute all spans starting from root
 

    def to_spanmap(self):
        """Returns a map from the tree spans to their labels.
        
        Let [L1, L2, ..., Lk] be the list of leaf positions in DFS order.
        The span of position P is (i,j), where:
            - i+1 is the minimum index such that Li is a descendant position of P
              (i.e. P is a prefix of Li)
            - j is the maximum index such that Lj is a descendant position of P
              (i.e. P is a prefix of Lj)

        A span (i,j) is a span of the tree if there exists some tree position P such
        that (i,j) is the span of P.

        Returns
        -------
        dict:
            A dictionary that maps every tree span (i,j) to a list of the labels
            of its associated positions (in root-to-leaf order).
        """

        def to_spanmap_rec(pos):
            current_span = self.position_to_span[ pos ]
            current_label = self.get_label( pos )
            if current_span in spanmap:
                spanmap[current_span].append(current_label)
            else:
                spanmap[current_span] = [ current_label ]
            for p in self.get_children(pos):
                to_spanmap_rec(p)

        spanmap = {}
        current_pos = self.get_root()
        if current_pos in self.positions:
            to_spanmap_rec(current_pos)
        return spanmap

    def __str__(self):
        def to_string_rec(pos):
            retval = ''
            if self.is_leaf(pos):
                retval += str(self.get_label(pos))
            else:
                child_positions = dfs_sort(self.get_children(pos))
                retval += '(' + str(self.get_label(pos))
                for childpos in child_positions:
                    retval += ' ' + to_string_rec(childpos)
                retval += ')'
            return retval
        return to_string_rec( () )


def str_to_position_tree(s, token_parser=lambda x: x, cls=PositionBasedTree):
    """Construct a PositionBasedTree based on a string.

    An example would be: '(TOP-1 (S-2 (NP-24 (PRP "a")) (VP *0 (AUX *1) (RB "c") (VB "d"))))'

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
        for _ in range(end_paren_count):
            num_children.pop()
    if len(num_children) > 0:
        raise PositionTreeReadError('Reached end of tree string without balanced parens: ' + s )
    t = cls()
    t.positions = labels.keys()
    t.labels = labels
    t.compile_spans()    
    return t


def dfs_sort(positions, postorder=True):
    """Sorts a list of positions in depth-first search order.
    
    If the postorder flag is set to True, then visit children before parents;
    otherwise visit parents before children.

    Parameters
    ----------
    positions : list[tuple]
        A list of positions.
    postorder : boolean
        Set True in order to visit children before parents in the DFS search.

    Returns
    -------
    list[tuple]:
        The positions of the argument list, sorted in DFS order.

    >>> dfs_sort( [ (1,2), (2,1,1), (2,1,2), (), (1,2,1), (1,), (2,) , (1,1), (3,), (1,2,2) ] )
    [(1, 1), (1, 2, 1), (1, 2, 2), (1, 2), (1,), (2, 1, 1), (2, 1, 2), (2,), (3,), ()]
    >>> dfs_sort( [ (1,2), (2,1,1), (2,1,2), (), (1,2,1), (1,), (2,) , (1,1), (3,), (1,2,2) ], False )
    [(), (1,), (1, 1), (1, 2), (1, 2, 1), (1, 2, 2), (2,), (2, 1, 1), (2, 1, 2), (3,)]
    """

    def get_kth_element(x,k):
        post_constant = 0
        if postorder:
            post_constant = float('inf')
        if k < len(x):
            return x[k]
        else:
            return post_constant
    if len(positions) == 0:
        return []
    treedepth = max( map( len, positions ))
    depths = list(range(treedepth))
    depths.reverse()
    for depth in depths:
        positions = sorted(positions, key=lambda x: get_kth_element(x, depth) )		
    return positions


class TreeNode(object):
    """A node of a recursively represented tree.

    Methods
    -------
    get_label()
        Returns the node label.
    get_children()
        Returns a list of the children of the node.
    get_num_children()
        Returns the number of children of the node.
    is_leaf()
        Returns whether the node is a leaf, i.e. has no children.
    get_child(self, index)
        Returns a specified child.
    """

    def __init__(self):
        self.label = None
        self.children = []

    def get_label(self):
        """Returns the node label.

        Important: By convention, TreeNode labels should be tuples.

        Returns
        -------
        tuple:
            The label of this node
        """
        return self.label

    def get_simple_label(self):
        """Assumes that a label is a singleton tuple, and returns the first element.

        Returns
        -------
        str:
            The first element of the label of this node
        """
        return self.label[0]

    def get_children(self):
        """Returns a list of the children of the node.

        Returns
        -------
        list[TreeNode]:
            The children of this node
        """
        return self.children

    def get_num_children(self):
        """Returns the number of children of the node.

        Returns
        -------
        int:
            The number of children of this node
        """
        return len(self.children)

    def is_leaf(self):
        """Returns whether the node is a leaf, i.e. has no children.

        Returns
        -------
        bool:
            True iff the node has no children
        """
        return len(self.children) == 0

    def get_child(self, index):
        """Returns a specified child.

        Parameters
        ----------
        index : int
            The index of the desired child node.

        Returns
        -------
        TreeNode:
            The child corresponding to the specified index (or None if no such child exists)
        """
        if index < len(self.children):
            return self.children[index]
        else:
            return None

    def get_leaves(self):
        """Returns a list of the leaves of the tree rooted at this node.

        Returns
        -------
        list[TreeNode]:
            The leaves of the tree rooted at this node
        """
        if self.is_leaf():
            return [self]
        else:
            result = []
            for i in range(self.get_num_children()):
                result += self.get_child(i).get_leaves()
            return result

    def __str__(self):
        if self.is_leaf():
            retval = TreeNode.label_to_string(self.label)
        else:
            retval = '(' + TreeNode.label_to_string(self.label)
            for child in self.children:
                retval += ' ' + str(child)
            retval += ')'
        return retval

    def __eq__(self, obj):
        """Returns True if the argument tree has the same structure and labels as this one.

        Note: equality is by value, not pointer equality.
        """
        if not (obj.get_label() == self.get_label()):
            return False
        if not (obj.get_num_children() == self.get_num_children()):
            return False
        for childindex in range(self.get_num_children()):
            if not (obj.get_child(childindex) == self.get_child(childindex)):
                return False
        return True

    @staticmethod
    def string_to_label(s):
        return tuple(s.split(COMPOUND_SEP))

    @staticmethod
    def label_to_string(label):
        return COMPOUND_SEP.join([str(component) for component in label])

    @staticmethod
    def make(label_str, children=[]):
        """Constructs a TreeNode from the label string and list of children.

        The compound labels should be separated by the character
        testperanto.globals.COMPOUND_SEP. This function will use the separator to
        split each label string into a tuple.

        An example would be: 'S.13.$y1', for which the root
        label (assuming COMPOUND_SEP==".") would be the tuple ("S", "13", "$y1").

        Parameters
        ----------
        label_str : str
            A string representation of the label of the root node
        children : list[TreeNode]
            The list of child nodes

        Returns
        -------
        TreeNode:
            The TreeNode with the specified label and children
        """
        result = TreeNode()
        result.label = TreeNode.string_to_label(label_str)
        result.children = [c for c in children]
        return result

    @staticmethod
    def from_str(s):
        """Constructs a TreeNode based on a string.

        The compound labels should be separated by the character
        testperanto.globals.COMPOUND_SEP. This function will use the separator to
        split each label string into a tuple.

        An example would be: '(S.13 (NP they) (VP.12 ate))', for which the root
        label (assuming COMPOUND_SEP==".") would be the tuple ("S", "13").

        Parameters
        ----------
        s : int
            A nested parenthetical representation of a tree.

        Returns
        -------
        TreeNode:
            The TreeNode represented by the input string.
        """
        def construct_from_str_rec(postree, pos):
            retval = TreeNode()
            retval.label = postree.get_label(pos)
            retval.children = []
            for childpos in postree.get_children(pos):
                retval.children.append(construct_from_str_rec(postree, childpos))
            return retval
        ptree = str_to_position_tree(s, TreeNode.string_to_label)
        return construct_from_str_rec(ptree, ptree.get_root())


