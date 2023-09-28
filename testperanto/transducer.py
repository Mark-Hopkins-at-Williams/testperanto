##
# transducer.py
# Implementation of tree transduction.
##


from testperanto.trees import TreeNode
from testperanto.util import is_state


class TreeTransducer:
    """Recursively applies transducer rules to a tree until no state labels remain.

    States are labels prefixed by "$q".

    As an example, consider the input tree
        ($qs (S (NN i) (VP (VB am) (NN mark))))
    and a grammar with transducer rules
        1. ($qs (S $x1 (VP $x2 $x3))) -> (S (NP ($qsubj $x1) wa) ($qobj $x3) ($qv $x2))
        2. ($qsubj (NN i)) -> (NN watashi)
        3. ($qv (VB am)) -> (VB desu)
        4. ($qobj (NN mark)) -> (NN mark)
    Calling the method .run(in_tree) outputs the following tree:
        (S (NP (NN watashi) wa) (NN mark) (VB desu))
    because of the following sequence of rule applications:
        ($qs (S (NN i) (VP (VB am) (NN mark))))
        ==rule 1 ==>  (S (NP ($qsubj (NN i)) wa) ($qobj (NN mark)) ($qv (VB am)))
        ==rule 2 ==>  (S (NP (NN watashi) wa) ($qobj (NN mark)) ($qv (VB am)))
        ==rule 4 ==>  (S (NP (NN watashi) wa) (NN mark) ($qv (VB am)))
        ==rule 3 ==>  (S (NP (NN watashi) wa) (NN mark) (VB desu))

    Applicable rules are chosen in proportion to their weights, as determined by
    testperanto.rules.IndexedRuleSet.choose_rule().

    """

    def __init__(self, grammar):
        self.grammar = grammar

    def run(self, in_tree, recursion_depth=0):
        """Recursively applies transducer rules to a tree until no state labels remain.

        Parameters
        ----------
        in_tree : testperanto.trees.TreeNode
            The input tree
        recursion_depth : int
            Recursive depth of the derivation when this function is called

        Returns
        -------
        testperanto.trees.TreeNode
            A transformation of the input tree using the transducer rules of the grammar

        Raises
        ------
        IndexError
            If there is a point when no indexed rule in the grammar can be applied to
            the input tree
        """

        if is_state(in_tree.get_label()):
            rule = self.grammar.choose_rule(in_tree, recursion_depth)
            retval = self.run(rule.apply(in_tree), recursion_depth + 1)
        else:
            retval = TreeNode()
            retval.label = in_tree.get_label()
            retval.children = []
            for i in range(in_tree.get_num_children()):
                retval.children.append(self.run(in_tree.get_child(i), recursion_depth))
        return retval

    def __str__(self):
        """Overrides the default string representation to enumerate the grammar rules."""
        return str(self.grammar)


def run_transducer_cascade(cascade, start_state='$qstart', verbose=False):
    """Executes a cascade of tree transducers.

    A cascade is a sequence of tree transducers where the output of the kth transducer
    is given as input to the (k+1)th transducer, until a final output is obtained.

    Parameters
    ----------
    cascade : list[testperanto.transducer.TreeTransducer]
        The sequence of tree transducers to run
    start_state : str
        Start state for the transducers (they must all have the same start state)

    Returns
    -------
    testperanto.trees.TreeNode
        The output of the cascade
    """

    in_tree = TreeNode.from_str(start_state)
    for transducer in cascade[:-1]:
        out_tree = transducer.run(in_tree)
        if verbose:
            print(out_tree.pretty_print())
        in_tree = TreeNode.from_str(f'({start_state} {out_tree})')
    output = cascade[-1].run(in_tree)
    if is_state(output.get_label()):
        output = output.get_child(0)
    return output


class TransducerTree:
    def __init__(self, transducer):
        self.transducer = transducer
        self.children = []
    
    def add_child(self, tree):
        self.children.append(tree)

    def run(self, in_tree=TreeNode.from_str('$qstart')):
        out_tree = self.transducer.run(in_tree)
        if len(self.children) > 0:
            results = []
            in_tree = TreeNode.from_str(f'($qstart {out_tree})')
            for child in self.children:
                try:
                    results.extend(child.run(in_tree))
                except TypeError:
                    results.extend([child.run(in_tree)])
            return results
        else:
            if is_state(out_tree.get_label()):
                out_tree = out_tree.get_child(0)
            return [out_tree]
