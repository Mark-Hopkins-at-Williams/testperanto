import json
import os
import sys
from tqdm import tqdm
from testperanto.macros import TreeTransducer, init_transducer_cascade
from testperanto.trees import TreeNode, LeafLabelCollector
from testperanto.voicebox import VoiceboxFactory

SWITCHED = [None,
            "($qs (S (nsubj $x1) (head $x2))) -> (S (head ($qvp $x2)) (nsubj ($qnp $x1)))",
            "($qvp (VP (head $x1) (obj $x2))) -> (VP (head ($qword $x1)) (obj ($qnp $x2)))",
            None,
            None,
            "($qcomp (S (prt that) (head $x1))) -> (S (prt that) (head ($qs $x1)))",
            None,
            "($qpp (PP (prep $x1) (head $x2))) -> (PP (prep ($qword $x1)) (head ($qnp $x2)))",
            "($qnp (NP (head $x1) (relc (S (prt $x2) (head $x3))))) -> (NP (head $x1) (relc (S (prt $x2) (head $x3))))",
            "($qnp (NP (head $x1) (pp $x2))) -> (NP (head ($qnpb $x1)) (pp ($qpp $x2)))",
            None,
            None,
            "($qnpb (NP (dt $x1) (amod $x2) (head $x3))) -> (NP (dt ($qword $x1)) (head ($qword $x3)) (amod ($qword $x2)))",
            None,
            None,
            None,
            None]

RULES = ["($qstart $x1) -> ($qs $x1)",
         "($qs (S (nsubj $x1) (head $x2))) -> (S (nsubj ($qnp $x1)) (head ($qvp $x2)))",
         "($qvp (VP (head $x1) (obj $x2))) -> (VP (obj ($qnp $x2)) (head ($qword $x1)))",
         "($qvp (VP (head $x1))) -> (VP (head ($qword $x1)))",
         "($qvp (VP (head $x1) (comp $x2))) -> (VP (head ($qword $x1)) (comp ($qcomp $x2)))",
         "($qcomp (S (prt that) (head $x1))) -> (S (head ($qs $x1)) (prt that))",
         "($qrelc (S (prt which) (head $x1))) -> (S (prt which) (head ($qvp $x1)))",
         "($qpp (PP (prep $x1) (head $x2))) -> (PP (head ($qnp $x2)) (prep ($qword $x1)))",
         "($qnp (NP (head $x1) (relc (S (prt $x2) (head $x3))))) -> (NP (relc (S (head $x3) (prt $x2)) (head $x1)))",
         "($qnp (NP (head $x1) (pp $x2))) -> (NP (pp ($qpp $x2)) (head ($qnpb $x1)))",
         "($qnp (NP (head $x1) (comp $x2))) -> (NP (head ($qnpb $x1)) (comp ($qcomp $x2)))",
         "($qnp (NP (head $x1))) -> (NP (head ($qnpb $x1)))",
         "($qnpb (NP (dt $x1) (amod $x2) (head $x3))) -> (NP (dt ($qword $x1)) (amod ($qword $x2)) (head ($qword $x3)))",
         "($qnpb (NP (dt $x1) (head $x2))) -> (NP (dt ($qword $x1)) (head ($qword $x2)))",
         "($qnpb (NP (head (NN i)))) -> (NP (head (NN i)))",
         "($qnpb (NP (head (NN you)))) -> (NP (head (NN i)))",
         "($qword $x1) -> $x1"]

CONFIG = {'distributions': [], "macros": [{"rule": r} for r in RULES]}

"""
def choose_rule(code, index):
    if ((code[0] == "1" and index == 1)
        or (code[1] == "1" and index == 2)
        or (code[2] == "1" and index == 5)
        or (code[3] == "1" and index == 7)
        or (code[3] == "1" and index == 9)
        or (code[4] == "1" and index == 12)
        or (code[5] == "1" and index == 8)):

        return SWITCHED[index]
    else:
        return RULES[index]


def get_switching_transducer(code):
    rules = [choose_rule(code, index) for index in range(len(RULES))]
    config = {'distributions': [], "macros": [{"rule": r} for r in rules]}
    return TreeTransducer.from_config(config)
"""

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


def flatten(tree):
    collector = LeafLabelCollector()
    collector.execute(tree)
    leaves = ['~'.join(leaf) for leaf in collector.get_leaf_labels()]
    leaves = [leaf for leaf in leaves if leaf != "NULL"]
    return ' '.join(leaves)


def impose_typology(tree_file, switching_code):
    cascade = [get_switching_transducer(switching_code)]
    vfactory = VoiceboxFactory()
    vbox = vfactory.create_voicebox("seuss")
    cascade.append(vbox)
    with open('{}.{}.trees'.format(tree_file, switching_code), 'w') as writer:
        with open(tree_file, 'r') as reader:
            for line in tqdm(reader):
                orig_in_tree = TreeNode.construct_from_str('({} {})'.format('$qstart', line.strip()))
                in_tree = orig_in_tree
                for transducer in cascade[:-1]:
                    out_tree = transducer.run(in_tree)
                    in_tree = TreeNode.construct_from_str('({} {})'.format('$qstart', out_tree))
                output = cascade[-1].run(in_tree).get_child(0)
                writer.write('{}\n'.format(str(output)))


def all_switching_codes(k):
    if k == 1:
        return ["0", "1"]
    else:
        suffixes = all_switching_codes(k-1)
        return ["0" + suffix for suffix in suffixes] + ["1" + suffix for suffix in suffixes]


def train_valid_test_split(filename):
    sents = []
    with open(filename, 'r') as reader:
        lines = [line.strip() for line in reader]
        for line in tqdm(lines):
            sents.append(flatten(TreeNode.construct_from_str(line.strip())))
    directory, file = os.path.split(filename)
    stem, ext = os.path.splitext(file)
    for fold in range(10):
        base_dir = os.path.join(directory, '{}.fold{}'.format(stem, fold))
        os.mkdir(base_dir)
        tenth = sents[int(0.1 * fold * len(sents)):int((0.1 + 0.1 * fold) * len(sents))]
        train_file = os.path.join(base_dir, '{}.fold{}.train'.format(stem, fold))
        valid_file = os.path.join(base_dir, '{}.fold{}.valid'.format(stem, fold))
        test_file = os.path.join(base_dir, '{}.fold{}.test'.format(stem, fold))
        with open(train_file, 'w') as writer:
            for sent in tenth[:int(.8*len(tenth))]:
                writer.write('{}\n'.format(sent))
        with open(valid_file, 'w') as writer:
            for sent in tenth[int(.8*len(tenth)):int(.9*len(tenth))]:
                writer.write('{}\n'.format(sent))
        with open(test_file, 'w') as writer:
            for sent in tenth[int(.9*len(tenth)):]:
                writer.write('{}\n'.format(sent))


if __name__ == '__main__':
    for code in all_switching_codes(6):
        train_valid_test_split('{}.{}.trees'.format(sys.argv[1], code))


    #     impose_typology(sys.argv[1], code)



