import json
import os
import sys
from tqdm import tqdm
from testperanto.macros import TreeTransducer, init_transducer_cascade
from testperanto.trees import TreeNode, leaf_string
from testperanto.voicebox import VoiceboxFactory


def all_switching_codes(k):
    if k == 1:
        return ["0", "1"]
    else:
        suffixes = all_switching_codes(k-1)
        return ["0" + suffix for suffix in suffixes] + ["1" + suffix for suffix in suffixes]


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


def train_valid_test_split(filename):
    sents = []
    with open(filename, 'r') as reader:
        lines = [line.strip() for line in reader]
        for line in tqdm(lines):
            sents.append(leaf_string(TreeNode.construct_from_str(line.strip())))
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
    for c in all_switching_codes(6):
        train_valid_test_split('{}.{}.trees'.format(sys.argv[1], c))

