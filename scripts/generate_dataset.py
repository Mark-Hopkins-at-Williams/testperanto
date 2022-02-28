import argparse
import json
import os
import sys
from tqdm import tqdm
from testperanto.globals import DOT, EMPTY_STR
from testperanto.config import init_transducer_cascade
from testperanto.transducer import TreeTransducer, run_transducer_cascade
from testperanto.trees import TreeNode
from testperanto.voicebox import lookup_voicebox_theme


def generate(config_files, switching_code, num_to_generate, only_sents):
    cascade = init_transducer_cascade(config_files, switching_code)
    for _ in tqdm(range(num_to_generate)):
        output = run_transducer_cascade(cascade)
        if only_sents:
            leaves = [DOT.join(leaf.get_label()) for leaf in output.get_leaves()]
            leaves = [leaf for leaf in leaves if leaf != EMPTY_STR]
            output = ' '.join(leaves)
        yield output


def split_data(sents, base_dir, percentages=(.8,.1,.1)):
    os.mkdir(base_dir)
    train_pct, valid_pct, test_pct = percentages
    train_file = os.path.join(base_dir, 'sents.train')
    valid_file = os.path.join(base_dir, 'sents.valid')
    test_file = os.path.join(base_dir, 'sents.test')
    with open(train_file, 'w') as writer:
        for sent_index in range(int(train_pct * len(sents))):
            writer.write('{}\n'.format(sents[sent_index]))
    with open(valid_file, 'w') as writer:
        for sent_index in range(int(train_pct * len(sents)), int((train_pct+valid_pct) * len(sents))):
            writer.write('{}\n'.format(sents[sent_index]))
    with open(test_file, 'w') as writer:
        for sent_index in range(int((train_pct+valid_pct) * len(sents)), int(len(sents))):
            writer.write('{}\n'.format(sents[sent_index]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate LM training data using testperanto.')
    parser.add_argument('-c', '--configs', nargs='+', required=True,
                        help='names of the transducer config files in the cascade')
    parser.add_argument('-n', '--num', required=True, type=int,
                        help='number of trees to generate')
    parser.add_argument('-s', '--switches', required=False, default=None,
                        help='the typological switches, as a bitstring')
    parser.add_argument('--sents', dest='sents', action='store_true', default=False,
                        help='only output sentences (rather than trees)')
    parser.add_argument('-d', '--dir', required=True,
                        help='output directory for the data files')
    args = parser.parse_args()
    sentences = list(generate(args.configs, args.switches, args.num, args.sents))
    split_data(sentences, args.dir)
