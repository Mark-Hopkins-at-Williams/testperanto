import argparse
from tqdm import tqdm
from testperanto.config import init_transducer_cascade
from testperanto.globals import EMPTY_STR, DOT
from testperanto.transducer import run_transducer_cascade
from testperanto.amr import amr_str

import random
random.seed(18)

def main(config_files, switching_code, num_to_generate, only_sents, vbox_theme="goose"):
    cascade = init_transducer_cascade(config_files, switching_code, vbox_theme=vbox_theme)
    for _ in tqdm(range(num_to_generate)):
        output = run_transducer_cascade(cascade)
        if only_sents:
            leaves = [DOT.join(leaf.get_label()) for leaf in output.get_leaves()]
            leaves = [leaf for leaf in leaves if leaf != EMPTY_STR]
            output = ' '.join(leaves)
        print(output)
        #print(amr_str(output))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate trees using testperanto.')
    parser.add_argument('-c', '--configs', nargs='+', required=True,
                        help='names of the transducer config files in the cascade')
    parser.add_argument('-n', '--num', required=True, type=int,
                        help='number of trees to generate')
    parser.add_argument('-s', '--switches', required=False, default=None,
                        help='the typological switches, as a bitstring')
    parser.add_argument('-v', '--vbox_theme', required=False, default="goose",
                        help='the voicebox theme')
    parser.add_argument('--sents', dest='sents', action='store_true', default=False,
                        help='only output sentences (rather than trees)')
    args = parser.parse_args()
    main(args.configs, args.switches, args.num, args.sents, args.vbox_theme)

