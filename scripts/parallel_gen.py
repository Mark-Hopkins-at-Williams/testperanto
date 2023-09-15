import argparse
from tqdm import tqdm
from testperanto.config import init_transducer_tree
from testperanto.globals import EMPTY_STR, DOT
from testperanto.transducer import run_transducer_cascade
import yaml

def main(config_file, num_to_generate, output_file_prefix, vbox_theme="universal"):
    ttree = init_transducer_tree(config_file, vbox_theme)
    parallel_corpora = []
    for _ in tqdm(range(num_to_generate)):
        outputs = ttree.run()
        for i, output in enumerate(outputs):
            while i >= len(parallel_corpora):
                parallel_corpora.append([])
            leaves = [DOT.join(leaf.get_label()) for leaf in output.get_leaves()]
            leaves = [leaf for leaf in leaves if leaf != EMPTY_STR]
            output = ' '.join(leaves)
            parallel_corpora[i].append(output)            
    for i, corpus in enumerate(parallel_corpora):
        with open(f'{output_file_prefix}.{i}', 'w') as writer:
            for line in corpus:
                writer.write(line + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate trees using testperanto.')
    parser.add_argument('-c', '--config', required=True,
                        help='name of the YAML config file')
    parser.add_argument('-n', '--num', required=True, type=int,
                        help='number of trees to generate')
    parser.add_argument('-o', '--out_prefix', required=True,
                        help='prefix of the produced parallel corpus files')
    parser.add_argument('-v', '--vbox_theme', required=False, default="universal",
                        help='the voicebox theme')
    args = parser.parse_args()
    main(args.config, args.num, args.out_prefix, args.vbox_theme)

