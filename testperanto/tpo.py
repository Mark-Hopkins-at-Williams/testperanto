import json
import sys
from tqdm import tqdm
from testperanto.macros import TreeTransducer
from testperanto.trees import TreeNode, LeafLabelCollector
from testperanto.voicebox import VoiceboxFactory


def init_transducer_cascade(config_files):
    cascade = []
    for config_file in config_files:
        with open(config_file, 'r') as reader:
            cascade.append(TreeTransducer.from_config(json.load(reader)))
    vfactory = VoiceboxFactory()
    vbox = vfactory.create_voicebox("seuss")
    cascade.append(vbox)
    return cascade


def run_transducer_cascade(cascade, start_state='$qstart'):
    in_tree = TreeNode.construct_from_str(start_state)
    for transducer in cascade[:-1]:
        out_tree = transducer.run(in_tree)
        in_tree = TreeNode.construct_from_str('({} {})'.format(start_state, out_tree))
    output = cascade[-1].run(in_tree).get_child(0)
    return output


def main(config_files, num_to_generate):
    cascade = init_transducer_cascade(config_files)
    for _ in tqdm(range(num_to_generate)):
        output = run_transducer_cascade(cascade)
        print(output)
        # collector = LeafLabelCollector()
        # collector.execute(output)
        # leaves = ['~'.join(leaf) for leaf in collector.get_leaf_labels()]
        # leaves = [leaf for leaf in leaves if leaf != "NULL"]
        # print(' '.join(leaves))


if __name__ == '__main__':
    main(sys.argv[1].split(','), int(sys.argv[2]))



