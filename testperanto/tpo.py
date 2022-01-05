import json
import sys
from testperanto.macros import TreeTransducer
from testperanto.trees import TreeNode
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
    output = cascade[-1].run(in_tree)
    return output


def main(config_files):
    cascade = init_transducer_cascade(config_files)
    for _ in range(10):
        output = run_transducer_cascade(cascade)
        print(output)




if __name__ == '__main__':
    main(sys.argv[1].split(','))



