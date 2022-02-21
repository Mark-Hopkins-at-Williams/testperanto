import argparse
from tqdm import tqdm
from testperanto.parses import get_dependencies
from testperanto.trees import TreeNode


def main(ptb_file, desired_deprels):
    with open(ptb_file, 'r') as reader:
        for line in tqdm(list(reader)):
            tree = TreeNode.construct_from_str(line)
            desired = [(x,z) for (x,deprel,z) in get_dependencies(tree) if deprel in desired_deprels]
            for (dependent, head) in desired:
                print("{} {}".format(dependent.lower(), head.lower()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Harvest dependency relationships for testperanto trees.')
    parser.add_argument('-t', '--trees', required=True, type=str,
                        help='file containing the testperanto trees (one per line)')
    parser.add_argument('-r', '--relations', nargs='+', required=True,
                        help='dependency relations to record')
    args = parser.parse_args()
    main(args.trees, args.relations)
