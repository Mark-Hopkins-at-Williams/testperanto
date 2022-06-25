import argparse
from collections import defaultdict
from testperanto.util import stream_lines
from testperanto.distributions import CategoricalDistribution

def main(corpus):
    adj_count = defaultdict(int)
    noun_count = defaultdict(int)
    for line in stream_lines(corpus):
        adj, noun = line.split()
        adj_count[adj] += 1
        noun_count[noun] += 1
    adj_dist = CategoricalDistribution(weights=[v for _, v in dict(adj_count).items()],
                                       labels=[k for k, _ in dict(adj_count).items()])
    noun_dist = CategoricalDistribution(weights=[v for _, v in dict(noun_count).items()],
                                        labels=[k for k, _ in dict(noun_count).items()])
    with open('indep.en.adj.nn.txt', 'w') as writer:
        for _ in range(1000000):
            writer.write(f'{adj_dist.sample()} {noun_dist.sample()}\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate independent bigrams.')
    parser.add_argument('-c', '--corpus', required=True,
                        help='name of the original corpus file')
    args = parser.parse_args()
    main(args.corpus)
