import sys
from collections import defaultdict
from math import log

def histogram(token_stream):
    counts = defaultdict(int)
    for token in token_stream:
        counts[token] += 1
    return dict(counts)


def rank_frequency(token_stream, _):
    histo = histogram(token_stream)
    freq_count = defaultdict(int)
    for token in histo:
        freq_count[histo[token]] += 1
    cumulative_counts = dict()
    sum_so_far = 0
    for (freq, count) in sorted(freq_count.items())[::-1]:
        sum_so_far += count
        cumulative_counts[freq] = sum_so_far
    ranked_counts = sorted(cumulative_counts.items())
    return [x for (x, _) in ranked_counts], [y for (_, y) in ranked_counts]


def joint_prob(token_stream):
    type_counts = defaultdict(int)
    normalizer = 0
    for token in token_stream:
        type_counts[token] += 1
        normalizer += 1
    return {token: type_counts[token]/normalizer for token in type_counts}


def smoothed_prob(token_stream):
    import nltk
    tokens = list(token_stream)
    freq_dist = nltk.FreqDist(tokens)
    witten_bell = nltk.WittenBellProbDist(freq_dist, len(tokens))
    return {token: witten_bell.prob(token) for token in witten_bell.samples()}


def entropy(token_stream):
    joint = joint_prob(token_stream)
    surprisal = {token: -joint[token] * log(joint[token], 2) for token in joint}
    return sum([surprisal[tok] for tok in surprisal])


def conditional_entropies(bigrams):
    joint_entropy = entropy(bigrams)
    token1_entropy = entropy([bigram.split()[0] for bigram in bigrams])
    token2_entropy = entropy([bigram.split()[1] for bigram in bigrams])
    return token1_entropy, token2_entropy, joint_entropy


def type_counts(bigrams):
    joint = len(set(bigrams))
    token1 = len(set([bigram.split()[0] for bigram in bigrams]))
    token2 = len(set([bigram.split()[1] for bigram in bigrams]))
    return token1, token2, joint


def bigram_info(bigrams):
    type1_count, type2_count, joint_count = type_counts(bigrams)
    print('|X1|       = {}'.format(type1_count))
    print('|X2|       = {}'.format(type2_count))
    print('|X1, X2|   = {}'.format(joint_count))
    token1_entropy, token2_entropy, joint_entropy = conditional_entropies(bigrams)
    print('H(X1)      = {}'.format(token1_entropy))
    print('H(X2)      = {}'.format(token2_entropy))
    print('H(X1, X2)  = {}'.format(joint_entropy))
    print('H(X2 | X1) = {}'.format(joint_entropy - token1_entropy))
    print('H(X1 | X2) = {}'.format(joint_entropy - token2_entropy))


def main(bigram_file):
    with open(bigram_file, 'r') as reader:
        bigrams = list(reader)
        bigram_info(bigrams)

if __name__ == '__main__':
    main(sys.argv[1])