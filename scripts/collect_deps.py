import argparse
from collections import defaultdict
import spacy
from tqdm import tqdm


def spacy_deps(in_file, spacy_model_name):
    nlp = spacy.load(spacy_model_name)
    with open(in_file, 'r') as reader:
        for doc in tqdm(nlp.pipe(reader)):
            for token in doc:
                yield (token.pos_, token.text), token.dep_, (token.head.pos_, token.head.text)


def spacy_chains(in_file, spacy_model_name):
    nlp = spacy.load(spacy_model_name)
    with open(in_file, 'r') as reader:
        for doc in tqdm(nlp.pipe(reader)):
            for token in doc:
                yield (token.pos_, token.text), token.dep_, (token.head.pos_, token.head.text), token.head.dep_, (token.head.head.pos_, token.head.head.text)


def spacy_dependencies(in_file, spacy_model_name):
    bigrams = defaultdict(list)
    desired_deps = [("ADJ", "amod", "NOUN"),
                    ("NOUN", "nsubj", "VERB"),
                    ("NOUN", "dobj", "VERB")]
    for (tok_pos, tok), dep, (head_pos, head) in spacy_deps(in_file, spacy_model_name):
        if (tok_pos, dep, head_pos) in desired_deps:
            bigrams[dep].append('{} {}'.format(tok, head))
    for dep in bigrams:
        with open('{}.{}.txt'.format(in_file, dep), 'w') as writer:
            for bigram in bigrams[dep]:
                writer.write("{}\n".format(bigram))


def extract_chains(in_file, spacy_model_name):
    bigrams = defaultdict(list)
    desired = [("NOUN", "pobj", "ADP", "prep", "NOUN")]
    for (tok_pos, tok), dep1, (parent_pos, parent), dep2, (grandparent_pos, grandparent) in spacy_chains(in_file, spacy_model_name):
        if (tok_pos, dep1, parent_pos, dep2, grandparent_pos) in desired:
            bigrams[(dep1, dep2)].append('{} {} {}'.format(grandparent, parent, tok))
    for (dep1, dep2) in bigrams:
        with open('{}.{}.{}.txt'.format(in_file, dep1, dep2), 'w') as writer:
            for bigram in bigrams[(dep1, dep2)]:
                writer.write("{}\n".format(bigram))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Organize tokens by dependency relationships.')
    parser.add_argument('-c', '--corpus', type=str, required=True,
                        help='name of corpus file (one sentence per line)')
    parser.add_argument('-m', '--model', type=str,
                        help='name of spacy model (e.g. "en_core_web_md", "el_core_news_md")')
    args = parser.parse_args()
    extract_chains(args.corpus, args.model)
    #for thing in spacy_chains(args.corpus, args.model):
    #    print(thing)


