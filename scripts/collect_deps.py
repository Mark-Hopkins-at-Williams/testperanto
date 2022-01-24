import argparse
from collections import defaultdict
import spacy
from tqdm import tqdm


def spacy_dependencies(in_file, spacy_model_name):
    nlp = spacy.load(spacy_model_name)
    bigrams = defaultdict(list)
    desired_deps = [("ADJ", "amod", "NOUN"),
                    ("NOUN", "nsubj", "VERB"),
                    ("NOUN", "dobj", "VERB")]
    with open(in_file, 'r') as reader:
        for doc in tqdm(nlp.pipe(reader)):
            for token in doc:
                if (token.pos_, token.dep_, token.head.pos_) in desired_deps:
                    bigrams[token.dep_].append('{} {}'.format(token.text, token.head))
    for dep in bigrams:
        with open('{}.{}.txt'.format(in_file, dep), 'w') as writer:
            for bigram in bigrams[dep]:
                writer.write("{}\n".format(bigram))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Organize tokens by dependency relationships.')
    parser.add_argument('-c', '--corpus', type=str, required=True,
                        help='name of corpus file (one sentence per line)')
    parser.add_argument('-m', '--model', type=str,
                        help='name of spacy model (e.g. "en_core_web_md", "el_core_news_md")')
    args = parser.parse_args()
    spacy_dependencies(args.corpus, args.model)


