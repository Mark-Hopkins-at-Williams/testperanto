import argparse
from collections import defaultdict
import spacy
from tqdm import tqdm


def tag_with_spacy(in_file, spacy_model_name):
    nlp = spacy.load(spacy_model_name)
    words = defaultdict(list)
    with open(in_file, 'r') as reader:
        for doc in tqdm(nlp.pipe(reader)):
            for token in doc:
                words[token.pos_].append(token.text)
    for tag in words:
        with open('{}.{}.txt'.format(in_file, tag), 'w') as writer:
            for word in words[tag]:
                writer.write("{}\n".format(word))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Organize tokens by part-of-speech tag.')
    parser.add_argument('-c', '--corpus', type=str, required=True,
                        help='name of corpus file (one sentence per line)')
    parser.add_argument('-m', '--model', type=str,
                        help='name of spacy model (e.g. "en_core_web_md", "el_core_news_md")')
    args = parser.parse_args()
    tag_with_spacy(args.corpus, args.model)


