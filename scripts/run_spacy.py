from collections import defaultdict
from os.path import join
import spacy
from spacy.tokens import DocBin
import sys
from testperanto.corpora import TaggedWordCounter
from tqdm import tqdm


def run_spacy(in_file, out_file, spacy_model_name):
    doc_bin = DocBin(store_user_data=True)
    nlp = spacy.load(spacy_model_name)
    with open(in_file, 'r') as reader:
        for doc in tqdm(nlp.pipe(reader)):
            doc_bin.add(doc)
    with open(out_file, 'wb') as writer:
        bytes_data = doc_bin.to_bytes()
        writer.write(bytes_data)


def process_spacy_doc(f, spacy_doc_file):
    with open(spacy_doc_file, 'rb') as reader:
        print("loading")
        nlp = spacy.blank("en")
        doc_bin = DocBin().from_bytes(reader.read())
        for doc in tqdm(doc_bin.get_docs(nlp.vocab)):
            f(doc)


def main(in_file, spacy_model_name):
    out_file = '{}.spacy'.format(in_file)
    run_spacy(in_file, out_file, spacy_model_name)
    counter = TaggedWordCounter()
    process_spacy_doc(counter, out_file)
    counter.to_json('{}.tags.json'.format(out_file))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])