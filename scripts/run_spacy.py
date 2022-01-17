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
    return doc_bin


def process_spacy_doc(f, spacy_doc_file):
    with open(spacy_doc_file, 'rb') as reader:
        print("loading")
        nlp = spacy.blank("en")
        doc_bin = DocBin().from_bytes(reader.read())
        for doc in tqdm(doc_bin.get_docs(nlp.vocab)):
            f(doc)


class TaggedWordCounter:

    def __init__(self):
        self.words = defaultdict(list)

    def __call__(self, doc):
        for token in doc:
            self.words[token.pos_].append(token.text)

    def to_json(self, json_file):
        with open(json_file, 'w', encoding='utf-8') as writer:
            json.dump(dict(self.words), writer, ensure_ascii=False)

    @staticmethod
    def from_json(json_file):
        with open(json_file, 'r', encoding='utf-8') as reader:
            words = json.load(reader)
        result = TaggedWordCounter()
        result.words = words
        return result


def tag_with_spacy(in_file, spacy_model_name):
    nlp = spacy.load(spacy_model_name)
    words = defaultdict(list)
    with open(in_file, 'r') as reader:
        for doc in tqdm(nlp.pipe(reader)):
            for token in doc:
                # print('{} {} {} {} {}'.format(token.text, token.pos_, token.dep_, token.head, token.head.pos_))
                words[token.pos_].append(token.text)
    for tag in words:
        with open('{}.{}.txt'.format(in_file, tag), 'w') as writer:
            for word in words[tag]:
                writer.write("{}\n".format(word))


def spacy_dependencies(in_file, spacy_model_name):
    nlp = spacy.load(spacy_model_name)
    bigrams = defaultdict(list)
    desired_deps = [("ADJ", "amod", "NOUN"), ("NOUN", "nsubj", "VERB"), ("NOUN", "dobj", "VERB")]
    with open(in_file, 'r') as reader:
        for doc in tqdm(nlp.pipe(reader)):
            for token in doc:
                # print('{} {} {} {} {}'.format(token.text, token.pos_, token.dep_, token.head, token.head.pos_))
                if (token.pos_, token.dep_, token.head.pos_) in desired_deps:
                    bigrams[token.dep_].append('{} {}'.format(token.text, token.head))
    for dep in bigrams:
        with open('{}.{}.txt'.format(in_file, dep), 'w') as writer:
            for bigram in bigrams[dep]:
                writer.write("{}\n".format(bigram))


def main(in_file, spacy_model_name):
    out_file = '{}.spacy'.format(in_file)
    run_spacy(in_file, out_file, spacy_model_name)
    counter = TaggedWordCounter()
    process_spacy_doc(counter, out_file)
    counter.to_json('{}.tags.json'.format(out_file))


if __name__ == '__main__':
    tag_with_spacy(sys.argv[1], sys.argv[2])
    spacy_dependencies(sys.argv[1], sys.argv[2])