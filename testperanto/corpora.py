from collections import defaultdict
import json
from nltk.corpus import brown
from nltk import pos_tag, word_tokenize
import spacy
from spacy.tokens import DocBin
from tqdm import tqdm

def stream_ngrams(lines, ngram_order, tokenize = lambda line: line.split()):
    def is_number(s):
        try:
            float(s)
            return True
        except ValueError:
            return False

    def replace_numbers(word):
        if is_number(word):
            return "*NUM*"
        else:
            return word

    for line in lines:
        iwords = tokenize(line)
        words = [replace_numbers(wd) for wd in iwords]
        for i in range(0, len(words) - ngram_order + 1):
            ngram = ' '.join(words[i:i+ngram_order])
            yield ngram

def brown_ngrams(n):
    return stream_ngrams(brown.sents(), n, lambda line: line)

def stream_plaintext(filename, ngram_order):
    with open(filename, 'r') as reader:
        return list(stream_ngrams(reader, ngram_order))

def stream_one_word_per_line(lines, index, tokenize = lambda line: line.split()):
    for line in lines:
        iwords = tokenize(line)
        yield iwords[index]

def stream_lines(filename):
    with open(filename, 'r') as reader:
        for line in reader:
            yield line.strip()

def stream_plaintext_target_word(filename, index):
    with open(filename, 'r') as reader:
        return list(stream_one_word_per_line(reader, index))


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

if __name__ == '__main__':
    #counter = TaggedWordCounter()
    #process_spacy_doc(counter, '/Users/markhopkins/data/europarl/europarl.tagged.el')
    print("loading")
    counter = TaggedWordCounter.from_json('europarl.el.tokens.json')
    print("in memory")
