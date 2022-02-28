##
# util.py
# Miscellaneous utility functions.
##


from collections import defaultdict
import json
from nltk.corpus import brown
from nltk import pos_tag, word_tokenize
import spacy
from spacy.tokens import DocBin
from tqdm import tqdm
from testperanto.globals import COMPOUND_SEP


def compound(symbols):
    return COMPOUND_SEP.join([str(s) for s in symbols])


def rhs_refinement_var(i):
    return '$z{}'.format(i)


def is_state(label):
    try:
        return label[0][:2] == '$q'
    except Exception:
        return False


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


def stream_lines(filename):
    with open(filename, 'r') as reader:
        for line in reader:
            yield line.strip()
