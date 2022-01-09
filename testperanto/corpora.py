from nltk.corpus import brown

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

def stream_plaintext_target_word(filename, index):
    with open(filename, 'r') as reader:
        return list(stream_one_word_per_line(reader, index))
