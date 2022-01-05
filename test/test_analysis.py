import unittest
from testperanto.analysis import type_counts, singleton_proportion
from testperanto.corpora import stream_ngrams

class TestAnalysis(unittest.TestCase):

    def test_type_counts_1gram(self):
        sents = ['the dog barked',
                 'i said hello to the dog',
                 'the cat and the dog said hello']

        one_grams = stream_ngrams(sents, 1)
        x_vals, y_vals = type_counts(one_grams, range(1, 100))
        assert x_vals == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        assert y_vals == [1, 2, 3, 4, 5, 6, 7, 7, 7, 7, 8, 9, 9, 9, 9, 9]

    def test_type_counts_2gram(self):
        sents = ['the dog barked',
                 'i said hello to the dog',
                 'the cat and the dog said hello']

        two_grams = stream_ngrams(sents, 2)
        x_vals, y_vals = type_counts(two_grams, range(1, 100))
        assert x_vals == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
        assert y_vals == [1, 2, 3, 4, 5, 6, 6, 7, 8, 9, 9, 10, 10]

    def test_singleton_proportion_1gram(self):
        sents = ['the dog barked',
                 'i said hello to the dog',
                 'the cat and the dog said hello']

        one_grams = stream_ngrams(sents, 1)
        x_vals, y_vals = singleton_proportion(one_grams, range(1, 100))
        assert x_vals == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        assert y_vals == [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 6/7, 5/7,
                          5/7, 6/8, 7/9, 7/9, 7/9, 6/9, 5/9]

if __name__ == "__main__":
    unittest.main()
