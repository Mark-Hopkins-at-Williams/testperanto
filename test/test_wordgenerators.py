##
# test_wordgenerators.py
# Unit tests for wordgenerators.py.
##

import unittest
from testperanto.distributions import Distribution
from testperanto.wordgenerators import WordGeneratorFactory
from testperanto.wordgenerators import IteratingWordGenerator, PrefixSuffixWordGenerator
from testperanto.wordgenerators import ListBasedWordGenerator, AtomBasedWordGenerator

class IteratingIntegerDistribution(Distribution):
    def __init__(self):
        self.ints = [1,2,3]
        self.current = 0

    def sample(self):
        result = self.ints[self.current]
        self.current = (self.current + 1) % 3
        return result


class TestVoicebox(unittest.TestCase):

    def test_list_based1(self):
        generator = ListBasedWordGenerator(['a', 'b', 'c', 'd'],  lambda x: x[2])
        self.assertEqual(generator.generate(), 'c')
        self.assertEqual(generator.generate(), 'c')

    def test_iterating(self):
        backup = ListBasedWordGenerator(['a', 'b', 'c', 'd'],  lambda x: x[-1])
        generator = IteratingWordGenerator(['z', 'y'], backup)
        self.assertEqual(generator.generate(), 'z')
        self.assertEqual(generator.generate(), 'y')
        self.assertEqual(generator.generate(), 'd')
        self.assertEqual(generator.generate(), 'd')

    def test_prefix_suffix(self):
        backup = ListBasedWordGenerator(['a', 'b', 'c', 'bon'],  lambda x: x[-1])
        prefix_generator = IteratingWordGenerator(['artisan', 'craft'], backup)
        suffix_generator = IteratingWordGenerator(['ally', 'ed'], backup)
        generator = PrefixSuffixWordGenerator(prefix_generator, suffix_generator)
        self.assertEqual(generator.generate(), 'artisanally')
        self.assertEqual(generator.generate(), 'crafted')
        self.assertEqual(generator.generate(), 'bonbon')
        self.assertEqual(generator.generate(), 'bonbon')

    def test_atom_based(self):
        atom_generator = IteratingWordGenerator(list('abcdefghijklmnop'), None)
        generator = AtomBasedWordGenerator(atom_generator, IteratingIntegerDistribution())
        self.assertEqual(generator.generate(), 'a')
        self.assertEqual(generator.generate(), 'bc')
        self.assertEqual(generator.generate(), 'def')
        self.assertEqual(generator.generate(), 'g')
        self.assertEqual(generator.generate(), 'hi')
        self.assertEqual(generator.generate(), 'jkl')


if __name__ == "__main__":
    unittest.main()   