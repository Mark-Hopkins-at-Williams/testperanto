##
# test_util.py
# Unit tests for util.py.
##


import unittest
from testperanto.globals import DOT
from testperanto.util import compound, zvar, is_state, stream_ngrams



class TestUtil(unittest.TestCase):

    def test_config(self):
        self.assertEqual(compound(['NP', '3', '$y1']), f"NP{DOT}3{DOT}$y1")

    def test_zvar(self):
        self.assertEqual(zvar(32), "$z32")

    def test_is_state(self):
        self.assertEqual(is_state(("$q3",)), True)
        self.assertEqual(is_state(("$x3",)), False)
        self.assertEqual(is_state("$q3"), False)
        self.assertEqual(is_state("$"), False)
        self.assertEqual(is_state(32), False)

    def test_stream_ngrams(self):
        lines = ['the dog barked', 'why did the dog bark']
        expected = ['the dog', 'dog barked', 'why did', 'did the', 'the dog', 'dog bark']
        self.assertEqual(expected, list(stream_ngrams(lines, 2)))

    def test_stream_ngrams_numbers(self):
        lines = ['3 dogs barked', 'why didnt 4 dogs bark']
        expected = ['*NUM* dogs', 'dogs barked', 'why didnt', 'didnt *NUM*',
                    '*NUM* dogs', 'dogs bark']
        self.assertEqual(expected, list(stream_ngrams(lines, 2)))


if __name__ == "__main__":
    unittest.main()   