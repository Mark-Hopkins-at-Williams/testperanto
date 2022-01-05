##
# test_morphology.py
# Unit tests for morphology.py.
# $Revision: 32586 $
# $Date: 2012-04-17 14:26:33 -0700 (Tue, 17 Apr 2012) $
##

import unittest
import sys
from testperanto.morphology import SuffixMorphologizer

class TestMorphology(unittest.TestCase):

    def test_suffix_morphologizer(self):
        morphologizer = SuffixMorphologizer(properties=('COUNT',),
                                            suffix_map={('sng',): '', ('plu',): 's'})
        result = morphologizer.mutate('apple', {'COUNT': 'plu', 'DEF': 'def'})
        self.assertEqual(result, 'apples')

    def test_suffix_morphologizer2(self):
        morphologizer = SuffixMorphologizer(properties=('GENDER', 'CASE'),
                                            suffix_map={('m', 'acc'): 'en',
                                                        ('f', 'acc'): 'e',
                                                        ('n', 'acc'): 'es',
                                                        ('m', 'dat'): 'em',
                                                        ('f', 'dat'): 'er',
                                                        ('n', 'dat'): 'em'})
        result = morphologizer.mutate('rot', {'GENDER': 'n', 'CASE': 'acc'})
        self.assertEqual(result, 'rotes')
        result = morphologizer.mutate('rot', {'GENDER': 'm', 'CASE': 'dat'})
        self.assertEqual(result, 'rotem')


if __name__ == "__main__":
    unittest.main()   