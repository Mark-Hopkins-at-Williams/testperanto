##
# test_voicebox.py
# Unit tests for voicebox.py.
# $Revision: 32586 $
# $Date: 2012-04-17 14:26:33 -0700 (Tue, 17 Apr 2012) $
##

import unittest
import sys
from testperanto.morphology import SuffixMorpher
from testperanto.trees import TreeNode
from testperanto.voicebox import VoiceboxFactory
from testperanto.voicebox import WordGeneratorVoicebox, EnglishDeterminerVoicebox
from testperanto.voicebox import MorphologyVoicebox
from testperanto.voicebox import read_preterminal_tree, read_property_tree
from testperanto.wordgenerators import WordGeneratorFactory
from testperanto.wordgenerators import IteratingWordGenerator, PrefixSuffixWordGenerator


def english_noun_generator():
    wordlist = ['cat', 'dog', 'cookie']
    wg_factory = WordGeneratorFactory()
    prefix_generator = wg_factory.create_generator('EnglishSyllables')
    suffix_generator = wg_factory.create_generator('EnglishConsonants')
    default_generator = PrefixSuffixWordGenerator(prefix_generator, suffix_generator)
    return IteratingWordGenerator(wordlist, default_generator)


def french_noun_generator():
    wordlist = ['tabl', 'lun', 'pip']
    wg_factory = WordGeneratorFactory()
    prefix_generator = wg_factory.create_generator('EnglishSyllables')
    suffix_generator = wg_factory.create_generator('EnglishConsonants')
    default_generator = PrefixSuffixWordGenerator(prefix_generator, suffix_generator)
    return IteratingWordGenerator(wordlist, default_generator)


class TestVoicebox(unittest.TestCase):

    def test_read_preterminal(self):
        in_tree = TreeNode.construct_from_str('(DEF indef)')
        self.assertEqual(read_preterminal_tree(in_tree), ('DEF', 'indef'))

    def test_read_property_tree(self):
        in_tree = TreeNode.construct_from_str('(@dt (DEF indef) (COUNT sng))')
        self.assertEqual(read_property_tree(in_tree), {'DEF': 'indef', 'COUNT': 'sng'})

    def test_morphology_vbox(self):
        morph1 = SuffixMorpher(properties=('COUNT',),
                                     suffix_map={('sng',): '', ('plu',): 's'})
        in_tree = TreeNode.construct_from_str('(@nn (STEM n~0) (DEF indef) (COUNT plu))')
        vbox = MorphologyVoicebox(english_noun_generator(), [morph1])
        self.assertEqual(vbox.express(in_tree), "cats")

    def test_morphology_vbox_cascade(self):
        morph1 = SuffixMorpher(properties=('GENDER',),
                                     suffix_map={('m',): 'eau', ('f',): 'ette'})
        morph2 = SuffixMorpher(properties=('COUNT',),
                                     suffix_map={('sng',): '', ('plu',): 's'})
        vbox = MorphologyVoicebox(french_noun_generator(), [morph1, morph2])
        in_tree = TreeNode.construct_from_str('(@nn (STEM n~10) (GENDER f) (COUNT plu))')
        self.assertEqual(vbox.express(in_tree), "tablettes")
        in_tree = TreeNode.construct_from_str('(@nn (STEM n~10) (GENDER m) (COUNT plu))')
        self.assertEqual(vbox.express(in_tree), "tableaus")
        in_tree = TreeNode.construct_from_str('(@nn (STEM n~212) (GENDER f) (COUNT sng))')
        self.assertEqual(vbox.express(in_tree), "lunette")
        in_tree = TreeNode.construct_from_str('(@nn (STEM n~10) (GENDER m) (COUNT sng))')
        self.assertEqual(vbox.express(in_tree), "tableau")

    def test_english_det_vbox(self):
        vbox = EnglishDeterminerVoicebox()
        in_tree_str = TreeNode.construct_from_str('(@dt (DEF indef) (COUNT sng))')
        self.assertEqual(vbox.express(in_tree_str), 'a')
        in_tree_str = TreeNode.construct_from_str('(@dt (DEF def) (COUNT sng))')
        self.assertEqual(vbox.express(in_tree_str), 'the')
        in_tree_str = TreeNode.construct_from_str('(@dt (DEF indef) (COUNT plu))')
        self.assertEqual(vbox.express(in_tree_str), '')
        in_tree_str = TreeNode.construct_from_str('(@dt (DEF def) (COUNT plu))')
        self.assertEqual(vbox.express(in_tree_str), 'these')

    def test_tree_voicebox(self):
        vfactory = VoiceboxFactory()
        vbox = vfactory.create_voicebox("seuss")
        in_tree_str = "(S (NP (JJ (@adj (STEM adj~0) (DEF indef) (COUNT sng))) (NN (@nn (STEM nn~0) (COUNT sng)))))"
        in_tree = TreeNode.construct_from_str(in_tree_str)
        print(vbox.express(in_tree))


if __name__ == "__main__":
    unittest.main()   