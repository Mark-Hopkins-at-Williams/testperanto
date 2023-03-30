##
# test_voicebox.py
# Unit tests for voicebox.py.
##

import unittest
import sys
from testperanto.globals import DOT, EMPTY_STR
from testperanto.morphology import SuffixMorpher
from testperanto.trees import TreeNode
from testperanto.util import compound
from testperanto.voicebox import lookup_voicebox_theme
from testperanto.voicebox import VerbatimVoicebox, ManagingVoicebox, MorphologyVoicebox
from testperanto.voicebox import read_preterminal_tree, read_terminal_structure
from testperanto.wordgenerators import IteratingWordGenerator, PrefixSuffixWordGenerator


make_tree = TreeNode.from_str

class TestVoicebox(unittest.TestCase):

    def test_read_preterminal_tree(self):
        in_tree = make_tree('(DEF indef)')
        self.assertEqual(read_preterminal_tree(in_tree), ('DEF', 'indef'))

    def test_read_terminal_structure(self):
        ptree = make_tree('(@dt (DEF indef) (COUNT sng))')
        self.assertEqual(read_terminal_structure(ptree), {'DEF': 'indef', 'COUNT': 'sng'})
        ptree = make_tree('(@dt (DEF def) (COUNT plu) (PERSON 3))')
        self.assertEqual(read_terminal_structure(ptree), {'DEF': 'def', 'COUNT': 'plu', 'PERSON': '3'})

    def test_managing_vbox1(self):
        vbox = ManagingVoicebox()
        dt_morph = SuffixMorpher(property_names=('COUNT', 'DEF'),
                                 suffix_map={('sng', 'def'): 'the',
                                             ('plu', 'def'): 'these',
                                             ('sng', 'indef'): 'a',
                                             ('plu', 'indef'): EMPTY_STR})
        vbox.delegate('dt', MorphologyVoicebox(None, [dt_morph]))
        ptree = make_tree('(@dt (DEF def) (COUNT sng))')
        self.assertEqual(vbox.run(ptree), make_tree("(X the)"))

    def test_managing_vbox2(self):
        vbox = ManagingVoicebox()
        dt_morph = SuffixMorpher(property_names=('COUNT', 'DEF'),
                                 suffix_map={('sng', 'def'): 'the',
                                             ('plu', 'def'): 'these',
                                             ('sng', 'indef'): 'a',
                                             ('plu', 'indef'): EMPTY_STR})
        vbox.delegate('dt', MorphologyVoicebox(None, [dt_morph]))
        ptree = make_tree('(TOP (@dt (DEF def) (COUNT sng)) (@dt (DEF indef) (COUNT sng)))')
        self.assertEqual(vbox.run(ptree), make_tree("(TOP (X the) (X a))"))

    def test_morphology_vbox(self):
        word_generator = IteratingWordGenerator(['cat', 'dog', 'cookie'], None)
        morph1 = SuffixMorpher(property_names=('COUNT',),
                               suffix_map={('sng',): '', ('plu',): 's'})
        vbox = MorphologyVoicebox(word_generator, [morph1])
        in_tree = make_tree(f'(@nn (STEM n{DOT}0) (DEF indef) (COUNT plu))')
        self.assertEqual(vbox.run(in_tree), make_tree("(X cats)"))

    def test_morphology_vbox_cascade(self):
        morph1 = SuffixMorpher(property_names=('GENDER',),
                               suffix_map={('m',): 'eau', ('f',): 'ette'})
        morph2 = SuffixMorpher(property_names=('COUNT',),
                               suffix_map={('sng',): '', ('plu',): 's'})
        word_generator = IteratingWordGenerator(['tabl', 'lun', 'pip'], None)
        vbox = MorphologyVoicebox(word_generator, [morph1, morph2])
        in_tree = make_tree('(@nn (STEM {}) (GENDER f) (COUNT plu))'
                                              .format(compound(["n", "10"])))
        self.assertEqual(vbox.run(in_tree), make_tree("(X tablettes)"))
        in_tree = make_tree('(@nn (STEM {}) (GENDER m) (COUNT plu))'
                                              .format(compound(["n", "10"])))
        self.assertEqual(vbox.run(in_tree), make_tree("(X tableaus)"))
        in_tree = make_tree('(@nn (STEM {}) (GENDER f) (COUNT sng))'
                                              .format(compound(["n", "212"])))
        self.assertEqual(vbox.run(in_tree), make_tree("(X lunette)"))
        in_tree = make_tree('(@nn (STEM {}) (GENDER m) (COUNT sng))'
                                              .format(compound(["n", "10"])))
        self.assertEqual(vbox.run(in_tree), make_tree("(X tableau)"))


    def test_verbatim_vbox(self):
        vbox = VerbatimVoicebox()
        in_tree = make_tree('(@verbatim hello)')
        self.assertEqual(vbox.run(in_tree), make_tree("hello"))


    def test_english_det_vbox(self):
        vbox = lookup_voicebox_theme("english").init_vbox()
        in_tree_str = make_tree('(@dt (DEF indef) (COUNT sng))')
        self.assertEqual(vbox.run(in_tree_str), make_tree('(X a)'))
        in_tree_str = make_tree('(@dt (DEF def) (COUNT sng))')
        self.assertEqual(vbox.run(in_tree_str), make_tree('(X the)'))
        in_tree_str = make_tree('(@dt (DEF indef) (COUNT plu))')
        self.assertEqual(vbox.run(in_tree_str), make_tree(f'(X {EMPTY_STR})'))
        in_tree_str = make_tree('(@dt (DEF def) (COUNT plu))')
        self.assertEqual(vbox.run(in_tree_str), make_tree('(X these)'))

    def test_tree_voicebox(self):
        vbox = lookup_voicebox_theme("english").init_vbox()
        in_tree_str = "(S (NP (JJ (@adj (STEM {}) (DEF indef) (COUNT sng))) (NN (@nn (STEM {}) (COUNT sng)))))".format(compound(["adj", "0"]), compound(["n", "0"]))
        in_tree = make_tree(in_tree_str)
        # print(vbox.run(in_tree))


if __name__ == "__main__":
    unittest.main()   