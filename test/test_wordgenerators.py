##
# test_wordgenerators.py
# Unit tests for wordgenerators.py.
# $Revision: 32586 $
# $Date: 2012-04-17 14:26:33 -0700 (Tue, 17 Apr 2012) $
##

import unittest
import sys
from testperanto.trees import TreeNode
from testperanto.voicebox import VoiceboxFactory

class TestWordGenerators(unittest.TestCase):

    def test_voicebox_factory(self):
        vfactory = VoiceboxFactory()
        vbox = vfactory.create_voicebox("seuss")
        in_tree_str = "(S (NP (DT (@dt (DEF indef) (COUNT sng))) (JJ (@adj (STEM adj~0) (DEF indef) (COUNT sng))) (NN (@nn (STEM nn~0) (COUNT sng)))) (VB (@vb (STEM vb~0~0) (TENSE present) (DEF indef) (COUNT sng))) (NP (DT (@dt (DEF indef) (COUNT plu))) (JJ (@adj (STEM adj~1) (DEF indef) (COUNT plu))) (NN (@nn (STEM nn~1) (COUNT plu)))) (TO (@default to) (NP (DT (@dt (DEF def) (COUNT sng))) (JJ (@adj (STEM adj~2) (DEF def) (COUNT sng))) (NN (@nn (STEM nn~2) (COUNT sng))))))"
        in_tree = TreeNode.construct_from_str(in_tree_str)
        vbox.express(in_tree)



if __name__ == "__main__":
    unittest.main()   