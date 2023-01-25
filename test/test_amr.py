import unittest
from testperanto.amr import amr_str, amr_parse, text_stats, english_amr_str
from testperanto.config import init_transducer_cascade, run_transducer_cascade
from testperanto.trees import TreeNode

class TestAmr(unittest.TestCase):

    def test_amr_str1(self):
        tree_str = "(ROOT (inst want-01) (arg0 (X (inst boy))) (arg1 (X (inst go-01))))"
        tree = TreeNode.from_str(tree_str)
        expected = "\n".join(["(want-01",
                              "   :arg0 (boy)",
                              "   :arg1 (go-01))"])
        self.assertEqual(amr_str(tree), expected)

    def test_amr_str2(self):
        tree_str = "(ROOT (inst obligate-01) (arg1 (X (inst i))) (arg2 (X (inst grow-02) (arg1 (X (inst i))) (arg2 (X (inst old))))))"
        tree = TreeNode.from_str(tree_str)
        expected = "\n".join(["(obligate-01",
                              "   :arg1 (i)",
                              "   :arg2 (grow-02",
                              "      :arg1 (i)",
                              "      :arg2 (old)))"])
        self.assertEqual(amr_str(tree), expected)        
    
    def test_amr_str3(self):
        tree_str = "(ROOT (inst look-01) (arg0 (X (inst i))) (arg2 (X (inst around) (arg0 (X (inst i))) (arg3 (X (inst all))))) (arg4 (X (inst careful))))"
        tree = TreeNode.from_str(tree_str)
        expected = "\n".join(["(look-01",
                              "   :arg0 (i)", 
                              "   :arg2 (around",
                              "      :arg0 (i)",
                              "      :arg3 (all))",
                              "   :arg4 (careful))"])
        self.assertEqual(amr_str(tree), expected)  

    def test_amr_english_str1(self):
        tree_str = "(Y (PP in the (INST (NN (@nn (STEM nn.$y1) (PERSON 3) (COUNT sng) (DEF def) (TENSE present))))) , (ARG0 (INST (NN (@nn (STEM nn.1019611202) (PERSON 3) (COUNT sng) (DEF def) (TENSE present))))) (INST (VB (@vb (STEM vb.2121862795) (PERSON 3) (COUNT sng) (DEF def) (TENSE present)))) (ARG1 (INST (NN (@nn (STEM nn.3589675348) (PERSON 3) (COUNT sng) (DEF def) (TENSE present))))) (ARG2 (INST (NN (@nn (STEM nn.3122889036) (PERSON 3) (COUNT sng) (DEF def) (TENSE present))))))" 
        tree = TreeNode.from_str(tree_str)
        expected = "\n".join([
            "(",
            "   :PP (in the",
            "      :INST (",
            "         :NN (@nn (",
            "            :STEM nn.$y1",
            "            :PERSON 3",
            "            :COUNT sng",
            "            :DEF def",
            "            :TENSE present))))",
            "   :ARG0 (",
            "      :INST (",
            "         :NN (@nn (",
            "            :STEM nn.1019611202",
            "            :PERSON 3",
            "            :COUNT sng",
            "            :DEF def",
            "            :TENSE present))))",
            "   :INST (",
            "      :VB (@vb (",
            "         :STEM vb.2121862795",
            "         :PERSON 3",
            "         :COUNT sng",
            "         :DEF def",
            "         :TENSE present)))",
            "   :ARG1 (",
            "      :INST (",
            "         :NN (@nn (",
            "            :STEM nn.3589675348",
            "            :PERSON 3",
            "            :COUNT sng",
            "            :DEF def",
            "            :TENSE present))))",
            "   :ARG2 (",
            "      :INST (",
            "         :NN (@nn (",
            "            :STEM nn.3122889036",
            "            :PERSON 3",
            "            :COUNT sng",
            "            :DEF def",
            "            :TENSE present)))))"
        ])
        # print(english_amr_str(tree))
        self.maxDiff = None
        self.assertEqual(english_amr_str(tree), expected)


    def test_amr_parse(self):
        tree_str = '\n'.join([  '# ::id lpp_1943.295 ::date 2012-11-18T16:56:51 ::annotator ISI-AMR-05 ::preferred',
                                '# ::snt " Yes , even flowers that have thorns . ',
                                '# ::zh “有刺的也吃！”',
                                '# ::save-date Tue Apr 23, 2013 ::file lpp_1943_295.txt',
                                '(f / flower',
                                '      :mod (e / even)',
                                '      :ARG0-of (h / have-03',
                                '            :ARG1 (t / thorn)))"'])
        tree = amr_parse(tree_str)
        expected = "\n".join(["(f/flower",
                              "   :mod (e/even)",
                              "   :ARG0-of (h/have-03",
                              "      :ARG1 (t/thorn)))"])
        self.assertEqual(amr_str(tree), expected)
    
    def test_text_stats(self):
        amrs = text_stats("examples/amr/text.txt")
        
 

if __name__ == "__main__":
    unittest.main()   