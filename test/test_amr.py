import unittest
from testperanto.amr import amr_str, amr_parse, text_stats
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
        amrs = text_stats("text.txt")
        print(amrs)
        
 

if __name__ == "__main__":
    unittest.main()   