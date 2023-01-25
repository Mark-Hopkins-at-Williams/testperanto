import unittest
from testperanto.amr import amr_str, amr_parse, text_stats, file_parse, english_amr_str
from testperanto.trees import TreeNode
from testperanto.config import init_transducer_cascade, run_transducer_cascade

class TestAmr(unittest.TestCase):

    def test_amr_str1(self):
        tree_str = "(ROOT (inst want-01) (arg0 (X (inst boy))) (arg1 (X (inst go-01))))"
        tree = TreeNode.from_str(tree_str)
        expected = "\n".join(["(want-01",
                              "   :arg0 (boy)",
                              "   :arg1 (go-01))"])
        self.assertEqual(amr_str(tree), expected)

    def test_amr_str2(self):
        tree_str = "(X (inst obligate-01) (arg1 (X (inst i))) (arg2 (X (inst grow-02) (arg1 (X (inst i))) (arg2 (X (inst old))))))"
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
    
    def test_clausal_complement(self):
        tree_str = "(X (inst vb.1) (arg0 (X (inst nn.2))) (arg1 (X (inst vb.5) (arg0 (X (inst nn.2))) (arg1 (X (inst nn.5))) (mods -null-))) (mods -null-))"
        tree = TreeNode.from_str(tree_str) 
        cascade = init_transducer_cascade(["examples/amr/english.json"], None, vbox_theme="goose")
        #output = run_transducer_cascade(cascade, start_state=f'{tree_str}')
        in_tree = TreeNode.from_str(f'($qstart {tree_str})')
        transducer = cascade[0]
        output = transducer.run(in_tree)
        print(in_tree)
        print(english_amr_str(output))
        

    def test_file_parse(self):
        str_to_test = "\n".join([
            '# ::id lpp_1943.292 ::date 2012-11-18T16:49:43 ::annotator ISI-AMR-05 ::preferred',
            '# ::snt " A sheep , "',
            '# ::zh 一只羊',
            '# ::save-date Sun Nov 18, 2012 ::file lpp_1943_292.txt',
            '(s / sheep)',
            '',
            '# ::id lpp_1943.293 ::date 2012-11-18T16:50:42 ::annotator ISI-AMR-05 ::preferred',
            '# ::snt I answered , " eats anything it finds in its reach . "',
            '# ::zh “它碰到什么吃什么。” 我回答。',
            '# ::save-date Thu Apr 18, 2013 ::file lpp_1943_293.txt',
            '(a / answer-01',
            '   :ARG0 (i / i)',
            '   :ARG1 (e / eat-01',
            '       :ARG1 (a2 / anything',
            '           :ARG1-of (f / find-01',
            '               :ARG0 (i2 / it)',
            '               :location (r / reach-03',
            '                   :ARG0 i2)))))'
        ])
        treeNodes = file_parse(str_to_test)
        expected_one = "\n".join([
            '(s/sheep)'
        ])
        self.assertEqual(amr_str(treeNodes[0]), expected_one)
        expected_two = "\n".join([
            '(a/answer-01',
            '   :ARG0 (i/i)',
            '   :ARG1 (e/eat-01',
            '      :ARG1 (a2/anything',
            '         :ARG1-of (f/find-01',
            '            :ARG0 (i2/it)',
            '            :location (r/reach-03',
            '               :ARG0 i2)))))'
        ])
        self.assertEqual(amr_str(treeNodes[1]), expected_two)

    def test_text_stats(self):
        amrs = text_stats("examples/amr/text.txt")
        
 

if __name__ == "__main__":
    unittest.main()   