from collections import defaultdict
from os.path import join
import spacy
from spacy.tokens import DocBin
import sys
from testperanto.globals import DOT, EMPTY_STR
from testperanto.analysis import plot_statistic, singleton_proportion, type_count_over_time
from testperanto.analysis import powers_of_2, multiples_of_1000
from testperanto.transducer import TreeTransducer, run_transducer_cascade
from testperanto.trees import TreeNode
from testperanto.voicebox import lookup_voicebox_theme
from testperanto.util import stream_lines
from tqdm import tqdm


def pyor_config(strength, discount):
    result = {"distributions": [
                {"name": "nn", "type": "pyor", "strength": strength, "discount": discount}
              ],
              "macros": [
                {"rule": f"$qstart -> $qnn{DOT}$z1", "zdists": ["nn"]},
                {"rule": f"$qnn{DOT}$y1 -> (NN (@nn (STEM nn{DOT}$y1) (COUNT sng)))"}
              ]}
    return result


def hpyor_config(base_strength, base_discount, strength, discount):
    result = {"distributions": [
                {"name": "nn", "type": "pyor", "strength": base_strength, "discount": base_discount},
                {"name": f"nn{DOT}$y1", "type": "pyor", "strength": strength, "discount": discount}
              ],
              "macros": [
                {"rule": f"$qstart -> $qnp{DOT}$z1", "zdists": ["nn"]},
                {"rule": f"$qnp{DOT}$y1 -> $qnn{DOT}$z1", "zdists": [f"nn{DOT}$y1"]},
                {"rule": f"$qnn{DOT}$y1 -> (NN (@nn (STEM nn{DOT}$y1) (COUNT sng)))"}
              ]}
    return result


def dual_config(head_strength, head_discount, base_strength, base_discount, strength, discount):
    result = {
        "distributions": [
            {"name": "nn", "type": "pyor", "strength": head_strength, "discount": head_discount},
            {"name": "adj", "type": "pyor", "strength": base_strength, "discount": base_discount},
            {"name": f"adj{DOT}$y1", "type": "pyor", "strength": strength, "discount": discount}
        ],
        "macros": [
            {"rule": f"$qstart -> $qnp{DOT}$z1", "zdists": ["nn"]},
            {"rule": f"$qnp{DOT}$y1 -> (NP (amod $qadj{DOT}$z1) (head $qnn{DOT}$y1))", "zdists": [f"adj{DOT}$y1"]},
            {"rule": f"$qnn{DOT}$y1 -> (NN (@nn (STEM nn{DOT}$y1) (COUNT sng)))"},
            {"rule": f"$qadj{DOT}$y1 -> (ADJ (@adj (STEM adj{DOT}$y1)))"}
        ]
    }
    return result


def dual_config_alt(head_strength, head_discount, s1, d1, s2, d2, s3, d3):
    result = {
        "distributions": [
            {"name": "vb", "type": "pyor", "strength": head_strength, "discount": head_discount},
            {"name": "nn", "type": "pyor", "strength": s1, "discount": d1},
            {"name": f"nn{DOT}subj", "type": "pyor", "strength": s2, "discount": d2},
            {"name": f"nn{DOT}subj{DOT}$y1", "type": "pyor", "strength": s3, "discount": d3}
        ],
        "macros": [
            {"rule": f"$qstart -> $qs{DOT}$z1", "zdists": ["vb"]},
            {"rule": f"$qs{DOT}$y1 -> (NP (nsubj $qnn{DOT}subj{DOT}$z1) (head $qvb{DOT}$y1))", "zdists": [f"nn{DOT}subj{DOT}$y1"]},
            {"rule": f"$qvb{DOT}$y1 -> (VB (@vb (STEM vb{DOT}$y1) (PERSON 3) (COUNT sng) (TENSE present)))"},
            {"rule": f"$qnn{DOT}subj{DOT}$y1 -> (NN (@nn (STEM nn{DOT}$y1) (COUNT sng)))"},
        ]
    }
    return result


def indep_config(head_strength, head_discount, base_strength, base_discount):
    result = {
        "distributions": [
            {"name": "nn", "type": "pyor", "strength": head_strength, "discount": head_discount},
            {"name": "adj", "type": "pyor", "strength": base_strength, "discount": base_discount},
        ],
        "macros": [
            {"rule": "$qstart -> $qnp", "zdists": []},
            {"rule": f"$qnp -> (NP (amod $qadj{DOT}$z1) (head $qnn{DOT}$z2))", "zdists": ["adj", "nn"]},
            {"rule": f"$qnn{DOT}$y1 -> (NN (@nn (STEM nn{DOT}$y1) (COUNT sng)))"},
            {"rule": f"$qadj{DOT}$y1 -> (ADJ (@adj (STEM adj{DOT}$y1)))"}
        ]
    }
    return result


def init_cascade(config_constructor, args):
    vbox = lookup_voicebox_theme("english")
    cascade = [TreeTransducer.from_config(config_constructor(*args)), vbox]
    return cascade


def token_stream(cascade, num_to_generate=500000):
    for _ in tqdm(range(num_to_generate)):
        output = run_transducer_cascade(cascade)
        leaves = [DOT.join(leaf) for leaf in output.get_leaves()]
        leaves = [leaf for leaf in leaves if leaf != EMPTY_STR]
        segment = ' '.join(leaves)
        yield segment


def generate(cascade, out_file, num_to_generate):
    with open(out_file, 'w') as writer:
        for token in token_stream(cascade, num_to_generate):
            writer.write('{}\n'.format(token))


def main():
    generate(init_cascade(dual_config_alt, (500.0, 0.4, 500.0, 0.4, 500.0, 0.4, 20.0, 0.6)),
            'data/dualalt.20.0_6.txt', 1000000)


if __name__ == '__main__':
    main()