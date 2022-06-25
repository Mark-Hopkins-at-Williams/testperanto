##
# config.py
# Code for reading testperanto configuration files.
##

import json
from abc import ABC, abstractmethod
from copy import deepcopy
from tqdm import tqdm
from testperanto.globals import EMPTY_STR
from testperanto.voicebox import lookup_voicebox_theme
from testperanto.transducer import TreeTransducer, run_transducer_cascade
from testperanto.distmanager import DistributionManager
from testperanto.rules import IndexedRuleSet

def configure_transducer(config, switching_code=None):
    """Constructs a tree transducer from a configuration dictionary.

    Parameters
    ----------
    config : dict
        The configuration dictionary
    switching_code : str
        Binary switch string to configure alternative versions of indexed rules.
    """
    config = deepcopy(config)
    if "distributions" not in config:
        config["distributions"] = []
    if "grammar" in config:
        config = rewrite_wrig_config(config)
    code = switching_code
    if code is not None:
        rules = []
        for indexed_rule in config['macros']:
            next_rule = {key: indexed_rule[key] for key in indexed_rule}
            if 'alt' in indexed_rule and 'switch' in indexed_rule and code[indexed_rule['switch']] == "1":
                next_rule['rule'] = next_rule['alt']
            next_rule = {key: next_rule[key] for key in next_rule if key not in ['alt', 'switch']}
            rules.append(next_rule)
        config = {"distributions": config["distributions"], "macros": rules}
    manager = DistributionManager.from_config(config)
    grammar = IndexedRuleSet.from_config(config, manager)
    return TreeTransducer(grammar)


def init_transducer_cascade(config_files, switching_code=None, vbox_theme="english"):
    """Initializes a transducer cascade from a sequence of JSON configurations.

    Parameters
    ----------
    config_files : list[str]
        The filenames containing the transducer configurations.
    switching_code=None : str
        A bitstring for which the ith element is 1 if the alternative indexed rule
        for rules with switch i is desired
    vbox_theme : str
        The voicebox theme to render the terminal structures of the output tree
        of the cascade

    Returns
    -------
    list[testperanto.transducer.TreeTransducer]
        The initialized tree transducer cascade
    """

    cascade = []
    for config_file in config_files:
        with open(config_file, 'r') as reader:
            config = json.load(reader)
            cascade.append(configure_transducer(config, switching_code))
    vbox = lookup_voicebox_theme(vbox_theme).init_vbox()
    cascade.append(vbox)
    return cascade


def generate_sentence(cascade, start_state):
    output = run_transducer_cascade(cascade, start_state)
    leaves = ['.'.join(leaf.get_label()) for leaf in output.get_leaves()]
    leaves = [leaf for leaf in leaves if leaf != EMPTY_STR]
    return ' '.join(leaves)


def generate_sentences(transducer, num_to_generate, start_state, vbox_theme="english"):
    vbox = lookup_voicebox_theme(vbox_theme).init_vbox()
    cascade = [transducer, vbox]
    start_state = rewrite_wrig_symbol(start_state)
    result = []
    for _ in tqdm(range(num_to_generate)):
        result.append(generate_sentence(cascade, start_state))
    return result


def rewrite_wrig_symbol(symbol):
    if symbol[0].isupper():
        symbol = '$q{}'.format(symbol.lower())
    return symbol


def rewrite_wrig_config(config):
    def split_wrig_rule_rhs(rhs):
        retval = []
        tokens = rhs.split()
        open_parens = 0
        next_token = []
        for token in tokens:
            next_token.append(token)
            open_parens += token.count('(')
            open_parens -= token.count(')')
            if open_parens == 0:
                retval.append(' '.join(next_token))
                next_token = []
        return retval

    def rewrite_wrig_rule_config(rule_config):
        try:
            rule = rule_config['rule']
            lhs, rhs = [x.strip() for x in rule.split("->")]
            lhs = rewrite_wrig_symbol(lhs)
            rhs = [rewrite_wrig_symbol(symbol) for symbol in split_wrig_rule_rhs(rhs)]
            result = {key: rule_config[key] for key in rule_config if key != "rule"}
            result['rule'] = '{} -> (X {})'.format(lhs, ' '.join(rhs))
        except Exception:
            raise Exception("Badly formed rule config: {}".format(rule_config))
        return result

    result = {key: config[key] for key in config if key != "grammar"}
    if "grammar" in config:
        rules = [rewrite_wrig_rule_config(rule) for rule in config["grammar"]]
        result["macros"] = rules
    return result


def init_wrig(config):
    return configure_transducer(config)

