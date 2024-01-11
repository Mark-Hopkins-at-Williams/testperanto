import json


def generate_basic_event_rules(switches="000000"):
    s_switch = bool(switches[0])
    vp_switch = bool(switches[0])

    start_rules = ["($qstart $x1) -> ($qevent $x1)"]

    if not s_switch and not vp_switch:
        event_rules = ["($qevent (EVENT (inst $x1) (args (arg1 $x2)) $x4 $x5)) -> (S ($qarg0 $x2) ($qvb $x1 $x5) ($qvmods $x4))",
                       "($qevent (EVENT (inst $x1) (args (arg0 $x2) (arg1 $x3)) $x4 $x5)) -> (S ($qarg0 $x2) ($qarg1 $x3) ($qvb $x1 $x5) ($qvmods $x4))"]    
    elif not s_switch and vp_switch:
        event_rules = ["($qevent (EVENT (inst $x1) (args (arg1 $x2)) $x4 $x5)) -> (S ($qarg0 $x2) ($qvb $x1 $x5) ($qvmods $x4))",
                       "($qevent (EVENT (inst $x1) (args (arg0 $x2) (arg1 $x3)) $x4 $x5)) -> (S ($qarg0 $x2) ($qvb $x1 $x5) ($qarg1 $x3) ($qvmods $x4))"]    
    elif s_switch and not vp_switch:
        event_rules = ["($qevent (EVENT (inst $x1) (args (arg1 $x2)) $x4 $x5)) -> (S ($qvb $x1 $x5) ($qarg0 $x2) ($qvmods $x4))",
                       "($qevent (EVENT (inst $x1) (args (arg0 $x2) (arg1 $x3)) $x4 $x5)) -> (S ($qarg1 $x3) ($qvb $x1 $x5) ($qarg0 $x2) ($qvmods $x4))"]    
    else:
        event_rules = ["($qevent (EVENT (inst $x1) (args (arg1 $x2)) $x4 $x5)) -> (S ($qvb $x1 $x5) ($qarg0 $x2)  ($qvmods $x4))",
                       "($qevent (EVENT (inst $x1) (args (arg0 $x2) (arg1 $x3)) $x4 $x5)) -> (S ($qvb $x1 $x5) ($qarg1 $x3) ($qarg0 $x2) ($qvmods $x4))"]
 
    # "to say the wrong thing"
    ccomp_rules = ["($qccomp (EVENT (inst $x1) (args (arg1 $x2)) $x4 $x5)) -> (S ($qvbinf $x1 $x5) ($qarg1 $x2) ($qvmods $x4))"]
    # "that said the wrong thing"
    rcevent_rules = ["($qrcevent (EVENT (inst $x1) (args (arg1 $x2)) $x4 $x5)) -> (S ($qvb $x1 $x5) ($qarg1 $x2) ($qvmods $x4))"]
    # "because they said the wrong thing"
    subclause_rules = ["($qsubclause (EVENT (inst cause.0) (args (arg0 $x1)) $x2 $x3)) -> (S because ($qevent $x1))"]
    
    entity_arg_rules = ["($qarg0 (ENTITY $x1 $x2 $x3)) -> ($qentity (ENTITY $x1 $x2 $x3))",
                        "($qarg1 (ENTITY $x1 $x2 $x3)) -> ($qentity (ENTITY $x1 $x2 $x3))"]
    event_arg_rules = ["($qarg0 (EVENT $x1 $x2 $x3 $x4)) -> ($qccomp (EVENT $x1 $x2 $x3 $x4))",
                       "($qarg1 (EVENT $x1 $x2 $x3 $x4)) -> ($qccomp (EVENT $x1 $x2 $x3 $x4))"]
    
    entity_rules = ["($qentity (ENTITY (inst nn.$y1) $x2 $x3)) -> (NP ($qdt $x3) ($qnn nn.$y1 $x3) ($qnmods $x2))",
                    "($qentity (ENTITY (inst pron.$y1) $x2 $x3)) -> (NP ($qnn pron.$y1 $x3))"]
    
    determiner_rules = ["($qdt (props (count $x1) (def def) (case $x2))) -> (DT the)",
                        "($qdt (props (count sng) (def indef) (case $x2))) -> (DT a)",
                        "($qdt (props (count plu) (def indef) (case $x2))) -> (DT *eps*)"]
    
    noun_rules = ["($qnn nn.$y1 (props (count $x2) (def $x3) (case $x4))) -> (NN (@nn.en (STEM nn.$y1) (PERSON 3) (COUNT $x2) (CASE $x4)))",
                  "($qnn pron.$y1 (props (count $x2) (def $x3) (case $x4))) -> (NN (@pron.en (STEM pron) (PERSON $y1) (COUNT $x2) (CASE $x4)))"]
    
    nmods_rules = ["($qnmods -null-) -> (MOD *eps*)",
                   "($qnmods (mods (arg0-of $x1))) -> (RC that ($qrcevent $x1))"]
    
    verb_rules = ["($qvb vb.$y1 (props (tense $x1) (polarity $x4) (voice $x2) (count $x3) (person $x5))) -> (VB (@vb.en (STEM vb.$y1) (PERSON $x5) (COUNT $x3) (POLARITY $x4) (TENSE $x1) (VOICE $x2)))",
                  "($qvbinf vb.$y1 (props (tense $x1) (polarity $x4) (voice $x2) (count $x3) (person $x5))) -> ($qvb vb.$y1 (props (tense $x1) (polarity $x4) (voice $x2) (count inf) (person $x5)))"]
    
    vmods_rules = ["($qvmods -null-) -> (MOD *eps*)",
                   ["($qvmods (mods (location (ENTITY (inst nn.$y1) $x2 $x3)))) -> (PP (@prep.en (STEM prep.$z1)) ($qentity (ENTITY (inst nn.$y1) $x2 $x3)))", ["prep.loc.$y1"]],
                   ["($qvmods (mods (location (ENTITY (inst pron.$y1) $x2 $x3)))) -> (PP (@prep.$z2 (STEM prep.$z1)) ($qentity (ENTITY (inst pron.$y1) $x2 $x3)))", ["prep.loc.pron", "lang"]],
                   "($qvmods (mods (arg1-of $x1))) -> ($qsubclause $x1)",
                   "($qvmods (mods (time $x1))) -> (PP after ($qevent $x1))"]

    all_rules = (start_rules + event_rules + ccomp_rules + rcevent_rules + subclause_rules + entity_arg_rules
                 + event_arg_rules + entity_rules + determiner_rules + noun_rules + nmods_rules + verb_rules + vmods_rules)
    grammar = {"distributions": [], "rules": []}
    grammar["distributions"] = [{"name": "lang", "type": "uniform", "domain": ["en"]},
                                {"name": "count", "type": "uniform", "domain": ["sng", "plu"]},
                                {"name": "def", "type": "uniform", "domain": ["def", "indef"]},
                                {"name": "prep.loc", "type": "uniform", "domain": [1, 2, 3, 4, 5, 6]},
                                {"name": "prep.loc.$y0", "type": "pyor", "strength": 1, "discount": 0.2}]
    for rule in all_rules:
        if type(rule) is list:
            rule_json = {'rule': rule[0], 'zdists': rule[1]}
        elif type(rule) is str:
            rule_json = {'rule': rule}
        grammar["rules"].append(rule_json)

    return grammar

if __name__ == "__main__":
    g = generate_basic_event_rules()
    with open('amr2.json', 'w') as writer: 
        writer.write(json.dumps(g, indent=4))   

