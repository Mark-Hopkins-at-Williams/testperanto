##
# nahuatl.py
# Code for generating Nahuatl-like morphology.
##

from testperanto.morphology import Morpher
from testperanto.globals import EMPTY_STR


def conjugate_verb(stem, subject, object, tense, polarity):
    subject_prefix = {"1.sng": "ni", "2.sng": "ti", "3.sng": "", 
                      "1.plu": "ti", "2.plu": "in", "3.plu": ""}
    if tense == "present_simple":
        tense_suffix = {"1.sng": "", "2.sng": "", "3.sng": "", 
                        "1.plu": "h", "2.plu": "h", "3.plu": "h"}
    elif tense == "past_simple":
        tense_suffix = {"1.sng": "c", "2.sng": "c", "3.sng": "c", 
                        "1.plu": "queh", "2.plu": "queh", "3.plu": "queh"}
    else:
        raise Exception(f"tense not supported: {tense}")     
    object_prefix = {"1.sng": "nech", "2.sng": "mitz", "3.sng": "qui", 
                     "1.plu": "tech", "2.plu": "mech", "3.plu": "quin", "none": ""}
    polarity_prefix = {"pos": "", "neg": "ax"}
    return (polarity_prefix[polarity] + subject_prefix[subject] 
            + object_prefix[object] + stem + tense_suffix[subject])    

  
class NahuatlVerbMorpher(Morpher):

    def morph(self, stem, properties):
        subject = properties['SUBJECT']
        object = properties['OBJECT']
        tense = properties['TENSE']
        polarity = properties['POLARITY']
        if properties['VOICE'] == "active":
            return conjugate_verb(stem, subject, object, tense, polarity)
        #elif properties['POLARITY'] == 'neg' and properties['VOICE'] == "active":
        #    return conjugate_verb_neg(stem, subject, object, tense, position, modal)        
        else:
            raise Exception(f"Polarity not recognized: {properties['POLARITY']}")


class NahuatlNounMorpher(Morpher):
    
    def morph(self, word, properties):
        suffix = "en" if properties['COUNT'] == "plu" else ""
        return word + suffix
        
        
class NahuatlPronounMorpher(Morpher):
    
    def __init__(self):
        super().__init__()
        self.morph_map = {('sng', '1', 'nom'): "ni_",
                          ('sng', '2', 'nom'): "ti_",
                          ('sng', '3', 'nom'): "_",
                          ('plu', '1', 'nom'): "ti_",
                          ('plu', '2', 'nom'): "in_",
                          ('plu', '3', 'nom'): "_",
                          ('sng', '1', 'acc'): "???",
                          ('sng', '2', 'acc'): "???",
                          ('sng', '3', 'acc'): "???",
                          ('plu', '1', 'acc'): "???",
                          ('plu', '2', 'acc'): "???",
                          ('plu', '3', 'acc'): "???",
                          ('sng', '1', 'dat'): "???",
                          ('sng', '2', 'dat'): "???",
                          ('sng', '3', 'dat'): "???",
                          ('plu', '1', 'dat'): "???",
                          ('plu', '2', 'dat'): "???",
                          ('plu', '3', 'dat'): "???"} 

    def morph(self, word, properties):
        return self.morph_map[(properties['COUNT'], 
                               properties['PERSON'], 
                               properties['CASE'])]
        
        


