##
# german.py
# Code for generating German-like morphology.
##

from testperanto.morphology import Morpher
from testperanto.globals import EMPTY_STR

def conjugate_present(stem, person, count):
    if person == "1":
        return stem[:-2] + "e" if count == "sng" else stem[:-2] + "en" 
    elif person == "2":
        return stem[:-2] + "st"  if count == "sng" else stem[:-2] + "t" 
    elif person == "3":
        return stem[:-2] + "t"  if count == "sng" else stem[:-2] + "en" 
    else:
        raise Exception(f"person not recognized: {person}")
    

def conjugate_be(person, count, tense):
    if tense == "present":
        if person == "1":
            return "bin" if count == "sng" else "sind"
        elif person == "2":
            return "bist" if count == "sng" else "seit"
        elif person == "3":
            return "ist" if count == "sng" else "sind"
        else:
            raise Exception(f"person not recognized: {person}")
    elif tense == "past":
        if person == "1":
            return "war" if count == "sng" else "waren"
        elif person == "2":
            return "warst" if count == "sng" else "wart"
        elif person == "3":
            return "war" if count == "sng" else "waren"
        else:
            raise Exception(f"person not recognized: {person}")
    

def conjugate_have(person, count, tense):
    if tense == "present":
        if person == "1":
            return "habe" if count == "sng" else "haben"
        elif person == "2":
            return "hast" if count == "sng" else "habt"
        elif person == "3":
            return "hat" if count == "sng" else "haben"
        else:
            raise Exception(f"person not recognized: {person}")       
    elif tense == "past":
        if person == "1":
            return "hatte" if count == "sng" else "hatten"
        elif person == "2":
            return "hattest" if count == "sng" else "hattet"
        elif person == "3":
            return "hatte" if count == "sng" else "hatten"
        else:
            raise Exception(f"person not recognized: {person}")      
    

def add_ed(stem):
    return "ge" + stem[:-2] + "t"


def conjugate_verb_passive(stem, person, count, tense, position, modal="will"):
    if tense == "present_simple":
        return conjugate_be(person, count, "present") + " " + add_ed(stem)
    elif tense == "past_simple":
        return conjugate_be(person, count, "past") + " " + add_ed(stem)
    else:
        raise Exception(f"tense not recognized: {tense}")     

def conjugate_verb_passive_neg(stem, person, count, tense, position, modal="will"):
    if tense == "present_simple":
        return conjugate_be(person, count, "present") + " nicht " + add_ed(stem)
    elif tense == "past_simple":
        return conjugate_be(person, count, "past") + " nicht " + add_ed(stem)
    else:
        raise Exception(f"tense not recognized: {tense}")     

def conjugate_verb(stem, person, count, tense, position, modal="will"):
    if tense == "present_simple" and position == "1":
        return conjugate_present(stem, person, count)
    elif tense == "present_simple" and position == "2":
        return EMPTY_STR
    elif tense == "past_simple"  and position == "1":
        return conjugate_have(person, count, "present")
    elif tense == "past_simple"  and position == "2":
        return add_ed(stem)
    else:
        raise Exception(f"tense/position not recognized: {tense}, {position}")     

def conjugate_verb_neg(stem, person, count, tense, position, modal="will"):
    if tense == "present_simple" and position == "1":
        return conjugate_present(stem, person, count) + " nicht"
    elif tense == "present_simple" and position == "2":
        return EMPTY_STR
    elif tense == "past_simple" and position == "1":
        return conjugate_have(person, count, "present")
    elif tense == "past_simple" and position == "2":
        return "nicht " + add_ed(stem)
    else:
        raise Exception(f"tense/position not recognized: {tense}, {position}")     

def conjugate_infinitive(stem, tense, modal="will"):
    if tense == "present_simple":
        return stem
    elif tense == "past_simple":
        return "haben " + add_ed(stem)
    else:
        raise Exception(f"tense not recognized: {tense}")  


class GermanVerbMorpher(Morpher):

    def morph(self, stem, properties):
        person, count = properties['SUBJECT'].split('.')
        tense = properties['TENSE']
        position = properties['POSITION']
        modal = properties['MODAL'] if tense.startswith("modal") else "will"
        if properties['POLARITY'] == 'pos' and properties['VOICE'] == "active":
            return conjugate_verb(stem, person, count, tense, position, modal)
        elif properties['POLARITY'] == 'neg' and properties['VOICE'] == "active":
            return conjugate_verb_neg(stem, person, count, tense, position, modal)
        elif properties['POLARITY'] == 'pos' and properties['VOICE'] == 'passive':
            return conjugate_verb_passive(stem, person, count, tense, position, modal)
        elif properties['POLARITY'] == 'neg' and properties['VOICE'] == 'passive':
            return conjugate_verb_passive_neg(stem, person, count, tense, position, modal)
        else:
            raise Exception(f"Polarity not recognized: {properties['POLARITY']}")


class GermanNounMorpher(Morpher):
    
    def morph(self, word, properties):
        suffix = "en" if properties['COUNT'] == "plu" else ""
        return word + suffix
        
        
class GermanPronounMorpher(Morpher):
    
    def __init__(self):
        super().__init__()
        self.morph_map = {('sng', '1', 'nom'): "ich",
                          ('sng', '2', 'nom'): "du",
                          ('sng', '3', 'nom'): "er",
                          ('plu', '1', 'nom'): "wir",
                          ('plu', '2', 'nom'): "ihr",
                          ('plu', '3', 'nom'): "sie",
                          ('sng', '1', 'acc'): "mich",
                          ('sng', '2', 'acc'): "dich",
                          ('sng', '3', 'acc'): "ihn",
                          ('plu', '1', 'acc'): "uns",
                          ('plu', '2', 'acc'): "ihn",
                          ('plu', '3', 'acc'): "sie",
                          ('sng', '1', 'dat'): "mir",
                          ('sng', '2', 'dat'): "dir",
                          ('sng', '3', 'dat'): "ihn",
                          ('plu', '1', 'dat'): "uns",
                          ('plu', '2', 'dat'): "ihn",
                          ('plu', '3', 'dat'): "ihnen"} 

    def morph(self, word, properties):
        return self.morph_map[(properties['COUNT'], 
                               properties['PERSON'], 
                               properties['CASE'])]
        
        


