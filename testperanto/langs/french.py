##
# french.py
# Code for generating French-like morphology.
##

from testperanto.morphology import Morpher


def conjugate_present(stem, person, count):
    if person == "1":
        return stem[:-2] + "e" if count == "sng" else stem[:-2] + "ons" 
    elif person == "2":
        return stem[:-2] + "es"  if count == "sng" else stem[:-2] + "ez" 
    elif person == "3":
        return stem[:-2] + "e"  if count == "sng" else stem[:-2] + "ent" 
    else:
        raise Exception(f"person not recognized: {person}")
    

def conjugate_be(person, count, tense):
    if tense == "present":
        if person == "1":
            return "suis" if count == "sng" else "sommes"
        elif person == "2":
            return "es" if count == "sng" else "êtes"
        elif person == "3":
            return "est" if count == "sng" else "sont"
        else:
            raise Exception(f"person not recognized: {person}")
    elif tense == "past":
        if person == "1":
            return "étais" if count == "sng" else "étions"
        elif person == "2":
            return "étais" if count == "sng" else "étiez"
        elif person == "3":
            return "était" if count == "sng" else "étaient"
        else:
            raise Exception(f"person not recognized: {person}")
    
def conjugate_have(person, count, tense):
    if tense == "present":
        if person == "1":
            return "ai" if count == "sng" else "avons"
        elif person == "2":
            return "as" if count == "sng" else "avez"
        elif person == "3":
            return "a" if count == "sng" else "ont"
        else:
            raise Exception(f"person not recognized: {person}")       
    elif tense == "past":
        if person == "1":
            return "avais" if count == "sng" else "avions"
        elif person == "2":
            return "avais" if count == "sng" else "aviez"
        elif person == "3":
            return "avait" if count == "sng" else "avaient"
        else:
            raise Exception(f"person not recognized: {person}")      
    
def add_s(stem):
    return stem + "s"

def add_ing(stem):
    return stem[:-1] + "ing" if stem[-1] == "e" else stem + "ing"

def add_ed(stem):
    return stem[:-2] + "é"
    
def conjugate_verb_passive(stem, person, count, tense, modal="will"):
    if tense == "present_simple":
        return conjugate_be(person, count, "present") + " " + add_ed(stem)
    elif tense == "past_simple":
        return conjugate_be(person, count, "past") + " " + add_ed(stem)
    else:
        raise Exception(f"tense not recognized: {tense}")     

def conjugate_verb_passive_neg(stem, person, count, tense, modal="will"):
    if tense == "present_simple":
        return "ne " + conjugate_be(person, count, "present") + " pas " + add_ed(stem)
    elif tense == "past_simple":
        return "ne " + conjugate_be(person, count, "past") + " pas " + add_ed(stem)
    else:
        raise Exception(f"tense not recognized: {tense}")     

def conjugate_verb(stem, person, count, tense, modal="will"):
    if tense == "present_simple":
        return conjugate_present(stem, person, count)
    elif tense == "past_simple":
        return conjugate_have(person, count, "present") + " " + add_ed(stem)
    else:
        raise Exception(f"tense not recognized: {tense}")     

def conjugate_verb_neg(stem, person, count, tense, modal="will"):
    if tense == "present_simple":
        return "ne " + conjugate_present(stem, person, count) + " pas"
    elif tense == "past_simple":
        return "n'" + conjugate_have(person, count, "present") + " pas " + add_ed(stem)
    else:
        raise Exception(f"tense not recognized: {tense}")     

def conjugate_infinitive(stem, tense, modal="will"):
    if tense == "present_simple":
        return stem
    elif tense == "past_simple":
        return "avoir " + add_ed(stem)
    else:
        raise Exception(f"tense not recognized: {tense}")  


class FrenchVerbMorpher(Morpher):

    def morph(self, stem, properties):
        person = properties['PERSON']
        count = properties['COUNT']
        tense = properties['TENSE']
        modal = properties['MODAL'] if tense.startswith("modal") else "will"
        if count == 'inf':
            return conjugate_infinitive(stem, tense, modal)
        elif properties['POLARITY'] == 'pos' and properties['VOICE'] == "active":
            return conjugate_verb(stem, person, count, tense, modal)
        elif properties['POLARITY'] == 'neg' and properties['VOICE'] == "active":
            return conjugate_verb_neg(stem, person, count, tense, modal)
        elif properties['POLARITY'] == 'pos' and properties['VOICE'] == 'passive':
            return conjugate_verb_passive(stem, person, count, tense, modal)
        elif properties['POLARITY'] == 'neg' and properties['VOICE'] == 'passive':
            return conjugate_verb_passive_neg(stem, person, count, tense, modal)
        else:
            raise Exception(f"Polarity not recognized: {properties['POLARITY']}")


class FrenchNounMorpher(Morpher):
    
    def morph(self, word, properties):
        suffix = "s" if properties['COUNT'] == "plu" else ""
        return word + suffix
        
        
class FrenchPronounMorpher(Morpher):
    
    def morph(self, word, properties):
        if properties['COUNT'] == "sng":
            if properties['PERSON'] == "1":
                return "je" if properties['CASE'] == "nom" else "moi"
            elif properties['PERSON'] == "2":
                return "tu" if properties['CASE'] == "nom" else "toi"
            elif properties['PERSON'] == "3":
                return "il" if properties['CASE'] == "nom" else "lui"
            else:
                raise Exception(f"Person not recognized: {properties['PERSON']}")
        elif properties['COUNT'] == "plu":
            if properties['PERSON'] == "1":
                return "nous"
            elif properties['PERSON'] == "2":
                return "vous"
            elif properties['PERSON'] == "3":
                return "ils" if properties['CASE'] == "nom" else "celles-ci"
            else:
                raise Exception(f"Person not recognized: {properties['PERSON']}")
        else:
            raise Exception(f"Count not recognized: {properties['COUNT']}")
        


