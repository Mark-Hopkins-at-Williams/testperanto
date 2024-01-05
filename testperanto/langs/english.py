##
# morphology.py
# Tools for generating word morphology.
##

from abc import ABC, abstractmethod
import json
from testperanto.morphology import Morpher, SuffixMorpher
from random import shuffle


def conjugate_be(person, count, tense):
    if tense == "present":
        if person == "1":
            return "am" if count == "sng" else "are"
        elif person == "2":
            return "are"
        elif person == "3":
            return "is" if count == "sng" else "are"
        else:
            raise Exception(f"person not recognized: {person}")
    elif tense == "past":
        if count == "sng" and person in ["1", "3"]:
            return "was"
        else:
            return "were"
    
def conjugate_have(person, count, tense):
    if tense == "present":
        return "has" if person == "3" and count == "sng" else "have"            
    elif tense == "past":
        return "had"
    
def conjugate_do(person, count, tense):
    if tense == "present":
        return "does" if person == "3" and count == "sng" else "do"            
    elif tense == "past":
        return "did"
    
def add_s(stem):
    return stem + 's'

def add_ing(stem):
    return stem[:-1] + "ing" if stem[-1] == "e" else stem + "ing"

def add_ed(stem):
    return stem[:-1] + "ed" if stem[-1] == "e" else stem + "ed"
    
def conjugate_verb_passive(stem, person, count, tense, modal="will"):
    if tense == "present_simple":
        return conjugate_be(person, count, "present") + " " + add_ed(stem)
    elif tense == "past_simple":
        return conjugate_be(person, count, "past") + " " + add_ed(stem)
    else:
        raise Exception(f"tense not recognized: {tense}")     

def conjugate_verb_passive_neg(stem, person, count, tense, modal="will"):
    if tense == "present_simple":
        return conjugate_be(person, count, "present") + " not " + add_ed(stem)
    elif tense == "past_simple":
        return conjugate_be(person, count, "past") + " not " + add_ed(stem)
    else:
        raise Exception(f"tense not recognized: {tense}")     

def conjugate_verb(stem, person, count, tense, modal="will"):
    if tense == "present_simple":
        return add_s(stem) if person == "3" and count == "sng" else stem
    elif tense == "past_simple":
        return add_ed(stem)
    else:
        raise Exception(f"tense not recognized: {tense}")     

def conjugate_verb_neg(stem, person, count, tense, modal="will"):
    if tense == "present_simple":
        return conjugate_do(person, count, "present") + " not " + stem
    elif tense == "past_simple":
        return conjugate_do(person, count, "past") + " not " + stem
    else:
        raise Exception(f"tense not recognized: {tense}")     

def conjugate_infinitive(stem, tense, modal="will"):
    if tense == "present_simple":
        return "to " + stem
    elif tense == "past_simple":
        return "to have " + add_ed(stem)
    else:
        raise Exception(f"tense not recognized: {tense}")  


class EnglishVerbMorpher(Morpher):

    def morph(self, stem, properties):
        #person, count = properties['SUBJECT'].split('.')
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


class EnglishNounMorpher(Morpher):
    
    def morph(self, word, properties):
        suffix = "s" if properties['COUNT'] == "plu" else ""
        return word + suffix
    

class EnglishPronounMorpher(Morpher):
    
    def morph(self, word, properties):
        if properties['COUNT'] == "sng":
            if properties['PERSON'] == "1":
                return "I" if properties['CASE'] == "nom" else "me"
            elif properties['PERSON'] == "2":
                return "you"
            elif properties['PERSON'] == "3":
                return "he" if properties['CASE'] == "nom" else "him"
            else:
                raise Exception(f"Person not recognized: {properties['PERSON']}")
        elif properties['COUNT'] == "plu":
            if properties['PERSON'] == "1":
                return "we" if properties['CASE'] == "nom" else "us"
            elif properties['PERSON'] == "2":
                return "you"
            elif properties['PERSON'] == "3":
                return "they" if properties['CASE'] == "nom" else "them"
            else:
                raise Exception(f"Person not recognized: {properties['PERSON']}")
        else:
            raise Exception(f"Count not recognized: {properties['COUNT']}")
        
