##
# test_german.py
# Unit tests for german.py.
##

import unittest
import sys
from testperanto.german import GermanVerbMorpher

class TestFrench(unittest.TestCase):

    def test_present_simple_active(self):
        morpher = GermanVerbMorpher()
        properties = {'PERSON': '1', 
                      'COUNT': 'sng', 
                      'VOICE': 'active', 
                      'TENSE': 'present_simple', 
                      'POLARITY': 'pos'}
        self.assertEqual(morpher.morph('machen', properties), 'mache')
        properties['PERSON'] = '2'
        self.assertEqual(morpher.morph('machen', properties), 'machst')
        properties['PERSON'] = '3'
        self.assertEqual(morpher.morph('machen', properties), 'macht')
        properties['COUNT'] = 'plu'
        properties['PERSON'] = '1'
        self.assertEqual(morpher.morph('machen', properties), 'machen')
        properties['PERSON'] = '2'
        self.assertEqual(morpher.morph('machen', properties), 'macht')
        properties['PERSON'] = '3'
        self.assertEqual(morpher.morph('machen', properties), 'machen')
        properties['POLARITY'] = 'neg'
        properties['COUNT'] = 'sng'
        properties['PERSON'] = '1'
        self.assertEqual(morpher.morph('machen', properties), 'mache nicht')
        properties['PERSON'] = '2'
        self.assertEqual(morpher.morph('machen', properties), 'machst nicht')
        properties['PERSON'] = '3'
        self.assertEqual(morpher.morph('machen', properties), 'macht nicht')
        properties['COUNT'] = 'plu'
        properties['PERSON'] = '1'
        self.assertEqual(morpher.morph('machen', properties), 'machen nicht')
        properties['PERSON'] = '2'
        self.assertEqual(morpher.morph('machen', properties), 'macht nicht')
        properties['PERSON'] = '3'
        self.assertEqual(morpher.morph('machen', properties), 'machen nicht')

    def test_past_simple_active(self):
        morpher = GermanVerbMorpher()
        properties = {'PERSON': '1', 
                      'COUNT': 'sng', 
                      'VOICE': 'active', 
                      'TENSE': 'past_simple', 
                      'POLARITY': 'pos'}
        self.assertEqual(morpher.morph('machen', properties), "habe gemacht")
        properties['PERSON'] = '2'
        self.assertEqual(morpher.morph('machen', properties), 'hast gemacht')
        properties['PERSON'] = '3'
        self.assertEqual(morpher.morph('machen', properties), 'hat gemacht')
        properties['COUNT'] = 'plu'
        properties['PERSON'] = '1'
        self.assertEqual(morpher.morph('machen', properties), 'haben gemacht')
        properties['PERSON'] = '2'
        self.assertEqual(morpher.morph('machen', properties), 'habt gemacht')
        properties['PERSON'] = '3'
        self.assertEqual(morpher.morph('machen', properties), 'haben gemacht')
        properties['POLARITY'] = 'neg'
        properties['COUNT'] = 'sng'
        properties['PERSON'] = '1'
        self.assertEqual(morpher.morph('machen', properties), "habe nicht gemacht")
        properties['PERSON'] = '2'
        self.assertEqual(morpher.morph('machen', properties), "hast nicht gemacht")
        properties['PERSON'] = '3'
        self.assertEqual(morpher.morph('machen', properties), "hat nicht gemacht")
        properties['COUNT'] = 'plu'
        properties['PERSON'] = '1'
        self.assertEqual(morpher.morph('machen', properties), "haben nicht gemacht")
        properties['PERSON'] = '2'
        self.assertEqual(morpher.morph('machen', properties), "habt nicht gemacht")
        properties['PERSON'] = '3'
        self.assertEqual(morpher.morph('machen', properties), "haben nicht gemacht")

    def test_present_simple_passive(self):
        morpher = GermanVerbMorpher()
        properties = {'PERSON': '1', 
                      'COUNT': 'sng', 
                      'VOICE': 'passive', 
                      'TENSE': 'present_simple', 
                      'POLARITY': 'pos'}
        self.assertEqual(morpher.morph('machen', properties), 'bin gemacht')
        properties['PERSON'] = '2'
        self.assertEqual(morpher.morph('machen', properties), 'bist gemacht')
        properties['PERSON'] = '3'
        self.assertEqual(morpher.morph('machen', properties), 'ist gemacht')
        properties['COUNT'] = 'plu'
        properties['PERSON'] = '1'
        self.assertEqual(morpher.morph('machen', properties), 'sind gemacht')
        properties['PERSON'] = '2'
        self.assertEqual(morpher.morph('machen', properties), 'seit gemacht')
        properties['PERSON'] = '3'
        self.assertEqual(morpher.morph('machen', properties), 'sind gemacht')
        properties['POLARITY'] = 'neg'
        properties['COUNT'] = 'sng'
        properties['PERSON'] = '1'
        self.assertEqual(morpher.morph('machen', properties), 'bin nicht gemacht')
        properties['PERSON'] = '2'
        self.assertEqual(morpher.morph('machen', properties), 'bist nicht gemacht')
        properties['PERSON'] = '3'
        self.assertEqual(morpher.morph('machen', properties), 'ist nicht gemacht')
        properties['COUNT'] = 'plu'
        properties['PERSON'] = '1'
        self.assertEqual(morpher.morph('machen', properties), 'sind nicht gemacht')
        properties['PERSON'] = '2'
        self.assertEqual(morpher.morph('machen', properties), 'seit nicht gemacht')
        properties['PERSON'] = '3'
        self.assertEqual(morpher.morph('machen', properties), 'sind nicht gemacht')

    def test_past_simple_passive(self):
        morpher = GermanVerbMorpher()
        properties = {'PERSON': '1', 
                      'COUNT': 'sng', 
                      'VOICE': 'passive', 
                      'TENSE': 'past_simple', 
                      'POLARITY': 'pos'}
        self.assertEqual(morpher.morph('machen', properties), 'war gemacht')
        properties['PERSON'] = '2'
        self.assertEqual(morpher.morph('machen', properties), 'warst gemacht')
        properties['PERSON'] = '3'
        self.assertEqual(morpher.morph('machen', properties), 'war gemacht')
        properties['COUNT'] = 'plu'
        properties['PERSON'] = '1'
        self.assertEqual(morpher.morph('machen', properties), 'waren gemacht')
        properties['PERSON'] = '2'
        self.assertEqual(morpher.morph('machen', properties), 'wart gemacht')
        properties['PERSON'] = '3'
        self.assertEqual(morpher.morph('machen', properties), 'waren gemacht')
        properties['POLARITY'] = 'neg'
        properties['COUNT'] = 'sng'
        properties['PERSON'] = '1'
        self.assertEqual(morpher.morph('machen', properties), 'war nicht gemacht')
        properties['PERSON'] = '2'
        self.assertEqual(morpher.morph('machen', properties), 'warst nicht gemacht')
        properties['PERSON'] = '3'
        self.assertEqual(morpher.morph('machen', properties), 'war nicht gemacht')
        properties['COUNT'] = 'plu'
        properties['PERSON'] = '1'
        self.assertEqual(morpher.morph('machen', properties), 'waren nicht gemacht')
        properties['PERSON'] = '2'
        self.assertEqual(morpher.morph('machen', properties), 'wart nicht gemacht')
        properties['PERSON'] = '3'
        self.assertEqual(morpher.morph('machen', properties), 'waren nicht gemacht')


if __name__ == "__main__":
    unittest.main()   