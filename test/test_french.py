##
# test_french.py
# Unit tests for french.py.
##

import unittest
import sys
from testperanto.french import FrenchVerbMorpher

class TestFrench(unittest.TestCase):

    def test_present_simple_active(self):
        morpher = FrenchVerbMorpher()
        properties = {'PERSON': '1', 
                      'COUNT': 'sng', 
                      'VOICE': 'active', 
                      'TENSE': 'present_simple', 
                      'POLARITY': 'pos'}
        self.assertEqual(morpher.morph('marcher', properties), 'marche')
        properties['PERSON'] = '2'
        self.assertEqual(morpher.morph('marcher', properties), 'marches')
        properties['PERSON'] = '3'
        self.assertEqual(morpher.morph('marcher', properties), 'marche')
        properties['COUNT'] = 'plu'
        properties['PERSON'] = '1'
        self.assertEqual(morpher.morph('marcher', properties), 'marchons')
        properties['PERSON'] = '2'
        self.assertEqual(morpher.morph('marcher', properties), 'marchez')
        properties['PERSON'] = '3'
        self.assertEqual(morpher.morph('marcher', properties), 'marchent')
        properties['POLARITY'] = 'neg'
        properties['COUNT'] = 'sng'
        properties['PERSON'] = '1'
        self.assertEqual(morpher.morph('marcher', properties), 'ne marche pas')
        properties['PERSON'] = '2'
        self.assertEqual(morpher.morph('marcher', properties), 'ne marches pas')
        properties['PERSON'] = '3'
        self.assertEqual(morpher.morph('marcher', properties), 'ne marche pas')
        properties['COUNT'] = 'plu'
        properties['PERSON'] = '1'
        self.assertEqual(morpher.morph('marcher', properties), 'ne marchons pas')
        properties['PERSON'] = '2'
        self.assertEqual(morpher.morph('marcher', properties), 'ne marchez pas')
        properties['PERSON'] = '3'
        self.assertEqual(morpher.morph('marcher', properties), 'ne marchent pas')

    def test_past_simple_active(self):
        morpher = FrenchVerbMorpher()
        properties = {'PERSON': '1', 
                      'COUNT': 'sng', 
                      'VOICE': 'active', 
                      'TENSE': 'past_simple', 
                      'POLARITY': 'pos'}
        self.assertEqual(morpher.morph('marcher', properties), "ai marché")
        properties['PERSON'] = '2'
        self.assertEqual(morpher.morph('marcher', properties), 'as marché')
        properties['PERSON'] = '3'
        self.assertEqual(morpher.morph('marcher', properties), 'a marché')
        properties['COUNT'] = 'plu'
        properties['PERSON'] = '1'
        self.assertEqual(morpher.morph('marcher', properties), 'avons marché')
        properties['PERSON'] = '2'
        self.assertEqual(morpher.morph('marcher', properties), 'avez marché')
        properties['PERSON'] = '3'
        self.assertEqual(morpher.morph('marcher', properties), 'ont marché')
        properties['POLARITY'] = 'neg'
        properties['COUNT'] = 'sng'
        properties['PERSON'] = '1'
        self.assertEqual(morpher.morph('marcher', properties), "n'ai pas marché")
        properties['PERSON'] = '2'
        self.assertEqual(morpher.morph('marcher', properties), "n'as pas marché")
        properties['PERSON'] = '3'
        self.assertEqual(morpher.morph('marcher', properties), "n'a pas marché")
        properties['COUNT'] = 'plu'
        properties['PERSON'] = '1'
        self.assertEqual(morpher.morph('marcher', properties), "n'avons pas marché")
        properties['PERSON'] = '2'
        self.assertEqual(morpher.morph('marcher', properties), "n'avez pas marché")
        properties['PERSON'] = '3'
        self.assertEqual(morpher.morph('marcher', properties), "n'ont pas marché")

if __name__ == "__main__":
    unittest.main()   