##
# test_nahuatl.py
# Unit tests for nahuatl.py.
##

import unittest
from testperanto.nahuatl import NahuatlVerbMorpher

class TestNahuatl(unittest.TestCase):

    def test_present_simple_active_intransitive(self):
        morpher = NahuatlVerbMorpher()
        properties = {'SUBJECT': '1.sng', 
                      'OBJECT': 'none', 
                      'VOICE': 'active', 
                      'TENSE': 'present_simple', 
                      'POLARITY': 'pos'}
        self.assertEqual(morpher.morph('choca', properties), 'nichoca')
        properties['SUBJECT'] = '2.sng'
        self.assertEqual(morpher.morph('choca', properties), 'tichoca')
        properties['SUBJECT'] = '3.sng'
        self.assertEqual(morpher.morph('choca', properties), 'choca')
        properties['SUBJECT'] = '1.plu'
        self.assertEqual(morpher.morph('choca', properties), 'tichocah')
        properties['SUBJECT'] = '2.plu'
        self.assertEqual(morpher.morph('choca', properties), 'inchocah')
        properties['SUBJECT'] = '3.plu'
        self.assertEqual(morpher.morph('choca', properties), 'chocah')

    def test_present_simple_active_transitive(self):
        morpher = NahuatlVerbMorpher()
        properties = {'SUBJECT': '1.sng', 
                      'OBJECT': '2.sng', 
                      'VOICE': 'active', 
                      'TENSE': 'present_simple', 
                      'POLARITY': 'pos'}
        self.assertEqual(morpher.morph('itta', properties), 'nimitzitta')
        properties['OBJECT'] = '3.sng'
        self.assertEqual(morpher.morph('itta', properties), 'niquiitta')
        properties['OBJECT'] = '1.plu'
        self.assertEqual(morpher.morph('itta', properties), 'nitechitta')
        properties['OBJECT'] = '2.plu'
        self.assertEqual(morpher.morph('itta', properties), 'nimechitta')        
        properties['OBJECT'] = '3.plu'
        self.assertEqual(morpher.morph('itta', properties), 'niquinitta')
        properties['SUBJECT'] = '2.sng'
        properties['OBJECT'] = '1.sng'
        self.assertEqual(morpher.morph('itta', properties), 'tinechitta')
        properties['OBJECT'] = '3.sng'
        self.assertEqual(morpher.morph('itta', properties), 'tiquiitta')
        properties['OBJECT'] = '1.plu'
        self.assertEqual(morpher.morph('itta', properties), 'titechitta')
        properties['OBJECT'] = '3.plu'        
        self.assertEqual(morpher.morph('itta', properties), 'tiquinitta')
        properties['SUBJECT'] = '3.sng'
        properties['OBJECT'] = '1.sng'
        self.assertEqual(morpher.morph('itta', properties), 'nechitta')
        properties['OBJECT'] = '2.sng'
        self.assertEqual(morpher.morph('itta', properties), 'mitzitta')
        properties['OBJECT'] = '3.sng'
        self.assertEqual(morpher.morph('itta', properties), 'quiitta')
        properties['OBJECT'] = '1.plu'
        self.assertEqual(morpher.morph('itta', properties), 'techitta')
        properties['OBJECT'] = '2.plu'        
        self.assertEqual(morpher.morph('itta', properties), 'mechitta')
        properties['OBJECT'] = '3.plu'        
        self.assertEqual(morpher.morph('itta', properties), 'quinitta')
        properties['SUBJECT'] = '1.plu'
        properties['OBJECT'] = '2.sng'
        self.assertEqual(morpher.morph('itta', properties), 'timitzittah')
        properties['OBJECT'] = '3.sng'
        self.assertEqual(morpher.morph('itta', properties), 'tiquiittah')
        properties['OBJECT'] = '1.plu'
        self.assertEqual(morpher.morph('itta', properties), 'titechittah')
        properties['OBJECT'] = '2.plu'
        self.assertEqual(morpher.morph('itta', properties), 'timechittah')        
        properties['OBJECT'] = '3.plu'
        self.assertEqual(morpher.morph('itta', properties), 'tiquinittah')
        properties['SUBJECT'] = '2.plu'
        properties['OBJECT'] = '1.sng'
        self.assertEqual(morpher.morph('itta', properties), 'innechittah')
        properties['OBJECT'] = '3.sng'
        self.assertEqual(morpher.morph('itta', properties), 'inquiittah')
        properties['OBJECT'] = '1.plu'
        self.assertEqual(morpher.morph('itta', properties), 'intechittah')
        properties['OBJECT'] = '3.plu'        
        self.assertEqual(morpher.morph('itta', properties), 'inquinittah')
        properties['SUBJECT'] = '3.plu'
        properties['OBJECT'] = '1.sng'
        self.assertEqual(morpher.morph('itta', properties), 'nechittah')
        properties['OBJECT'] = '2.sng'
        self.assertEqual(morpher.morph('itta', properties), 'mitzittah')
        properties['OBJECT'] = '3.sng'
        self.assertEqual(morpher.morph('itta', properties), 'quiittah')
        properties['OBJECT'] = '1.plu'
        self.assertEqual(morpher.morph('itta', properties), 'techittah')
        properties['OBJECT'] = '2.plu'        
        self.assertEqual(morpher.morph('itta', properties), 'mechittah')
        properties['OBJECT'] = '3.plu'        
        self.assertEqual(morpher.morph('itta', properties), 'quinittah')        

    def test_past_simple_active_intransitive(self):
        morpher = NahuatlVerbMorpher()
        properties = {'SUBJECT': '1.sng', 
                      'OBJECT': 'none', 
                      'VOICE': 'active', 
                      'TENSE': 'past_simple', 
                      'POLARITY': 'pos'}
        self.assertEqual(morpher.morph('choca', properties), 'nichocac')
        properties['SUBJECT'] = '2.sng'
        self.assertEqual(morpher.morph('choca', properties), 'tichocac')
        properties['SUBJECT'] = '3.sng'
        self.assertEqual(morpher.morph('choca', properties), 'chocac')
        properties['SUBJECT'] = '1.plu'
        self.assertEqual(morpher.morph('choca', properties), 'tichocaqueh')
        properties['SUBJECT'] = '2.plu'
        self.assertEqual(morpher.morph('choca', properties), 'inchocaqueh')
        properties['SUBJECT'] = '3.plu'
        self.assertEqual(morpher.morph('choca', properties), 'chocaqueh')

    def test_past_simple_active_transitive(self):
        morpher = NahuatlVerbMorpher()
        properties = {'SUBJECT': '1.sng', 
                      'OBJECT': '2.sng', 
                      'VOICE': 'active', 
                      'TENSE': 'past_simple', 
                      'POLARITY': 'pos'}
        self.assertEqual(morpher.morph('itta', properties), 'nimitzittac')
        properties['OBJECT'] = '3.sng'
        self.assertEqual(morpher.morph('itta', properties), 'niquiittac')
        properties['OBJECT'] = '1.plu'
        self.assertEqual(morpher.morph('itta', properties), 'nitechittac')
        properties['OBJECT'] = '2.plu'
        self.assertEqual(morpher.morph('itta', properties), 'nimechittac')        
        properties['OBJECT'] = '3.plu'
        self.assertEqual(morpher.morph('itta', properties), 'niquinittac')
        properties['SUBJECT'] = '2.sng'
        properties['OBJECT'] = '1.sng'
        self.assertEqual(morpher.morph('itta', properties), 'tinechittac')
        properties['OBJECT'] = '3.sng'
        self.assertEqual(morpher.morph('itta', properties), 'tiquiittac')
        properties['OBJECT'] = '1.plu'
        self.assertEqual(morpher.morph('itta', properties), 'titechittac')
        properties['OBJECT'] = '3.plu'        
        self.assertEqual(morpher.morph('itta', properties), 'tiquinittac')
        properties['SUBJECT'] = '3.sng'
        properties['OBJECT'] = '1.sng'
        self.assertEqual(morpher.morph('itta', properties), 'nechittac')
        properties['OBJECT'] = '2.sng'
        self.assertEqual(morpher.morph('itta', properties), 'mitzittac')
        properties['OBJECT'] = '3.sng'
        self.assertEqual(morpher.morph('itta', properties), 'quiittac')
        properties['OBJECT'] = '1.plu'
        self.assertEqual(morpher.morph('itta', properties), 'techittac')
        properties['OBJECT'] = '2.plu'        
        self.assertEqual(morpher.morph('itta', properties), 'mechittac')
        properties['OBJECT'] = '3.plu'        
        self.assertEqual(morpher.morph('itta', properties), 'quinittac')
        properties['SUBJECT'] = '1.plu'
        properties['OBJECT'] = '2.sng'
        self.assertEqual(morpher.morph('itta', properties), 'timitzittaqueh')
        properties['OBJECT'] = '3.sng'
        self.assertEqual(morpher.morph('itta', properties), 'tiquiittaqueh')
        properties['OBJECT'] = '1.plu'
        self.assertEqual(morpher.morph('itta', properties), 'titechittaqueh')
        properties['OBJECT'] = '2.plu'
        self.assertEqual(morpher.morph('itta', properties), 'timechittaqueh')        
        properties['OBJECT'] = '3.plu'
        self.assertEqual(morpher.morph('itta', properties), 'tiquinittaqueh')
        properties['SUBJECT'] = '2.plu'
        properties['OBJECT'] = '1.sng'
        self.assertEqual(morpher.morph('itta', properties), 'innechittaqueh')
        properties['OBJECT'] = '3.sng'
        self.assertEqual(morpher.morph('itta', properties), 'inquiittaqueh')
        properties['OBJECT'] = '1.plu'
        self.assertEqual(morpher.morph('itta', properties), 'intechittaqueh')
        properties['OBJECT'] = '3.plu'        
        self.assertEqual(morpher.morph('itta', properties), 'inquinittaqueh')
        properties['SUBJECT'] = '3.plu'
        properties['OBJECT'] = '1.sng'
        self.assertEqual(morpher.morph('itta', properties), 'nechittaqueh')
        properties['OBJECT'] = '2.sng'
        self.assertEqual(morpher.morph('itta', properties), 'mitzittaqueh')
        properties['OBJECT'] = '3.sng'
        self.assertEqual(morpher.morph('itta', properties), 'quiittaqueh')
        properties['OBJECT'] = '1.plu'
        self.assertEqual(morpher.morph('itta', properties), 'techittaqueh')
        properties['OBJECT'] = '2.plu'        
        self.assertEqual(morpher.morph('itta', properties), 'mechittaqueh')
        properties['OBJECT'] = '3.plu'        
        self.assertEqual(morpher.morph('itta', properties), 'quinittaqueh')     

    def test_selected_negation(self):
        morpher = NahuatlVerbMorpher()
        properties = {'SUBJECT': '1.sng', 
                      'OBJECT': 'none', 
                      'VOICE': 'active', 
                      'TENSE': 'present_simple', 
                      'POLARITY': 'neg'}  
        self.assertEqual(morpher.morph('choca', properties), 'axnichoca') 
        properties = {'SUBJECT': '2.plu', 
                      'OBJECT': '1.plu', 
                      'VOICE': 'active', 
                      'TENSE': 'past_simple', 
                      'POLARITY': 'neg'}  
        self.assertEqual(morpher.morph('itta', properties), 'axintechittaqueh') 

if __name__ == "__main__":
    unittest.main()   