##
# testdistmanager.py
# Unit tests for distmanager.py.
##


import unittest
import sys
from testperanto.distmanager import DistributionManager
from testperanto.distributions import PitmanYorProcess
from testperanto.util import compound
from testperanto.substitutions import SymbolSubstitution
import testperanto.examples

class TestDistributionFactory(unittest.TestCase):

    def setUp(self):
        self.manager = DistributionManager()
        self.manager.add_config(('a',), {'type': 'alternating'})
        self.manager.add_config(('b',), {'type': 'alternating'})
        self.manager.add_config(('c',), {'type': 'alternating'})
        self.manager.add_config(('a', '1'), {'type': 'averager'})
        self.manager.add_config(('a', '2'), {'type': 'averager'})
        self.manager.add_config(('b', '1'), {'type': 'averager'})
        self.manager.add_config(('a', '$y1'), {'type': 'averager'})
        self.manager.add_config(('a', '$y1', '$y2'), {'type': 'averager'})
        self.manager.add_config(('c', '1'), {'type': 'averager', 'base': ('a',)})
        self.manager.add_config(('c', '2'), {'type': 'averager'})

    def test_simple_factory(self):
        """Tests that a simple factory manager maintains state across calls to sample."""
        sub = SymbolSubstitution()
        sample = self.manager.get(('a',), sub).sample()
        self.assertEqual(sample, 0)                
        sample = self.manager.get(('a',), sub).sample()
        self.assertEqual(sample, 100)                
        sample = self.manager.get(('a',), sub).sample()
        self.assertEqual(sample, 0)                
        sample = self.manager.get(('a',), sub).sample()
        self.assertEqual(sample, 100)                
 
    def test_single_factory_with_base(self):
        """
        Tests that a single distribution factory maintains the same base
        distribution across calls to manager.get.

        """
        sub = SymbolSubstitution()
        sample = self.manager.get(('a', '1'), sub).sample()
        self.assertEqual(sample, 0)
        sample = self.manager.get(('a', '1'), sub).sample()
        self.assertEqual(sample, 60)
        sample = self.manager.get(('a', '1'), sub).sample()
        self.assertEqual(sample, 0)

    def test_specified_base1(self):
        sub = SymbolSubstitution()
        sample = self.manager.get(('a', '1'), sub).sample()
        self.assertEqual(sample, 0)
        sample = self.manager.get(('c', '1'), sub).sample()
        self.assertEqual(sample, 100)

    def test_specified_base2(self):
        sub = SymbolSubstitution()
        sample = self.manager.get(('a', '1'), sub).sample()
        self.assertEqual(sample, 0)
        sample = self.manager.get(('c', '2'), sub).sample()
        self.assertEqual(sample, 0)

    def test_multiple_factories_with_base(self):
        """
        Tests that multiple distribution factories with common base
        distributions exhibit correct behavior.

        """
        sub = SymbolSubstitution()
        sample = self.manager.get(('a', '1'), sub).sample()
        self.assertEqual(sample, 0)                
        sample = self.manager.get(('a', '2'), sub).sample()
        self.assertEqual(sample, 100)                
        sample = self.manager.get(('a', '2'), sub).sample()
        self.assertEqual(sample, 40)                
        sample = self.manager.get(('b', '1'), sub).sample()
        self.assertEqual(sample, 0)                
        sample = self.manager.get(('a', '2'), sub).sample()
        self.assertEqual(sample, 100)                
        sample = self.manager.get(('a', '1'), sub).sample()
        self.assertEqual(sample, 40)                
        sample = self.manager.get(('a', '1'), sub).sample()
        self.assertEqual(sample, 100)                
        sample = self.manager.get(('a', '1'), sub).sample()
        self.assertEqual(sample, 40)                
        sample = self.manager.get(('b', '1'), sub).sample()
        self.assertEqual(sample, 60)                
        sample = self.manager.get(('b', '1'), sub).sample()
        self.assertEqual(sample, 0)                
        sample = self.manager.get(('a', '2'), sub).sample()
        self.assertEqual(sample, 60)                
 
    def test_factory_family_with_base(self):
        sub_11_21 = SymbolSubstitution()
        sub_11_21.add_substitution('$y1', '11')
        sub_11_21.add_substitution('$y2', '21')
        sub_11_22 = SymbolSubstitution()
        sub_11_22.add_substitution('$y1', '11')
        sub_11_22.add_substitution('$y2', '22')
        sub_12_21 = SymbolSubstitution()
        sub_12_21.add_substitution('$y1', '12')
        sub_12_21.add_substitution('$y2', '21')
        sub_12_22 = SymbolSubstitution()
        sub_12_22.add_substitution('$y1', '12')
        sub_12_22.add_substitution('$y2', '22')
        sample = self.manager.get(('a','$y1','$y2'), sub_11_21).sample()
        self.assertEqual(sample, 0)                
        sample = self.manager.get(('a','$y1','$y2'), sub_12_21).sample()
        self.assertEqual(sample, 100)
        sample = self.manager.get(('a','$y1','$y2'), sub_12_22).sample()
        self.assertEqual(sample, 40)
        sample = self.manager.get(('a','$y1','$y2'), sub_12_21).sample()
        self.assertEqual(sample, 76)
        sample = self.manager.get(('a','$y1','$y2'), sub_11_22).sample()
        self.assertEqual(sample, 40)                
        sample = self.manager.get(('a','$y1','$y2'), sub_12_22).sample()
        self.assertEqual(sample, 36)                
        sample = self.manager.get(('a','$y1','$y2'), sub_11_21).sample()
        self.assertEqual(sample, 24)  
 
    def test_config(self):
        config = {'distributions': [{"name": "verb",
                                     "type": "pyor",
                                     "strength": 150,
                                     "discount": 0.35}]}
        manager = DistributionManager.from_config(config)
        dist = manager.get(('verb',))
        self.assertEqual(type(dist), PitmanYorProcess)
        self.assertEqual(dist.strength, 150)
        self.assertEqual(dist.discount, 0.35)


if __name__ == "__main__":
    unittest.main()   