##
# distfactories.py
# Representations and algorithms for distribution factories.
# $Author: mhopkins $
# $Revision: 34246 $
# $Date: 2012-05-23 18:26:36 -0700 (Wed, 23 May 2012) $
##

from abc import ABC, abstractmethod
from testperanto import distributions
from testperanto.substitutions import SymbolSubstitution
from testperanto.config import CONSTRUCTORS


class DistributionManager:
    
    def __init__(self, factories=None):
        super().__init__()
        if factories is None:
            self.factories = dict()
        else:
            self.factories = factories
        self.distributions = dict()

    def add_factory(self, key, factory):
        self.factories[key] = factory
        
    def get(self, key, expansion=SymbolSubstitution()):
        try:
            factory = self.factories[key]
        except KeyError:
            raise KeyError('no matches for distribution: {}'.format(key))
        ground_key = expansion.substitute_in_sequence(key)
        if ground_key not in self.distributions:
            base = distributions.IdGenerator()
            if len(ground_key) > 1:
                try:
                    base = self.get(ground_key[:-1], expansion)
                except KeyError:
                    pass
            self.distributions[ground_key] = factory.instantiate_dist(base)
        return self.distributions[ground_key]

    @staticmethod
    def from_config(config):
        factory_configs = []
        if 'distributions' in config:
            factory_configs = config['distributions']
        factories = dict()
        for fconfig in factory_configs:
            name = tuple(fconfig['name'].split('~'))
            type = fconfig['type']
            args = {key: fconfig[key] for key in fconfig if key not in ['name', 'type']}
            factories[name] = CONSTRUCTORS[type](**args)
        return DistributionManager(factories)


class DistributionFactory(ABC):

    @abstractmethod
    def instantiate_dist(self, base):
        ...


class UniformDistributionFactory(DistributionFactory):
    def __init__(self, domain):
        super().__init__()
        self.domain = domain
    
    def instantiate_dist(self, base):
        return distributions.UniformDistribution(self.domain)


class CategoricalDistributionFactory(DistributionFactory):
    def __init__(self, domain, weights):
        super().__init__()
        self.domain = domain
        self.weights = weights

    def instantiate_dist(self, base):
        return distributions.CategoricalDistribution(self.weights, self.domain)


class GEMDistributionFactory(DistributionFactory):
    def __init__(self, concentration):
        super().__init__()
        self.concentration = concentration
    
    def instantiate_dist(self, base):
        return distributions.GEMDistribution(self.concentration)


class CRPDistributionFactory(DistributionFactory):
    def __init__(self, concentration):
        super().__init__()
        self.concentration = concentration
    
    def instantiate_dist(self, base):
        return distributions.CRPDistribution(base, self.concentration)


class PitmanYorDistributionFactory(DistributionFactory):
    def __init__(self, discount, strength):
        super().__init__()
        self.strength = strength
        self.discount = discount
    
    def instantiate_dist(self, base):
        return distributions.PitmanYorProcess(base, self.discount, self.strength)


CONSTRUCTORS['gem'] = GEMDistributionFactory
CONSTRUCTORS['crp'] = CRPDistributionFactory
CONSTRUCTORS['pyor'] = PitmanYorDistributionFactory
CONSTRUCTORS['uniform'] = UniformDistributionFactory
CONSTRUCTORS['categorical'] = CategoricalDistributionFactory
