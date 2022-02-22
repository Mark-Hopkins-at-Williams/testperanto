##
# distmanager.py
# A structure that manages the distributions of a weighted grammar macro.
# $Author: mhopkins $
# $Revision: 34246 $
# $Date: 2012-05-23 18:26:36 -0700 (Wed, 23 May 2012) $
##

from abc import ABC, abstractmethod
from testperanto import distributions
from testperanto.substitutions import SymbolSubstitution
from testperanto.config import CONSTRUCTORS


class DistributionManager:
    
    def __init__(self, factories=None, generate_consecutive_ids=False):
        if factories is None:
            self.factories = dict()
        else:
            self.factories = factories
        self.distributions = dict()
        self.generate_consecutive_ids = generate_consecutive_ids

    def add_factory(self, key, factory):
        self.factories[key] = factory
        
    def get(self, key, expansion=SymbolSubstitution()):
        try:
            factory = self.factories[key]
        except KeyError:
            raise KeyError('no matches for distribution: {}'.format(key))
        ground_key = expansion.substitute_in_sequence(key)
        if ground_key not in self.distributions:
            if len(key) > 1:
                base = self.get(key[:-1], expansion)
            else:
                base = distributions.IdGenerator(self.generate_consecutive_ids)
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


class PitmanYorDistributionFactory(DistributionFactory):
    def __init__(self, discount, strength):
        super().__init__()
        self.strength = strength
        self.discount = discount
    
    def instantiate_dist(self, base):
        return distributions.PitmanYorProcess(base, self.discount, self.strength)


CONSTRUCTORS['pyor'] = PitmanYorDistributionFactory
CONSTRUCTORS['uniform'] = UniformDistributionFactory
CONSTRUCTORS['categorical'] = CategoricalDistributionFactory
