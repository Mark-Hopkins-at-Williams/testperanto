##
# distmanager.py
# A structure that manages the distributions (weightings) of a WRIG.
##

from abc import ABC, abstractmethod
from testperanto import distributions
from testperanto.distributions import PitmanYorProcess, lookup_distribution
from testperanto.distributions import UniformDistribution, CategoricalDistribution
from testperanto.globals import DOT
from testperanto.substitutions import SymbolSubstitution



class DistributionManager:
    """Manages the weightings (usually distributions) of a WRIG.

    Methods
    -------
    add_config(key, dist_config)
        Associates a distribution configuration (dict) with a key
    get(key, substitution=SymbolSubstitution)
        Looks up the distribution associated with a key, initializing it if necessary
    """

    def __init__(self, dist_configs=None, generate_consecutive_ids=False):
        """
        Parameters
        ----------
        dist_configs : dict
            Initial dictionary associating keys with distribution configurations
        generate_consecutive_ids: bool
            Specifies whether the default base distribution (an IdGenerator) generates
            integers in consecutive increasing order, or uniformly at random
        """

        self.dist_configs = dist_configs if dist_configs is not None else dict()
        self.distributions = dict()
        self.generate_consecutive_ids = generate_consecutive_ids

    def add_config(self, key, dist_config):
        """Associates another distribution configuration with a key.

        A distribution configuration is a dictionary with the following keys:
            type: the key registered with the desired Distribution class
                  using the register_distribution function
            base: the key associated with the desired base distribution (optional);
                  if the base is not specified, testperanto assumes that the base is
                  key[-1] if the key is a tuple of length >1 and otherwise initializes
                  an IdGenerator to use as the base distribution
        It should also contain any arguments that need to be provided to the
        constructor for the chosen Distribution class.

        Parameters
        ----------
        key : str
            The key to use for future lookups of the distribution configuration
        dist_config : dict
            The new distribution configuration
        """

        self.dist_configs[key] = dist_config

    def get(self, key, substitution=SymbolSubstitution()):
        """Looks up the distribution associated with a key, initializing it if necessary.

        For instance, the code:
            sub = SymbolSubstitution()
            sub.add_substitution('$y1', '52')
            manager.get(('a','$y1'), sub_11_21)
        would look up the distribution associated with the key ('a', '52').

        If this distribution has not been previously "gotten", then it is constructed
        and cached by the DistributionManager according to the configuration associated
        with key ('a','$y1').

        Parameters
        ----------
        key : str
            The key associated with the desired distribution configuration
        substitution : testperanto.substitutions.Substitution
            Used to replace any variables in the key
        """

        def lookup_dist_config(k):
            backed_off_key = list(k)
            back_off_index = len(k) - 1
            while tuple(backed_off_key) not in self.dist_configs:
                backed_off_key[back_off_index] = "$y0"
            try:
                return self.dist_configs[tuple(backed_off_key)]
            except KeyError:
                raise KeyError('no matches for distribution: {}'.format(k))

        def get_base(argz, k, sub):
            if "base" in argz:
                return self.get(argz["base"], sub)
            elif len(k) > 1 and k[:-1] in self.dist_configs:
                return self.get(k[:-1], sub)
            else:
                return distributions.IdGenerator(self.generate_consecutive_ids)
        
        ground_key = substitution.substitute_into_compound_symbol(key)
        dist_config = lookup_dist_config(ground_key)
        if ground_key not in self.distributions:
            type = dist_config['type']
            args = {key: dist_config[key] for key in dist_config if key not in ['type']}
            args["base"] = get_base(args, key, substitution)
            try:
                dist = lookup_distribution(type)(**args)
            except TypeError:
                del args["base"]
                dist = lookup_distribution(type)(**args)
            self.distributions[ground_key] = dist
        return self.distributions[ground_key]

    @staticmethod
    def from_config(config):
        """Initializes the DistributionManager using a configuration dictionary.

        For instance:
            config = {'distributions': [{"name": "verb",
                                         "type": "pyor",
                                         "strength": 150,
                                         "discount": 0.35}]}
        specifies a DistributionManager that associates the key "verb" with
        a Pitman-Yor distribution with strength 150, discount 0.35, and a default
        base distribution (i.e. an IdGenerator).

        Parameters
        ----------
        config : dict
            The configuration dictionary

        Returns
        -------
        testperanto.distmanager.DistributionManager
            The newly constructed DistributionManager
        """
        factory_configs = []
        if 'distributions' in config:
            factory_configs = config['distributions']
        factories = dict()
        for fconfig in factory_configs:
            name = tuple(fconfig['name'].split(DOT))
            args = {key: fconfig[key] for key in fconfig if key not in ['name']}
            factories[name] = args
        return DistributionManager(factories)

