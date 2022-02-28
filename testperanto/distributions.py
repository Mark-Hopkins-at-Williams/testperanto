##
# distributions.py
# Miscellaneous probability distributions.
##

from abc import ABC, abstractmethod
from testperanto.globals import DIST_CONSTRUCTORS
import random


def register_distribution(id, dist_constructor):
    """Registers a custom distribution with the testperanto package.

    Once a distribution is registered with the name X, its constructor function can
    be subsequently looked up using testperanto.distributions.lookup_distribution(X).

    Parameters
    ----------
    id : str
        The id to associate with the distribution
    dist_constructor : constructor
        Constructor function for the distribution
    """

    DIST_CONSTRUCTORS[id] = dist_constructor


def lookup_distribution(id):
    """Retrieves the constructor for a registered distribution.

    Parameters
    ----------
    id : str
        The id associated with the distribution during its registration
    """

    try:
        return DIST_CONSTRUCTORS[id]
    except KeyError:
        raise KeyError(f"Distribution key not recognized: {id}")


class Distribution(ABC):
    """Abstract class for a probability distribution.

    Methods
    -------
    sample()
        Returns a random sample from the distribution.
    """

    @abstractmethod
    def sample(self):    
        """Returns a random sample from the distribution. Abstract method."""


class IdGenerator(Distribution):
    """Generates a non-negative integer, uniformly at random.

    The behavior of the IdGenerator can be made deterministic by providing the
    argument consecutive_ids=True to the constructor, in which case the IdGenerator
    will "sample" integers in consecutive ascending order, starting with 0.

    Methods
    -------
    sample()
        Returns a non-negative integer.
    """

    def __init__(self, consecutive_ids):
        """
        Parameters
        ----------
        consecutive_ids : bool
            If True, then the IdGenerator will "sample" integers in consecutive
            ascending order, starting with 0.
        """

        self.next_id = 0
        self.max_id = 2 ** 32 - 1
        self.consecutive_ids = consecutive_ids

    def sample(self):
        """Returns a non-negative integer.

        Returns
        -------
        int
            A non-negative integer, sampled uniformly at random. Exception: if
            self.consecutive_ids==True, then this is the lowest unused non-negative
            integer.
        """
        if self.consecutive_ids:
            retval = self.next_id
            self.next_id += 1
        else:
            retval = random.randint(0, self.max_id)
        return retval


class CategoricalDistribution(Distribution):
    """A probability distribution over a finite set of categories.

    Methods
    -------
    sample()
        Returns a category sampled from the distribution.
    """

    def __init__(self, weights, labels=None, random_gen=random.random):
        """
        Parameters
        ----------
        weights : list[float]
            Unnormalized non-negative weights to associate with each category
        labels : list[str]
            A string label to associate with each category (should have same length as
            weights)
        random_gen : function
            A random generator that generates floats between 0.0 and 1.0
        """

        self.random_gen = random_gen
        normalizer = float(sum(weights))
        self.normalized_weights = list(map(lambda x: float(x)/float(normalizer), weights))
        self.weights = []
        weight_sum = 0.0
        if labels is None:
            self.labels = list(range(len(weights)))
        else:
            self.labels = labels
        for i in range(len(self.normalized_weights)):
            self.weights.append( weight_sum + self.normalized_weights[i] )
            weight_sum += self.normalized_weights[i]
        self.enumerated = list(enumerate(self.weights))
       
    def sample(self):
        """Returns a category sampled from the distribution.

        Returns
        -------
        object
            The label of the sampled category, or the index if no labels are provided to
            the constructor
        """

        trial = self.random_gen()
        return self.labels[CategoricalDistribution.binary_search(trial, self.enumerated)]

    @staticmethod
    def binary_search(target, ls):
        """A binary search helper function."""
        if len(ls) == 1:
            return ls[0][0] + int(ls[0][1] < target)
        else:
            midpoint = len(ls) // 2
            if target <= ls[midpoint][1]:
                return CategoricalDistribution.binary_search(target, ls[:midpoint])
            else:
                return CategoricalDistribution.binary_search(target, ls[midpoint:])

    def __str__(self):
        """Overrides the default string representation to give info about the weights."""
        retval = 'categorical'
        for wt in self.normalized_weights:
            retval += ':' + str(wt)
        return retval


class UniformDistribution(Distribution):
    """A uniform distribution over a finite domain.

    Methods
    -------
    sample()
        Returns an element from the domain, sampled uniformly at random.
    """
    def __init__(self, domain):
        """
        Parameters
        ----------
        domain : list
            A list of elements to sample from
        """

        self.domain = domain

    def sample(self):
        """Returns an element from the domain, sampled uniformly at random.

        Returns
        -------
        object
            An element from the domain, sampled uniformly at random
        """
        domain_index = random.randint(0, len(self.domain) - 1)
        return self.domain[domain_index]


class PitmanYorProcess(Distribution):
    """Implementation of a Pitman-Yor process.

    Methods
    -------
    sample()
        Returns a non-negative integer sampled from the PY process.
    """
    def __init__(self, base, discount, strength, random_gen=random.random):
        """
        Pqrameters
        ----------
        base_dist : testperanto.distributions.Distribution
            Base distribution of the Pitman-Yor process
        discount : float
            Discount parameter
        strength : float
            Strength parameter
        random_gen : function
            A random generator that generates floats between 0.0 and 1.0
        """

        self.base_dist = base
        self.discount = float(discount)
        self.strength = float(strength)
        self.random_gen = random_gen
        self.num_samples_so_far = 0
        self.samples = []
        self.sample_multiplicity = []

    def sample(self):
        """Returns a non-negative integer sampled from the PY process.

        Returns
        -------
        int
            A non-negative integer sampled from the PY process
        """

        trial = (self.strength + self.num_samples_so_far) * self.random_gen()
        next_bound = self.strength + (self.discount * len(self.samples))
        if trial <= next_bound:
            obj = self.base_dist.sample()
            self.samples.append(obj)
            self.sample_multiplicity.append(1.0 - self.discount)
        else:
            sample_index = -1
            while trial > next_bound:
                sample_index += 1
                next_bound += self.sample_multiplicity[sample_index]
            obj = self.samples[sample_index]
            self.sample_multiplicity[sample_index] += 1.0
        self.num_samples_so_far += 1
        return obj


register_distribution('categorical', CategoricalDistribution)
register_distribution('uniform', UniformDistribution)
register_distribution('pyor', PitmanYorProcess)

