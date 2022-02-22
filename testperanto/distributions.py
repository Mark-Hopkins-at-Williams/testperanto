##
# distributions.py
# Miscellaneous probability distributions.
# $Author: mhopkins $
# $Revision: 34246 $
# $Date: 2012-05-23 18:26:36 -0700 (Wed, 23 May 2012) $
##

from abc import ABC, abstractmethod
import random


class IdGenerator:
    def __init__(self, consecutive_ids):
        self.next_id = 0
        self.max_id = 2 ** 32 - 1
        self.used_ids = set()
        self.consecutive_ids = consecutive_ids

    def sample(self):
        if self.consecutive_ids:
            retval = self.next_id
            self.next_id += 1
        else:
            retval = random.randint(0, self.max_id)
            self.used_ids.add(retval)
        return retval


class Distribution(ABC):

    @abstractmethod
    def sample(self):    
        """Generates a sample from the distribution."""


def binary_search(target, ls):
    if len(ls) == 1:
        return ls[0][0] + int(ls[0][1] < target)
    else:
        midpoint = len(ls) // 2
        if target <= ls[midpoint][1]:
            return binary_search(target, ls[:midpoint])
        else:
            return binary_search(target, ls[midpoint:])


class CategoricalDistribution(Distribution):

    def __init__(self, weights, labels=None):
        """Initialize a CategoricalDistribution from unnormalized weights."""
        normalizer = float(sum(weights))
        self.normalized_weights = list(map(lambda x: float(x)/float(normalizer), weights))
        self.weights = []
        weight_sum = 0.0
        self.most_likely_index = 0
        if labels is None:
            self.labels = list(range(len(weights)))
        else:
            self.labels = labels
        for i in range(len(self.normalized_weights)):
            self.weights.append( weight_sum + self.normalized_weights[i] )
            weight_sum += self.normalized_weights[i]
            if self.normalized_weights[i] > self.normalized_weights[self.most_likely_index]:
                self.most_likely_index = i
        self.enumerated = list(enumerate(self.weights))
       
    def sample(self):
        """Generate a sample from the distribution."""
        trial = random.random()
        return self.labels[binary_search(trial, self.enumerated)]

    def get_most_likely_sample(self):
        """Return the most likely category."""
        return self.labels[self.most_likely_index]

    def __str__(self):
        retval = 'categorical'
        for wt in self.normalized_weights:
            retval += ':' + str(wt)
        return retval


class UniformDistribution(Distribution):

    def __init__(self, domain):
        self.domain = domain

    def sample(self):
        domain_index = random.randint(0, len(self.domain) - 1)
        return self.domain[domain_index]


class PitmanYorProcess(Distribution):

    def __init__(self, base_dist, discount, strength):
        self.base_dist = base_dist
        self.discount = float(discount)
        self.strength = float(strength)
        self.num_samples_so_far = 0
        self.samples = []
        self.sample_multiplicity = []

    def sample(self):
        trial = (self.strength + self.num_samples_so_far) * random.random()
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
