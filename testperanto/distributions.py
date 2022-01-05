##
# distributions.py
# Miscellaneous probability distributions.
# $Author: mhopkins $
# $Revision: 34246 $
# $Date: 2012-05-23 18:26:36 -0700 (Wed, 23 May 2012) $
##


import random
    
class DistributionInitError(Exception): pass                

class Distribution(object):

    def sample(self):    
        """Generate a sample from the distribution.""" 
        raise NotImplementedError("Implement me.")


class CrossDistribution(Distribution):

    def __init__(self, dist_list):
        self.dist_list = dist_list

    def sample(self):
        return tuple([dist.sample() for dist in self.dist_list])


class AltCategoricalDistribution(Distribution):

    def __init__(self, weights):
        """Initialize a CategoricalDistribution from unnormalized weights."""
        normalizer = float(sum(weights))
        self.normalized_weights = map(lambda x: float(x)/float(normalizer), weights)
      
    def sample(self):
        """Generate a sample from the distribution."""
        trial = random.random()
        for i in range(len(self.weights)): #TODO: make this a binary search?
            if trial < self.weights[i]:
                return i
        return len(self.weights) - 1

    def __str__(self):
        retval = 'categorical'
        for wt in self.normalized_weights:
            retval += ':' + str(wt)
        return retval


class CategoricalDistribution(Distribution):

    def __init__(self, weights):
        """Initialize a CategoricalDistribution from unnormalized weights."""
        normalizer = float(sum(weights))
        self.normalized_weights = list(map(lambda x: float(x)/float(normalizer), weights))
        self.weights = []
        weight_sum = 0.0
        self.most_likely_index = 0
        for i in range(len(self.normalized_weights)):
            self.weights.append( weight_sum + self.normalized_weights[i] )
            weight_sum += self.normalized_weights[i]
            if self.normalized_weights[i] > self.normalized_weights[self.most_likely_index]:
                self.most_likely_index = i
       
    def sample(self):
        """Generate a sample from the distribution."""
        trial = random.random()
        for i in range(len(self.weights)): #TODO: make this a binary search?
            if trial < self.weights[i]:
                return i
        return len(self.weights)-1

    def get_most_likely_sample(self):
        """Return the most likely category."""
        return self.most_likely_index

    def __str__(self):
        retval = 'categorical'
        for wt in self.normalized_weights:
            retval += ':' + str(wt)
        return retval


class DirichletDistribution(Distribution):

    def __init__(self, concentration_params):
        self.concentration_params = concentration_params

    def sample(self):
        sample = [random.gammavariate(a, 1) for a in self.concentration_params]
        sample = [v / sum(sample) for v in sample]
        return CategoricalDistribution(sample)


class UniformDistribution(Distribution):

    def __init__(self, domain):
        self.domain = domain

    def sample(self):
        domain_index = random.randint(0, len(self.domain) - 1)
        return self.domain[domain_index]


class CRPDistribution(Distribution):

    def __init__(self, base_dist, alpha=10.0):
        """Constructor"""
        self.base_dist = base_dist
        self.alpha = float(alpha)
        self.num_samples_taken_so_far = 0
        self.obj_multiplicity = {}
        
    def sample(self):
        chance_of_new = self.alpha / (self.alpha + self.num_samples_taken_so_far)
        trial = random.random()
        if trial < chance_of_new:
            obj = self.base_dist.sample()
        else:
            lottery_number = random.randint(1,self.num_samples_taken_so_far)
            ticket_number = 0
            for cand_obj in self.obj_multiplicity:
                ticket_number += self.obj_multiplicity[cand_obj]
                if ticket_number >= lottery_number:
                    obj = cand_obj
                    break
        self.num_samples_taken_so_far += 1
        if obj not in self.obj_multiplicity:
            self.obj_multiplicity[obj] = 1
        else:
            self.obj_multiplicity[obj] += 1
        return obj


class IdGenerator(object):
    def __init__(self):
        self.next_id = 0
        
    def sample(self):
        retval = self.next_id
        self.next_id += 1
        return retval


class GEMDistribution(object):

    def __init__(self, alpha=10.0):
        """Constructor"""
        self.crp_dist = CRPDistribution(IdGenerator(), alpha)
       
    def sample(self):
        # TODO: reimplement as an actual stick-breaking dist.
        return self.crp_dist.sample()


class PitmanYorProcess(Distribution):

    def __init__(self, base_dist, discount, strength):
        self.base_dist = base_dist
        self.discount = float(discount)
        self.strength = float(strength)
        self.num_samples_so_far = 0
        self.samples = []
        self.sample_multiplicity = []

    def sample(self):
        # Seems right but the graphs are more erratic than the other implementation, for some reason.
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
                # print str(self.strength_param+self.num_samples_so_far) + ":" + str(trial) + ":" + str(next_bound) + ":" + str(sample_index)
                # print self.sample_multiplicity
                next_bound += self.sample_multiplicity[sample_index]
            obj = self.samples[sample_index]
            self.sample_multiplicity[sample_index] += 1.0
        self.num_samples_so_far += 1
        return obj

    def sample_alt(self):
        chance_of_new_numer = self.strength_param + (self.discount_param * len(self.samples))
        chance_of_new_denom = self.strength_param + self.num_samples_so_far
        chance_of_new = chance_of_new_numer / chance_of_new_denom
        trial = random.random()
        if trial < chance_of_new:
            obj = self.base_dist.sample()
            self.samples.append(obj)
            self.sample_multiplicity.append(1.0 - self.discount_param)
        else:
            dist = AltCategoricalDistribution(self.sample_multiplicity)
            sample_index = dist.sample()
            obj = self.samples[sample_index]
            self.sample_multiplicity[sample_index] += 1.0
        self.num_samples_so_far += 1
        return obj

