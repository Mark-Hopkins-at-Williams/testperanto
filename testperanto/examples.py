from testperanto.distmanager import DistributionFactory

class AlternatingDistribution(object):
    """A test distribution that deterministically alternates between sampling 0 and 100."""

    def __init__(self):
        self.next_sample = 0

    def sample(self):
        retval = self.next_sample
        self.next_sample += 100
        if self.next_sample > 100:
            self.next_sample = 0
        return retval


class AlternatingDistributionFactory(DistributionFactory):
    """A factory for generating AlternatingDistributions."""

    def __init__(self, manager):
        DistributionFactory.__init__(self)

    def instantiate_dist(self, base):
        return AlternatingDistribution()


class AveragerDistribution(object):
    """
    A test distribution that samples k times from a base distribution and returns the average.
    The value of k alternates between 1 and 5.

    """

    def __init__(self, base_dist):
        self.base_dist = base_dist
        self.num_base_samples = 1

    def sample(self):
        sum = 0.0
        for _ in range(self.num_base_samples):
            sum += self.base_dist.sample()
        retval = sum / (1.0 * self.num_base_samples)
        self.num_base_samples += 4
        if self.num_base_samples > 5:
            self.num_base_samples = 1
        return int(retval)


class AveragerDistributionFactory(DistributionFactory):
    """A factory for generating AveragerDistributions."""

    def __init__(self, manager, base_key):
        DistributionFactory.__init__(self)
        self.base_key = base_key

    def instantiate_dist(self, base):
        return AveragerDistribution(base)
