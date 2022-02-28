##
# examples.py
# Some example distributions (for use in the unit tests).
##


from testperanto.distributions import Distribution, register_distribution

class AlternatingDistribution(Distribution):
    """A test distribution that deterministically alternates between sampling 0 and 100."""

    def __init__(self):
        self.next_sample = 0

    def sample(self):
        """Returns 0 or 100, alternating every other call.

        Returns
        -------
        int
            0 on every odd-numbered call (including the first); 100 on every even call.
        """

        retval = self.next_sample
        self.next_sample += 100
        if self.next_sample > 100:
            self.next_sample = 0
        return retval


class AveragerDistribution(Distribution):
    """A test distribution that averages samples from a base distribution.

    Specifically, it samples k times from a base distribution and returns the average.
    The value of k alternates between 1 and 5.
    """

    def __init__(self, base):
        """
        Parameters
        ----------
        base : testperanto.distributions.Distribution
            The base distribution, from which samples are averaged and returned
        """

        self.base_dist = base
        self.num_base_samples = 1

    def sample(self):
        """Returns the average of k samples from the base distribution.

        k=1 on every odd-numbered call (including the first); k=5 on every even call.

        Returns
        -------
        int
            The (rounded down) average of k samples from the base distribution.
            k=1 on every odd-numbered call (including the first); k=5 on every even call.
        """

        sum = 0.0
        for _ in range(self.num_base_samples):
            sum += self.base_dist.sample()
        retval = sum / (1.0 * self.num_base_samples)
        self.num_base_samples += 4
        if self.num_base_samples > 5:
            self.num_base_samples = 1
        return int(retval)


register_distribution('alternating', AlternatingDistribution)
register_distribution('averager', AveragerDistribution)
