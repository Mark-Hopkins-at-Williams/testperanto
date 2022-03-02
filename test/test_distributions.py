##
# testdistributions.py
# Unit tests for distributions.py.
##


import unittest
from testperanto.distributions import IdGenerator, PitmanYorProcess
from testperanto.distributions import CategoricalDistribution, StickyCategorical


class PredictableRandomGen:
    def __init__(self, samples):
        self.samples = samples
        self.next_sample = 0

    def __call__(self):
        result = self.samples[self.next_sample]
        self.next_sample = (self.next_sample + 1) % len(self.samples)
        return result


class TestDistributions(unittest.TestCase):

    def test_id_generator_consecutive(self):
        generator = IdGenerator(consecutive_ids=True)
        for i in range(100):
            self.assertEqual(generator.sample(), i)

    def test_id_generator_non_consecutive(self):
        generator = IdGenerator(consecutive_ids=False)
        rands = [generator.sample() for _ in range(100)]
        assert rands != list(range(100))

    def test_categorical_distribution1(self):
        random_gen = PredictableRandomGen([0.4, 0.8, 0.1])
        dist = CategoricalDistribution([0.2, 0.3, 0.5], random_gen=random_gen)
        self.assertEqual(dist.sample(), 1)
        self.assertEqual(dist.sample(), 2)
        self.assertEqual(dist.sample(), 0)
        self.assertEqual(dist.sample(), 1)

    def test_categorical_distribution2(self):
        random_gen = PredictableRandomGen([0.4, 0.8, 0.1])
        dist = CategoricalDistribution([0.2, 0.3, 0.5], labels=['a', 'b', 'c'],
                                       random_gen=random_gen)
        self.assertEqual(dist.sample(), 'b')
        self.assertEqual(dist.sample(), 'c')
        self.assertEqual(dist.sample(), 'a')
        self.assertEqual(dist.sample(), 'b')

    def test_categorical_distribution3(self):
        random_gen = PredictableRandomGen([0.4, 0.8, 0.1])
        dist = CategoricalDistribution([20, 30, 50], labels=['a', 'b', 'c'],
                                       random_gen=random_gen)
        self.assertEqual(dist.sample(), 'b')
        self.assertEqual(dist.sample(), 'c')
        self.assertEqual(dist.sample(), 'a')
        self.assertEqual(dist.sample(), 'b')

    def test_sticky_categorical(self):
        random_gen = PredictableRandomGen([0.4, 0.8, 0.1])
        dist = StickyCategorical([20, 30, 50], domain=['a', 'b', 'c'],
                                  random_gen=random_gen)
        self.assertEqual(dist.sample(), 'b')
        self.assertEqual(dist.sample(), 'b')
        self.assertEqual(dist.sample(), 'b')
        self.assertEqual(dist.sample(), 'b')

    def test_pitman_yor1(self):
        random_gen = PredictableRandomGen([0.4, 0.8, 0.1])
        base = IdGenerator(consecutive_ids=True)
        dist = PitmanYorProcess(base, discount=0.4, strength=2, random_gen=random_gen)
        samples = [dist.sample() for _ in range(10)]
        self.assertEqual(samples, [0, 0, 1, 2, 0, 3, 4, 2, 5, 6])

    def test_pitman_yor2(self):
        random_gen = PredictableRandomGen([0.99])
        base = IdGenerator(consecutive_ids=True)
        dist = PitmanYorProcess(base, discount=0.4, strength=2, random_gen=random_gen)
        samples = [dist.sample() for _ in range(10)]
        self.assertEqual(samples, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    def test_pitman_yor3(self):
        random_gen = PredictableRandomGen([0.01])
        base = IdGenerator(consecutive_ids=True)
        dist = PitmanYorProcess(base, discount=0.4, strength=2, random_gen=random_gen)
        samples = [dist.sample() for _ in range(10)]
        self.assertEqual(samples, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])



if __name__ == "__main__":
    unittest.main()   