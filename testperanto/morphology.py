from abc import ABC, abstractmethod

class Morphologizer(ABC):
    def mutate(self, word, properties):
        ...


class SuffixMorphologizer(Morphologizer):
    def __init__(self, properties, suffix_map):
        self.properties = properties
        self.suffix_map = suffix_map

    def mutate(self, word, property_values):
        value = tuple([property_values[p] for p in self.properties])
        suffix = self.suffix_map[value]
        return word + suffix


class EnglishVerbMorpher(Morphologizer):
    def __init__(self):
        self.base_morpher = SuffixMorphologizer(properties=('PERSON', 'COUNT', 'TENSE'),
                                                suffix_map={('1', 'sng', 'present'): '',
                                                            ('1', 'plu', 'present'): '',
                                                            ('1', 'sng', 'perfect'): 'd',
                                                            ('1', 'plu', 'perfect'): 'd',
                                                            ('3', 'sng', 'present'): 's',
                                                            ('3', 'plu', 'present'): '',
                                                            ('3', 'sng', 'perfect'): 'd',
                                                            ('3', 'plu', 'perfect'): 'd'})

    def mutate(self, word, property_values):
        return self.base_morpher.mutate(word, property_values)


class EnglishNounMorpher(Morphologizer):
    def __init__(self):
        self.base_morpher = SuffixMorphologizer(properties=('COUNT',),
                                                suffix_map={('sng',): '', ('plu',): 's'})

    def mutate(self, word, property_values):
        return self.base_morpher.mutate(word, property_values)
