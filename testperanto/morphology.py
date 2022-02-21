from abc import ABC, abstractmethod

class Morpher(ABC):
    def morph(self, word, properties):
        ...


class SuffixMorpher(Morpher):
    def __init__(self, properties, suffix_map):
        self.properties = properties
        self.suffix_map = suffix_map

    def morph(self, word, property_values):
        value = tuple([property_values[p] for p in self.properties])
        suffix = self.suffix_map[value]
        return word + suffix


class EnglishVerbMorpher(Morpher):
    def __init__(self):
        self.base_morpher = SuffixMorpher(properties=('PERSON', 'COUNT', 'TENSE'),
                                                suffix_map={('1', 'sng', 'present'): '',
                                                            ('1', 'plu', 'present'): '',
                                                            ('1', 'sng', 'perfect'): 'd',
                                                            ('1', 'plu', 'perfect'): 'd',
                                                            ('3', 'sng', 'present'): 's',
                                                            ('3', 'plu', 'present'): '',
                                                            ('3', 'sng', 'perfect'): 'd',
                                                            ('3', 'plu', 'perfect'): 'd'})

    def morph(self, word, property_values):
        return self.base_morpher.morph(word, property_values)


class EnglishNounMorpher(Morpher):
    def __init__(self):
        self.base_morpher = SuffixMorpher(properties=('COUNT',),
                                                suffix_map={('sng',): '', ('plu',): 's'})

    def morph(self, word, property_values):
        return self.base_morpher.morph(word, property_values)
