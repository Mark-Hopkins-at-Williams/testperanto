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
        self.base_morpher = SuffixMorphologizer(properties=('COUNT',),
                                                suffix_map={('sng',): 's', ('plu',): ''})

    def mutate(self, word, property_values):
        return self.base_morpher.mutate(word, property_values)


class EnglishNounMorpher(Morphologizer):
    def __init__(self):
        self.base_morpher = SuffixMorphologizer(properties=('COUNT',),
                                                suffix_map={('sng',): '', ('plu',): 's'})

    def mutate(self, word, property_values):
        return self.base_morpher.mutate(word, property_values)
