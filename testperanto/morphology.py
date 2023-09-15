##
# morphology.py
# Tools for generating word morphology.
##

from abc import ABC, abstractmethod

class Morpher(ABC):
    """Modifies a word stem to express syntactic properties. Abstract class.

    Methods
    -------
    morph(stem, properties)
        Returns a declension of the word stem based on provided syntactic properties.
    """

    def morph(self, word, properties):
        """Returns a declension of the word stem based on provided syntactic properties.

        Parameters
        ----------
        word : str
            The original word stem
        properties: dict
            Maps (string) syntactic properties to their values

        Returns
        -------
        str
            A declension of the word stem based on the provided syntactic properties
        """




class SuffixMorpher(Morpher):
    """Adds a suffix to a word stem to express syntactic properties.

    To initialize, we provide a list of property names, and a "suffix map" that
    associates a suffix with tuples of corresponding property values.

    For instance, German adjectives have suffixes based on the gender and case
    of the noun they modify. To build a SuffixMorpher for this, we can do the
    following:
        morpher = SuffixMorpher(property_names=('GENDER', 'CASE'),
                                suffix_map={('m', 'acc'): 'en',
                                            ('f', 'acc'): 'e',
                                            ('n', 'acc'): 'es',
                                            ('m', 'dat'): 'em',
                                            ('f', 'dat'): 'er',
                                            ('n', 'dat'): 'em'})

    Then:
        result = morpher.morph('rot', {'GENDER': 'n', 'CASE': 'acc'})
    returns 'rotes', and:
        result = morpher.morph('rot', {'GENDER': 'm', 'CASE': 'dat'})
    returns 'rotem'.

    Methods
    -------
    morph(stem, properties)
        Appends a suffix to the word stem based on provided syntactic properties.
    """

    def __init__(self, property_names, suffix_map):
        """
        Parameters
        ----------
        property_names : list[str]
            Names of the syntactic properties used to determine the suffix
        suffix_map: dict
            Maps a tuple of property values (corresponding to property_names) to a
            corresponding suffix
        """

        self.property_names = property_names
        self.suffix_map = suffix_map

    def morph(self, word, properties):
        """Appends a suffix to the word stem based on provided syntactic properties.

        Parameters
        ----------
        word : str
            The original word stem
        properties: dict
            Maps (string) syntactic properties to their values

        Returns
        -------
        str
            A declension of the word stem based on the provided syntactic properties
        """
        value = tuple([properties[p] for p in self.property_names])
        suffix = self.suffix_map[value]
        return word + suffix

class PrefixMorpher(Morpher):

    def __init__(self, property_names, prefix_map):
        """
        Parameters
        ----------
        property_names : list[str]
            Names of the syntactic properties used to determine the suffix
        prefix_map: dict
            Maps a tuple of property values (corresponding to property_names) to a
            corresponding suffix
        """

        self.property_names = property_names
        self.prefix_map = prefix_map

    def morph(self, word, properties):
        """Prepends a prefix to the word stem based on provided syntactic properties.

        Parameters
        ----------
        word : str
            The original word stem
        properties: dict
            Maps (string) syntactic properties to their values

        Returns
        -------
        str
            A declension of the word stem based on the provided syntactic properties
        """
        value = tuple([properties[p] for p in self.property_names])
        prefix = self.prefix_map[value]
        return prefix + word



##
# Some example Morphers
##

class EnglishVerbMorpher(Morpher):
    def morph(self, word, properties):
        if properties['TENSE'] == 'present' and properties['POLARITY'] == 'pos':
            return self.present_tense(word, properties)
        elif properties['TENSE'] == 'perfect' and properties['POLARITY'] == 'pos':
            return self.perfect_tense(word, properties)
        elif properties['TENSE'] == 'present' and properties['POLARITY'] == 'neg':
            return self.negative_present_tense(word, properties)
        elif properties['TENSE'] == 'perfect' and properties['POLARITY'] == 'neg':
            return self.negative_perfect_tense(word, properties)
        else:
            raise Exception(f"Tense not recognized: {properties['TENSE']}")

    def get_auxiliary(self, tense, person, count):        
        if tense == "present":
            return "is" if person == "3" and count == "sng" else "are"
        elif tense == "perfect":
            return "was" if person == "3" and count == "sng" else "were"
        else:
            raise Exception(f"Auxiliary not found: {tense}, {person}, {count}")


    def present_tense(self, word, properties):
        person = properties['PERSON']
        count = properties['COUNT']
        voice = properties['VOICE']
        if voice == "active":
            if person == '3' and count == 'sng':
                return word + "s"
            elif count == 'inf':
                return "to " + word
            else:
                return word
        elif voice == "passive":
            aux = self.get_auxiliary("present", person, count)
            return aux + " " + word + "d"
        else:
            raise Exception(f"Voice not recognized: {voice}")
        
    def perfect_tense(self, word, properties):
        person = properties['PERSON']
        count = properties['COUNT']
        voice = properties['VOICE']
        if count == 'inf':
            return 'to have ' + word + 'd'
        elif voice == "passive":
            aux = self.get_auxiliary("perfect", person, count)
            return aux + " " + word + "d"
        else:
            return word + "d"
    
    def negative_present_tense(self, word, properties):
        person = properties['PERSON']
        count = properties['COUNT']
        if person == '3' and count == 'sng':
            return 'does not ' + word
        else:
            return 'do not ' + word

    def negative_perfect_tense(self, word, _):
        return 'did not ' + word

class MapBasedEnglishVerbMorpher(Morpher):
    def __init__(self):
        self.base_morpher = SuffixMorpher(property_names=('PERSON', 'COUNT', 'TENSE', 'POLARITY'),
                                          suffix_map={('1', 'sng', 'present', 'pos'): '',
                                                      ('1', 'plu', 'present', 'pos'): '',
                                                      ('1', 'sng', 'perfect', 'pos'): 'd',
                                                      ('1', 'plu', 'perfect', 'pos'): 'd',
                                                      ('3', 'sng', 'present', 'pos'): 's',
                                                      ('3', 'plu', 'present', 'pos'): '',
                                                      ('3', 'sng', 'perfect', 'pos'): 'd',
                                                      ('3', 'plu', 'perfect', 'pos'): 'd',
                                                      ('1', 'sng', 'present', 'neg'): '',
                                                      ('1', 'plu', 'present', 'neg'): '',
                                                      ('1', 'sng', 'perfect', 'neg'): '',
                                                      ('1', 'plu', 'perfect', 'neg'): '',
                                                      ('3', 'sng', 'present', 'neg'): '',
                                                      ('3', 'plu', 'present', 'neg'): '',
                                                      ('3', 'sng', 'perfect', 'neg'): '',
                                                      ('3', 'plu', 'perfect', 'neg'): '',
                                                      ('1', 'sng', 'future', 'pos'): '',
                                                      ('1', 'plu', 'future', 'pos'): '',
                                                      ('3', 'sng', 'future', 'pos'): '',
                                                      ('3', 'plu', 'future', 'pos'): '',
                                                      ('1', 'sng', 'future', 'neg'): '',
                                                      ('1', 'plu', 'future', 'neg'): '',
                                                      ('1', 'sng', 'future', 'neg'): '',
                                                      ('3', 'sng', 'future', 'neg'): '',
                                                      ('3', 'plu', 'future', 'neg'): '',
                                                      ('1', 'sng', 'past', 'pos'): '',
                                                      ('1', 'plu', 'past', 'pos'): '',
                                                      ('3', 'sng', 'past', 'pos'): '',
                                                      ('3', 'plu', 'past', 'pos'): '',
                                                      ('1', 'sng', 'past', 'neg'): '',
                                                      ('1', 'plu', 'past', 'neg'): '',
                                                      ('1', 'sng', 'past', 'neg'): '',
                                                      ('3', 'sng', 'past', 'neg'): '',
                                                      ('3', 'plu', 'past', 'neg'): '',
                                                      ('1', 'sng', 'passive', 'pos'): 'd',
                                                      ('1', 'plu', 'passive', 'pos'): 'd',
                                                      ('3', 'sng', 'passive', 'pos'): 'd',
                                                      ('3', 'plu', 'passive', 'pos'): 'd',
                                                      ('1', 'sng', 'passive', 'neg'): 'd',
                                                      ('1', 'plu', 'passive', 'neg'): 'd',
                                                      ('1', 'sng', 'passive', 'neg'): 'd',
                                                      ('3', 'sng', 'passive', 'neg'): 'd',
                                                      ('3', 'plu', 'passive', 'neg'): 'd'})
        self.negation_morpher = PrefixMorpher(property_names=('PERSON', 'COUNT', 'TENSE', 'POLARITY'),
                                              prefix_map={  ('1', 'sng', 'present', 'pos'): '',
                                                            ('1', 'plu', 'present', 'pos'): '',
                                                            ('1', 'sng', 'perfect', 'pos'): '',
                                                            ('1', 'plu', 'perfect', 'pos'): '',
                                                            ('3', 'sng', 'present', 'pos'): '',
                                                            ('3', 'plu', 'present', 'pos'): '',
                                                            ('3', 'sng', 'perfect', 'pos'): '',
                                                            ('3', 'plu', 'perfect', 'pos'): '',
                                                            ('1', 'sng', 'present', 'neg'): 'do not ',
                                                            ('1', 'plu', 'present', 'neg'): 'do not ',
                                                            ('1', 'sng', 'perfect', 'neg'): 'did not ',
                                                            ('1', 'plu', 'perfect', 'neg'): 'did not ',
                                                            ('3', 'sng', 'present', 'neg'): 'does not ',
                                                            ('3', 'plu', 'present', 'neg'): 'do not ',
                                                            ('3', 'sng', 'perfect', 'neg'): 'did not ',
                                                            ('3', 'plu', 'perfect', 'neg'): 'did not ',
                                                            ('1', 'sng', 'future', 'pos'): '',
                                                            ('1', 'plu', 'future', 'pos'): '',
                                                            ('3', 'sng', 'future', 'pos'): '',
                                                            ('3', 'plu', 'future', 'pos'): '',
                                                            ('1', 'sng', 'future', 'neg'): '',
                                                            ('1', 'plu', 'future', 'neg'): '',
                                                            ('3', 'sng', 'future', 'neg'): '',
                                                            ('3', 'plu', 'future', 'neg'): '',
                                                            ('1', 'sng', 'past', 'pos'): 'used to ',
                                                            ('1', 'plu', 'past', 'pos'): 'used to ',
                                                            ('3', 'sng', 'past', 'pos'): 'used to ',
                                                            ('3', 'plu', 'past', 'pos'): 'used to ',
                                                            ('1', 'sng', 'past', 'neg'): 'did not used to ',
                                                            ('1', 'plu', 'past', 'neg'): 'did not used to ',
                                                            ('3', 'sng', 'past', 'neg'): 'did not used to ',
                                                            ('3', 'plu', 'past', 'neg'): 'did not used to ',
                                                            ('1', 'sng', 'passive', 'pos'): '',
                                                            ('1', 'plu', 'passive', 'pos'): '',
                                                            ('3', 'sng', 'passive', 'pos'): '',
                                                            ('3', 'plu', 'passive', 'pos'): '',
                                                            ('1', 'sng', 'passive', 'neg'): '',
                                                            ('1', 'plu', 'passive', 'neg'): '',
                                                            ('1', 'sng', 'passive', 'neg'): '',
                                                            ('3', 'sng', 'passive', 'neg'): '',
                                                            ('3', 'plu', 'passive', 'neg'): ''
                                                            })
        self.conditional_morpher = PrefixMorpher(property_names=('PERSON', 'COUNT', 'TENSE', 'POLARITY'),
                                                 prefix_map={   ('1', 'sng', 'present', 'pos'): 'would ',
                                                                ('1', 'plu', 'present', 'pos'): 'would ',
                                                                ('1', 'sng', 'perfect', 'pos'): 'would have ',
                                                                ('1', 'plu', 'perfect', 'pos'): 'would have ',
                                                                ('3', 'sng', 'present', 'pos'): 'would ',
                                                                ('3', 'plu', 'present', 'pos'): 'would ',
                                                                ('3', 'sng', 'perfect', 'pos'): 'would have ',
                                                                ('3', 'plu', 'perfect', 'pos'): 'would have ',
                                                                ('1', 'sng', 'present', 'neg'): 'would not',
                                                                ('1', 'plu', 'present', 'neg'): 'would not ',
                                                                ('1', 'sng', 'perfect', 'neg'): 'would not have ',
                                                                ('1', 'plu', 'perfect', 'neg'): 'would not have ',
                                                                ('3', 'sng', 'present', 'neg'): 'would not ',
                                                                ('3', 'plu', 'present', 'neg'): 'would not ',
                                                                ('3', 'sng', 'perfect', 'neg'): 'would not have ',
                                                                ('3', 'plu', 'perfect', 'neg'): 'would not have ',
                                                                ('1', 'sng', 'future', 'pos'): 'will ',
                                                                ('1', 'plu', 'future', 'pos'): 'will ',
                                                                ('3', 'sng', 'future', 'pos'): 'will ',
                                                                ('3', 'plu', 'future', 'pos'): 'will ',
                                                                ('1', 'sng', 'future', 'neg'): 'will not ',
                                                                ('1', 'plu', 'future', 'neg'): 'will not ',
                                                                ('3', 'sng', 'future', 'neg'): 'will not ',
                                                                ('3', 'plu', 'future', 'neg'): 'will not ',
                                                                ('1', 'sng', 'past', 'pos'): 'had ',
                                                                ('1', 'plu', 'past', 'pos'): 'had ',
                                                                ('3', 'sng', 'past', 'pos'): 'had ',
                                                                ('3', 'plu', 'past', 'pos'): 'had ',
                                                                ('1', 'sng', 'past', 'neg'): 'had not ',
                                                                ('1', 'plu', 'past', 'neg'): 'had not ',
                                                                ('3', 'sng', 'past', 'neg'): 'had not ',
                                                                ('3', 'plu', 'past', 'neg'): 'had not ',
                                                                ('1', 'sng', 'passive', 'pos'): 'was ',
                                                                ('1', 'plu', 'passive', 'pos'): 'were ',
                                                                ('3', 'sng', 'passive', 'pos'): 'was ',
                                                                ('3', 'plu', 'passive', 'pos'): 'were ',
                                                                ('1', 'sng', 'passive', 'neg'): 'was not ',
                                                                ('1', 'plu', 'passive', 'neg'): 'were not ',
                                                                ('1', 'sng', 'passive', 'neg'): 'was not ',
                                                                ('3', 'sng', 'passive', 'neg'): 'was not ',
                                                                ('3', 'plu', 'passive', 'neg'): 'were not '})

    def morph(self, word, properties):
        result = self.base_morpher.morph(word, properties)
        result = self.negation_morpher.morph(result, properties)
        if properties.get("CONDITIONAL", "no") == "yes" or properties.get("TENSE", "no") == "passive":
            revised_properties = {k: properties[k] for k in properties if k != "CONDITIONAL"}
            result = self.conditional_morpher.morph(result, revised_properties)
        return result


class EnglishNounMorpher(Morpher):
    def __init__(self):
        self.base_morpher = SuffixMorpher(property_names=('COUNT',),
                                          suffix_map={('sng',): '', ('plu',): 's'})

    def morph(self, word, properties):
        return self.base_morpher.morph(word, properties)


class JapaneseVerbMorpher(Morpher):
    def __init__(self):
        self.base_morpher = SuffixMorpher(property_names=('SUBJECT', 'TENSE'),
                                          suffix_map={('1.sng', 'present_simple'): 'masu',
                                                      ('1.plu', 'present_simple'): 'masu',
                                                      ('1.sng', 'past_simple'): 'mashita',
                                                      ('1.plu', 'past_simple'): 'mashita',
                                                      ('2.sng', 'present_simple'): 'masu',
                                                      ('2.plu', 'present_simple'): 'masu',
                                                      ('2.sng', 'past_simple'): 'mashita',
                                                      ('2.plu', 'past_simple'): 'mashita',
                                                      ('3.sng', 'present_simple'): 'masu',
                                                      ('3.plu', 'present_simple'): 'masu',
                                                      ('3.sng', 'past_simple'): 'mashita',
                                                      ('3.plu', 'past_simple'): 'mashita'})

    def morph(self, word, properties):
        return self.base_morpher.morph(word, properties)
