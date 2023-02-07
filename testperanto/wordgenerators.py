##
# wordgenerators.py
# Data structures and algorithms for word generators.
##


import random
from abc import ABC, abstractmethod
from testperanto.distributions import CategoricalDistribution
from testperanto.globals import WORD_GENERATORS


def register_word_generator(name, generator):
    """Registers a custom word generator with the testperanto package.

    Once a word generator is registered with the name X, its constructor function can
    be subsequently looked up using testperanto.wordgenerators.lookup_word_generator(X).

    Parameters
    ----------
    name : str
        The id to associate with the word generator
    generator : constructor
        Constructor function for the word generator
    """

    WORD_GENERATORS[name] = generator


def lookup_word_generator(name):
    """Retrieves the constructor for a registered word generator.

    Parameters
    ----------
    name : str
        The id associated with the word generator during its registration
    """

    if name in WORD_GENERATORS:
        return WORD_GENERATORS[name]
    else:
        return None


class WordGenerator(ABC):
    """Generates a random string upon request. Abstract class.

    Methods
    -------
    generate()
        Generates a novel word
    """

    @abstractmethod
    def generate(self):
        """Generates a random word.

        Returns
        -------
        str
            The generated word
        """

        raise NotImplementedError("Implement me.")        


class ListBasedWordGenerator(WordGenerator):
    """Generates a new word by choosing it (uniformly at random) from a list."""

    def __init__(self, word_list, choice_fn=random.choice):
        """
        Parameters
        ----------
        word_list : list[str]
            The list of candidate words to choose from
        choice_fn : function
            Function that chooses an element at random from a list
        """

        self.word_list = word_list
        self.choice_fn = choice_fn

    def generate(self):
        """Chooses a random word uniformly at random from the provided word list.

        Returns
        -------
        str
            The generated word
        """

        return self.choice_fn(self.word_list)


class IteratingWordGenerator(WordGenerator):
    """Generates words from a list until exhausted, then defaults to a backup generator."""

    def __init__(self, word_list, backup_generator):
        """
        Parameters
        ----------
        word_list : list[str]
            The list of initial words to iterate through
        backup_generator : WordGenerator
            The backup generator to use after original list is exhausted
        """

        self.word_list = word_list
        self.backup_generator = backup_generator
        self.current_index = 0
                
    def generate(self):
        """Provides the next unused word in the list, until defaulting to a backup generator.

        Returns
        -------
        str
            The generated word
        """
        if self.current_index < len(self.word_list):
            result = self.word_list[self.current_index]
            self.current_index += 1
        else:
            result = self.backup_generator.generate()
        return result


class PrefixSuffixWordGenerator(WordGenerator):
    """Generates a new word by concatenating a prefix and suffix from two subgenerators."""

    def __init__(self, prefix_generator, suffix_generator):
        """
        Parameters
        ----------
        prefix_generator : WordGenerator
            The prefix generator
        suffix_generator : WordGenerator
            The suffix generator
        """

        self.prefix_generator = prefix_generator
        self.suffix_generator = suffix_generator
        
    def generate(self):
        """Concatenates a generated prefix and suffix.

        Returns
        -------
        str
            The generated word
        """
        return self.prefix_generator.generate() + self.suffix_generator.generate()
    
    
class AtomBasedWordGenerator(WordGenerator):
    """Generates a new word by concatenating atomic building blocks.

    The number of atoms concatenated is chosen from a supplied distribution over word
    lengths.
    """

    def __init__(self, atom_generator, word_length_distribution):
        """
        Parameters
        ----------
        atom_generator : WordGenerator
            The word generator that provides the atomic building blocks
        word_length_distribution : testperanto.distributions.Distribution
            A distribution over integers that determines how many atoms are concatenated
            to generate a new word
        """

        self.atom_generator = atom_generator
        self.word_length_distribution = word_length_distribution
        
    def generate(self):
        """Concatenates atomic building blocks to generate a word.

        Specifically:
        1. A word length K is drawn from the provided word length distribution.
        2. K atoms are drawn uniformly at random from the atomic WordGenerator
           and concatenated.

        """

        word_length = self.word_length_distribution.sample()
        result = ""
        for _ in range(word_length):
            result += self.atom_generator.generate()
        return result


##
# simple English word generators
##

register_word_generator("EnglishSyllables",
                        ListBasedWordGenerator(['ba','be','bi','bo','bu','co','cu',
                                                'cha','chi','cho','chu', 'da','di',
                                                'do','du','fa','fi','fo','fu','fla',
                                                'fli','flo', 'flu','fra','fri','fro',
                                                'fru','ga','go','gu','gla','glo',
                                                'glu','ha','he','hi','ho','hu','ja',
                                                'ji','jo','ju','ka','kee','ki','ko',
                                                'ku','la','lee','li','lo','lu','ma','mee',
                                                'mi','moo','mo']))
register_word_generator("EnglishConsonants",
                        ListBasedWordGenerator(['b','c','d','f','g','k','l','m','n',
                                                'p','r','t','x']))
register_word_generator("EnglishPrepositions",
                        ListBasedWordGenerator(["in", "on", "to", "at", "by", "of", "off",
                                                "with", "around", "from", "into", "before",
                                                "along", "above", "across", "against",
                                                "among", "behind", "below", "beneath",
                                                "beside", "between", "near", "toward",
                                                "under", "upon", "within"]))

def goose_generator():
    atom_generator = lookup_word_generator('EnglishSyllables')
    word_length_distribution = CategoricalDistribution([0, 0.2, 0.6, 0.2, 0.0])
    prefix_generator = AtomBasedWordGenerator(atom_generator, word_length_distribution)
    suffix_generator = lookup_word_generator('EnglishConsonants')
    return PrefixSuffixWordGenerator(prefix_generator, suffix_generator)


##
# "Mother Goose" word generators
##

register_word_generator("Goose", goose_generator())
register_word_generator("GooseAdjectives",
                        PrefixSuffixWordGenerator(goose_generator(),
                                                  ListBasedWordGenerator(['ish'])))
register_word_generator("GooseVerbs",
                        PrefixSuffixWordGenerator(goose_generator(),
                                                  ListBasedWordGenerator(['ize'])))
register_word_generator("GooseAdverbs",
                        PrefixSuffixWordGenerator(goose_generator(),
                                                  ListBasedWordGenerator(['ly'])))


##
# simple Japanese word generators
##

register_word_generator("JapaneseSyllables",
                        ListBasedWordGenerator(['ka','ki','ku','ke','ko','sa','si','su',
                                                'se','so','ta','ti','tu','te','to','na',
                                                'ni','nu','ne','no','ha','hi','fu','he',
                                                'ho','ma','mi','mu','me','mo','ra','ri',
                                                'ru','re','ro','ga','gi','gu','ge','go',
                                                'za','ji','zu','ze','zo','da','zu','de',
                                                'do','pa','pi','pu','pe','po','ba','bi',
                                                'bu','be','bo','ya','yu','yo','wa','wo',
                                                'n','a','e','i','o','u']))
register_word_generator("JapaneseStems",
                        AtomBasedWordGenerator(lookup_word_generator('JapaneseSyllables'),
                                               CategoricalDistribution([0, 0, 0.2, 0.5, 0.2, 0.1])))
register_word_generator("JapanesePrepositions",
                        ListBasedWordGenerator(['naka','ue','chikako','shita','mae','yoko','tonari','ushiro']))



german_syllables = ['flach', 'stau', 'bei', 'der', 'dich', 'dung', 'mein',
                    'fin', 'frisch', 'frau', 'geh', 'glied', 'gun', 'gnug' 'haf', 'han', 'heim'
                    'her', 'herr', 'hub', 'lag', 'hung', 'jahr', 'keit', 'kol', 'kom', 'kenn',
                    'kon', 'lang', 'lich', 'ler', 'lung', 'man', 'mensch', 'milch', 'mon', 'nach',
                    'nied', 'par', 'rech', 'rich', 'run', 'rung', 'schlag', 'sam',
                    'schmid', 'sich', 'ster', 'sung', 'tag', 'tel', 'ter', 'tik', 'trum', 'tun',
                    'tung', 'run', 'ver', 'vor', 'wir', 'wohn', 'zer', 'ziem', 'zum']
syllable_generator = ListBasedWordGenerator(german_syllables)
stem_generator = AtomBasedWordGenerator(syllable_generator,
                                        CategoricalDistribution([0, 0, 0.4, 0.4, 0.1]))
register_word_generator("german-stems", stem_generator)