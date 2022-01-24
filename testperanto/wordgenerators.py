##
# wordgenerators.py
# $Author: mhopkins $
# $Revision: 33130 $
# $Date: 2012-04-27 15:56:28 -0700 (Fri, 27 Apr 2012) $
##



import random
from abc import ABC, abstractmethod
from testperanto import distributions


class WordGenerator(ABC):
    """A WordGenerator generates a random string upon request.
    
    Subclasses of WordGenerator should implement .generate() to
    support whatever kind of words they desire.
    """

    @abstractmethod
    def generate(self):
        """Generates a random string according to the parameters of the generator."""
        raise NotImplementedError("Implement me.")        


class ListBasedWordGenerator(WordGenerator):
    """A ListBasedWordGenerator generates a new word by choosing it (uniformly
    at random) from a list.
    """

    def __init__(self, word_list):
        self.word_list = word_list
        
    def generate(self):
        return random.choice(self.word_list)


class IteratingWordGenerator(WordGenerator):
    """An IteratingWordGenerator generates a new word by first iterating through
    a provided word list (in order). Once the word list is exhausted, it falls back
    on a provided default WordGenerator.
    """

    def __init__(self, word_list, default_generator):
        self.word_list = word_list
        self.default_generator = default_generator
        self.current_index = 0
                
    def generate(self):
        if self.current_index < len(self.word_list):
            result = self.word_list[self.current_index]
            self.current_index += 1
        else:
            result = self.default_generator.generate()
        return result


class PrefixSuffixWordGenerator(WordGenerator):
    """A PrefixSuffixWordGenerator generates a new word by choosing the prefix
    (uniformly at random) from a WordGenerator of prefixes and a suffix from
    a WordGenerator of suffixes.
    """

    def __init__(self, prefix_generator, suffix_generator):
        self.prefix_generator = prefix_generator
        self.suffix_generator = suffix_generator
        
    def generate(self):
        return self.prefix_generator.generate() + self.suffix_generator.generate()
    
    
class AtomBasedWordGenerator(WordGenerator):
    """
    An AtomBasedWordGenerator generates a new word from a WordGenerator of atomic building
    blocks and a distribution over word lengths.
    
    To generate a word:
    1. A word length K is drawn from the provided word length distribution.
    2. K atoms are drawn uniformly at random from the atomic WordGenerator and concatenated.
    """

    def __init__(self, atom_generator, word_length_distribution):
        self.atom_generator = atom_generator
        self.word_length_distribution = word_length_distribution
        
    def generate(self):
        word_length = self.word_length_distribution.sample()
        result = ""
        for i in range(word_length):
            result += self.atom_generator.generate()
        return result

class WordGeneratorInitError(Exception): pass                

class WordGeneratorFactory(object):
    """A WordGeneratorFactory generates an instance of a specified WordGenerator."""

    def __init__(self):
        pass
    
    def create_generator(self, generator_name):
        if generator_name == 'EnglishSyllables':
            wordlist = ['ba','be','bi','bo','bu','co','cu','cha','chi','cho','chu',
                        'da','di','do','du','fa','fi','fo','fu','fla','fli','flo',
                        'flu','fra','fri','fro','fru','ga','go','gu','gla','glo',
                        'glu','ha','he','hi','ho','hu','ja','ji','jo','ju','ka',
                        'kee','ki','ko','ku','la','lee','li','lo','lu','ma','mee',
                        'mi','moo','mo']
            return ListBasedWordGenerator( wordlist )
        elif generator_name == 'EnglishConsonants':
            wordlist = ['b','c','d','f','g','k','l','m','n','p','r','t','x']        
            return ListBasedWordGenerator( wordlist )
        elif generator_name == 'EnglishPrepositions':
            wordlist = ["in", "on", "to", "at", "by", "of", "off", "with", "around", "from",
                        "into", "before", "along", "above", "across", "against", "among",
                        "behind", "below", "beneath", "beside", "between", "near", "toward",
                        "under", "upon", "within"]
            prefix_generator = self.create_generator('EnglishSyllables')
            suffix_generator = self.create_generator('EnglishConsonants')
            default_generator = PrefixSuffixWordGenerator( prefix_generator, suffix_generator )
            return IteratingWordGenerator( wordlist, default_generator )
        elif generator_name == 'Seuss':
            atom_generator = self.create_generator('EnglishSyllables')
            # word_length_distribution = distributions.CategoricalDistribution([0,0,0.2,0.5,0.2,0.1])
            # word_length_distribution = distributions.CategoricalDistribution([0,0.3,0.2,0.1,0.1])
            word_length_distribution = distributions.CategoricalDistribution([0,0.0,0.2,0.4,0.4])
            prefix_generator = AtomBasedWordGenerator( atom_generator, word_length_distribution )
            suffix_generator = self.create_generator('EnglishConsonants')
            return PrefixSuffixWordGenerator( prefix_generator, suffix_generator )
        elif generator_name == 'SeussAdjectives':
            prefix_generator = self.create_generator('Seuss')
            suffix_generator = ListBasedWordGenerator( ['ish'] )
            return PrefixSuffixWordGenerator( prefix_generator, suffix_generator )
        elif generator_name == 'SeussVerbs':
            prefix_generator = self.create_generator('Seuss')
            suffix_generator = ListBasedWordGenerator( ['ize'] )
            return PrefixSuffixWordGenerator( prefix_generator, suffix_generator )
        elif generator_name == 'SeussAdverbs':
            prefix_generator = self.create_generator('Seuss')
            suffix_generator = ListBasedWordGenerator( ['ly'] )
            return PrefixSuffixWordGenerator( prefix_generator, suffix_generator )
        elif generator_name == 'JapaneseRomanizedSyllables':
            wordlist = ['ka','ki','ku','ke','ko','sa','si','su','se','so','ta',
                        'ti','tu','te','to','na','ni','nu','ne','no','ha','hi',
                        'fu','he','ho','ma','mi','mu','me','mo','ra','ri','ru',
                        're','ro','ga','gi','gu','ge','go','za','ji','zu','ze',
                        'zo','da','zu','de','do','pa','pi','pu','pe','po','ba',
                        'bi','bu','be','bo','ya','yu','yo','wa','wo','n','a','e','i','o','u']
            return ListBasedWordGenerator( wordlist )
        elif generator_name == 'JapaneseRomanized':
            atom_generator = self.create_generator('JapaneseRomanizedSyllables')
            word_length_distribution = distributions.CategoricalDistribution([0,0,0.2,0.5,0.2,0.1])
            return AtomBasedWordGenerator( atom_generator, word_length_distribution )
        elif generator_name == 'JapaneseRomanizedVerbs':
            prefix_generator = self.create_generator('JapaneseRomanized')
            suffix_generator = ListBasedWordGenerator( ['i'] )
            return PrefixSuffixWordGenerator( prefix_generator, suffix_generator )
        elif generator_name == 'JapaneseRomanizedPrepositions':
            wordlist = ['naka','ue','chikako','shita','mae','yoko','tonari','ushiro']
            prefix_generator = self.create_generator('JapaneseRomanizedSyllables')
            suffix_generator = self.create_generator('JapaneseRomanizedSyllables')
            default_generator = PrefixSuffixWordGenerator( prefix_generator, suffix_generator )
            return IteratingWordGenerator( wordlist, default_generator )
        elif generator_name == 'ChinesePinyinInitials':
            wordlist = ['b','p','m','f','d','t','n','l','g','k','h','j','q','x','zh','ch','sh','r','z','c','s']
            return ListBasedWordGenerator( wordlist )
        elif generator_name == 'ChinesePinyinFinals':
            wordlist = ['a','o','e','ai','ei','ao','ou','an','ang','en','eng',
                        'er','u','ua','uo','uai','ui','uan','uang','un','ueng',
                        'ong','i','ia','ie','iao','iu','ian','iang','in','ing',
                        'ue','iong']
            return ListBasedWordGenerator( wordlist )
        elif generator_name == 'ChinesePinyinSyllables':
            prefix_generator = self.create_generator('ChinesePinyinInitials')
            suffix_generator = self.create_generator('ChinesePinyinFinals')
            return PrefixSuffixWordGenerator( prefix_generator, suffix_generator )
        elif generator_name == 'ChinesePinyin':
            atom_generator = self.create_generator('ChinesePinyinSyllables')
            word_length_distribution = distributions.CategoricalDistribution([0,0.1,0.5,0.3,0.1])
            return AtomBasedWordGenerator( atom_generator, word_length_distribution )
        elif generator_name == 'ChinesePinyinPrepositions':
            wordlist = ['limian','shangmian','near','under','houmian','pangbian','pangbian','behind']
            prefix_generator = self.create_generator('ChinesePinyinSyllables')
            suffix_generator = ListBasedWordGenerator( ['bian','mian'] )
            default_generator = PrefixSuffixWordGenerator( prefix_generator, suffix_generator )
            return IteratingWordGenerator( wordlist, default_generator )
        else:
            raise WordGeneratorInitError('word generator not recognized: ' + str(generator_name))            

