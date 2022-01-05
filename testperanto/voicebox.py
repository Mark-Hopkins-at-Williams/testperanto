##
# voicebox.py
# $Author: mhopkins $
# $Revision: 33130 $
# $Date: 2012-04-27 15:56:28 -0700 (Fri, 27 Apr 2012) $
##


from testperanto import wordgenerators
from abc import ABC, abstractmethod
from testperanto.morphology import SuffixMorphologizer, EnglishVerbMorpher, EnglishNounMorpher

class Voicebox(ABC):
    """A Voicebox flattens a syntax tree into a string representation (sentence)."""

    @abstractmethod
    def express(self, syntax_tree):
        """Converts a syntax tree into a string."""
        raise NotImplementedError("Implement me in your subclass.")

    def run(self, syntax_tree):
        return self.express(syntax_tree)


class VoiceboxExpressionError(Exception): pass


def read_preterminal_tree(tree):
    if tree.get_num_children() != 1:
        raise VoiceboxExpressionError('badly structured syntax tree: {}'.format(tree))
    leaf_tree = tree.get_child(0)
    if leaf_tree.get_num_children() > 0:
        raise VoiceboxExpressionError('badly structured syntax tree: {}'.format(tree))
    root_label = '~'.join(tree.get_label())
    leaf_label = '~'.join(leaf_tree.get_label())
    return root_label, leaf_label


def read_property_tree(tree):
    properties = dict()
    for child in tree.get_children():
        preterminal, leaf = read_preterminal_tree(child)
        properties[preterminal] = leaf
    return properties


class ManagingVoicebox(Voicebox):
    def __init__(self):
        super().__init__()
        self.listeners = dict()

    def express(self, syntax_tree):
        tok_str = '~'.join(syntax_tree.get_label())
        if tok_str[0] == '@':
            voicebox_code = tok_str[1:]
            if voicebox_code not in self.listeners:
                if 'default' in self.listeners:
                    voicebox_code = 'default'
                else:
                    raise VoiceboxExpressionError('voicebox code not recognized: ' + str(voicebox_code))
            rendering = self.listeners[voicebox_code].express(syntax_tree)
        else:
            segments = [self.express(syntax_tree.get_child(i)).strip()
                        for i in range(syntax_tree.get_num_children())]
            rendering = ' '.join(segments)
        return rendering

    def register_listener(self, name, listener_voicebox):
        self.listeners[name] = listener_voicebox
        

class MorphologyVoicebox(Voicebox):

    def __init__(self, stem_generator, morphologizers=[]):
        self.morphologizers = morphologizers
        self.stem_generator = stem_generator
        self.lexicon = dict()

    def express(self, syntax_tree):
        properties = read_property_tree(syntax_tree)
        if 'STEM' not in properties:
            word = ""
        else:
            if properties['STEM'] not in self.lexicon:
                self.lexicon[properties['STEM']] = self.stem_generator.generate()
            word = self.lexicon[properties['STEM']]
        for morphologizer in self.morphologizers:
            word = morphologizer.mutate(word, properties)
        return word


class EnglishDeterminerVoicebox(Voicebox):
    def __init__(self):
        morph = SuffixMorphologizer(properties=('COUNT', 'DEF'),
                                    suffix_map={('sng', 'def'): 'the',
                                                ('plu', 'def'): 'these',
                                                ('sng', 'indef'): 'a',
                                                ('plu', 'indef'): ''})
        self.base_vbox = MorphologyVoicebox(None, [morph])

    def express(self, syntax_tree):
        return self.base_vbox.express(syntax_tree)


class VerbatimVoicebox(Voicebox):

    def express(self, syntax_tree):
        _, leaf_label = read_preterminal_tree(syntax_tree)
        return leaf_label


class WordGeneratorVoicebox(Voicebox):
    
    def __init__(self, word_generator):
        super().__init__()
        self.lexicon = dict()
        self.word_generator = word_generator

    def express(self, syntax_tree):
        _, leaf_label = read_preterminal_tree(syntax_tree)
        if leaf_label not in self.lexicon:
            self.lexicon[leaf_label] = self.word_generator.generate()
        return self.lexicon[leaf_label]


class VoiceboxFactory(object):
    """A VoiceboxFactory generates prefab Voiceboxes."""

    def __init__(self):
        self.generator_factory = wordgenerators.WordGeneratorFactory()
    
    def create_voicebox(self, voicebox_name):
        manager = ManagingVoicebox()
        manager.register_listener('default', VerbatimVoicebox())
        if voicebox_name == 'seuss':
            manager.register_listener('vb', MorphologyVoicebox(self.generator_factory.create_generator('SeussVerbs'), [EnglishVerbMorpher()]))
            manager.register_listener('nn', MorphologyVoicebox(self.generator_factory.create_generator('Seuss'), [EnglishNounMorpher()]))
            manager.register_listener('adj', MorphologyVoicebox(self.generator_factory.create_generator('SeussAdjectives')))
            manager.register_listener('adv', MorphologyVoicebox(self.generator_factory.create_generator('SeussAdverbs')))
            manager.register_listener('prep', MorphologyVoicebox(self.generator_factory.create_generator('EnglishPrepositions')))
            manager.register_listener('dt', EnglishDeterminerVoicebox())
        else:
            raise Exception('voicebox code not recognized: ' + str(voicebox_name))
        return manager

