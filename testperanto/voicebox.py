##
# voicebox.py
# Data structures and algorithms for voiceboxes.
##

from abc import ABC, abstractmethod
from testperanto.globals import COMPOUND_SEP, EMPTY_STR, VOICEBOX_THEMES
from testperanto.morphology import SuffixMorpher, EnglishVerbMorpher, EnglishNounMorpher
from testperanto.morphology import JapaneseVerbMorpher
from testperanto.trees import TreeNode
from testperanto.wordgenerators import lookup_word_generator


def register_voicebox_theme(name, theme):
    """Registers a custom voicebox theme with the testperanto package.

    Once a voicebox_theme is registered with the name X, its constructor function can
    be subsequently looked up using testperanto.voicebox.lookup_voicebox_theme(X).

    Parameters
    ----------
    name : str
        The id to associate with the voicebox theme
    theme : constructor
        Constructor function for the voicebox theme
    """
    VOICEBOX_THEMES[name] = theme


def lookup_voicebox_theme(name):
    """Retrieves a registered voicebox theme.

    Parameters
    ----------
    name : str
        The id associated with the voicebox theme during its registration
    """
    if name in VOICEBOX_THEMES:
        return VOICEBOX_THEMES[name]()
    else:
        raise VoiceboxInitError("Theme not recognized: {}".format(name))


class Voicebox(ABC):
    """Renders the terminal structures of a syntax tree.

    Methods
    -------
    run(syntax_tree)
        Renders the terminal structures of a syntax tree. Abstract method.
    """

    @abstractmethod
    def run(self, syntax_tree):
        """Renders the terminal structures of a syntax tree. Abstract method.

        Parameters
        ----------
        syntax_tree : testperanto.trees.TreeNode
            Root of the syntax tree

        Returns
        -------
        testperanto.trees.TreeNode
            A new syntax tree, equivalent to the input tree, but with the
            terminal structures rendered as words.
        """


class VoiceboxInitError(Exception):
    """Thrown if there is an issue with initializing a voicebox theme."""


class VoiceboxExpressionError(Exception):
    """Thrown if the syntax tree input to a voicebox is badly formed."""


def read_preterminal_tree(tree):
    """Reads the two symbols of a two-node tree of the form (x y).

    Parameters
    ----------
    tree : testperanto.trees.TreeNode
        Root of the tree

    Returns
    -------
    str , str
        The root and leaf labels of the tree (respectively)

    Raises
    ------
    VoiceboxExpressionError
        If the input tree is not a two-node tree of the form (x y)
    """

    if tree.get_num_children() != 1:
        raise VoiceboxExpressionError('badly structured syntax tree: {}'.format(tree))
    leaf_tree = tree.get_child(0)
    if leaf_tree.get_num_children() > 0:
        raise VoiceboxExpressionError('badly structured syntax tree: {}'.format(tree))
    root_label = COMPOUND_SEP.join(tree.get_label())
    leaf_label = COMPOUND_SEP.join(leaf_tree.get_label())
    return root_label, leaf_label


def read_terminal_structure(tree):
    """Extracts properties of a tree of the form (@label (k_1 v_1) (k_2 v_2) ... (k_n v_n)).

    Note: here (and elsewhere in this file), we refer to a tree of the form
    (@label (k_1 v_1) (k_2 v_2) ... (k_n v_n)) as a "terminal structure".

    Parameters
    ----------
    tree : testperanto.trees.TreeNode
        Root of the tree

    Returns
    -------
    dict
        A dictionary that maps each key k_i to its value v_i

    Raises
    ------
    VoiceboxExpressionError
        If the input tree is not a well-formed property tree
    """

    properties = dict()
    for child in tree.get_children():
        preterminal, leaf = read_preterminal_tree(child)
        properties[preterminal] = leaf
    return properties


class ManagingVoicebox(Voicebox):
    """Renders the terminal structures of a syntax tree.

    The ManagingVoicebox does all of its transformations through helper voiceboxes.
    It traverses the tree, looking for nodes with a label starting with the character "@".
    The subtrees rooted at the @-nodes are sent to delegate voiceboxes, which transform
    those subtrees.

    Methods
    -------
    run(syntax_tree)
        Renders all terminal structures of the syntax tree using the delegated
        sub-voiceboxes.

    delegate(name, vbox)
        Registers a helper voicebox to render terminal structures headed by
        the label @name
    """

    def __init__(self):
        super().__init__()
        self.helpers = dict()

    def delegate(self, name, helper_vbox):
        """Enlists a helper voicebox to render terminal structures headed by @name.

        Once enlisted, the ManagingVoicebox will use the helper voicebox to render all
        subtrees of the form (@name (k_1 v_1) (k_2 v_2) ... (k_n v_n)).

        Parameters
        ----------
        name : str
            Root of the terminal structure
        helper_vbox : Voicebox
            Voicebox that will render any terminal structures headed by the label @name.
        """
        self.helpers[name] = helper_vbox

    def run(self, syntax_tree):
        """Renders all terminal structures using the helper sub-voiceboxes.

        Parameters
        ----------
        syntax_tree : testperanto.trees.TreeNode
            Root of the syntax tree

        Returns
        -------
        testperanto.trees.TreeNode
            A new syntax tree, equivalent to the input tree, but with the terminal
            structures rendered as words (using the helper voiceboxes)
        """

        tok_str = COMPOUND_SEP.join(syntax_tree.get_label())
        if tok_str[0] == '@':
            voicebox_code = tok_str[1:]
            if voicebox_code not in self.helpers:
                raise VoiceboxExpressionError('voicebox code not recognized: ' + str(voicebox_code))
            rendering = self.helpers[voicebox_code].run(syntax_tree)
            return rendering
        else:
            subtrees = [self.run(syntax_tree.get_child(i))
                        for i in range(syntax_tree.get_num_children())]
            result = TreeNode()
            result.label = syntax_tree.get_label()
            result.children = subtrees
            return result


class MorphologyVoicebox(Voicebox):
    """Renders a single terminal structure with morphological properties.

    Note: a "terminal structure" is a tree of the form
    (@label (key_1 value_1) (key_2 value_2) ... (key_n value_n))

    Methods
    -------
    run(syntax_tree)
        Renders a single terminal structure with morphological properties.
    """

    def __init__(self, stem_generator, morphers=[]):
        """
        Parameters
        ----------
        stem_generator : testperanto.wordgenerators.WordGenerator
            Generates the word stems.
        morphers : list[testperanto.morphology.Morpher]
            Morphers to be applied (in order) to the generated stem.
        """

        self.morphers = morphers
        self.stem_key = "STEM"
        self.stem_generator = stem_generator
        self.lexicon = dict()

    def run(self, syntax_tree):
        """Renders a single terminal structure with morphological properties.

        Note: a "terminal structure" is a tree of the form
        (@label (key_1 value_1) (key_2 value_2) ... (key_n value_n))

        If the value associated with the key "STEM" has previously been rendered,
        then the cached rendering is re-used. Otherwise, a new value is provided
        by the stem generator. The rendered stem is processed sequentially by each
        morpher to produce the final result.

        Parameters
        ----------
        syntax_tree : testperanto.trees.TreeNode
            Root of the terminal structure

        Returns
        -------
        testperanto.trees.TreeNode
            The rendering of the terminal structure as a word
        """

        properties = read_terminal_structure(syntax_tree)
        if self.stem_key not in properties:
            word = ""
        else:
            if properties[self.stem_key] not in self.lexicon:
                self.lexicon[properties[self.stem_key]] = self.stem_generator.generate()
            word = self.lexicon[properties[self.stem_key]]
        for morpher in self.morphers:
            word = morpher.morph(word, properties)
        return TreeNode.from_str(word)


class VerbatimVoicebox(Voicebox):
    """Renders a simple "preterminal" subtree of the form (@label leaf).

    It replaces this subtree with the leaf.

    Methods
    -------
    run(syntax_tree)
        Renders a tree of the form (@label X) as a single node labeled X.
    """

    def run(self, syntax_tree):
        """Renders a tree of the form (@label X) as a single node labeled X.

        Parameters
        ----------
        syntax_tree : testperanto.trees.TreeNode
            Root of the tree

        Returns
        -------
        testperanto.trees.TreeNode
            A single node labeled with the leaf label of the input tree
        """

        _, leaf_label = read_preterminal_tree(syntax_tree)
        return TreeNode.from_str(leaf_label)


class VoiceboxTheme(ABC):
    """Constructs preconfigured voiceboxes.

    Methods
    -------
    init_vbox()
        Constructs a preconfigured voicebox.
    """

    @abstractmethod
    def init_vbox(self):
        """Constructs a preconfigured voicebox.

        Returns
        -------
        testperanto.voicebox.Voicebox
            The preconfigured voicebox
        """


class MotherGooseTheme(VoiceboxTheme):
    """A voicebox theme that generates nursery rhyme-esque words."""

    def init_vbox(self):
        vbox = ManagingVoicebox()
        glookup = lookup_word_generator
        vbox.delegate('verbatim', VerbatimVoicebox())
        vbox.delegate('vb', MorphologyVoicebox(glookup('GooseVerbs'), [EnglishVerbMorpher()]))
        vbox.delegate('nn', MorphologyVoicebox(glookup('Goose'), [EnglishNounMorpher()]))
        vbox.delegate('adj', MorphologyVoicebox(glookup('GooseAdjectives')))
        vbox.delegate('adv', MorphologyVoicebox(glookup('GooseAdverbs')))
        vbox.delegate('prep', MorphologyVoicebox(glookup('EnglishPrepositions')))
        dt_morph = SuffixMorpher(property_names=('COUNT', 'DEF'),
                                 suffix_map={('sng', 'def'): 'the',
                                             ('plu', 'def'): 'these',
                                             ('sng', 'indef'): 'a',
                                             ('plu', 'indef'): EMPTY_STR})
        vbox.delegate('dt', MorphologyVoicebox(None, [dt_morph]))
        return vbox


class JapaneseTheme(VoiceboxTheme):
    """A voicebox theme that generates words using Romanized Japanese syllables."""

    def init_vbox(self):
        vbox = ManagingVoicebox()
        glookup = lookup_word_generator
        vbox.delegate('verbatim', VerbatimVoicebox())
        vbox.delegate('vb', MorphologyVoicebox(glookup('JapaneseStems'), [JapaneseVerbMorpher()]))
        vbox.delegate('nn', MorphologyVoicebox(glookup('JapaneseStems')))
        vbox.delegate('adj', MorphologyVoicebox(glookup('JapaneseStems')))
        vbox.delegate('adv', MorphologyVoicebox(glookup('JapaneseStems')))
        vbox.delegate('prep', MorphologyVoicebox(glookup('JapanesePrepositions')))
        dt_morph = SuffixMorpher(property_names=('COUNT', 'DEF'),
                                 suffix_map={('sng', 'def'): EMPTY_STR,
                                             ('plu', 'def'): EMPTY_STR,
                                             ('sng', 'indef'): EMPTY_STR,
                                             ('plu', 'indef'): EMPTY_STR})
        vbox.delegate('dt', MorphologyVoicebox(None, [dt_morph]))
        return vbox

class GermanTheme(VoiceboxTheme):

    def init_vbox(self):
        vbox = ManagingVoicebox()
        verb_morpher = SuffixMorpher(property_names=('COUNT',),
                                     suffix_map={('sng',): 'e', ('plu',): 'en'})
        noun_morpher = SuffixMorpher(property_names=('COUNT',),
                                     suffix_map={('sng',): '', ('plu',): 'en'})
        vbox.delegate('vb', MorphologyVoicebox(lookup_word_generator('german-stems'), [verb_morpher]))
        vbox.delegate('nn', MorphologyVoicebox(lookup_word_generator('german-stems'), [noun_morpher]))
        dt_morph = SuffixMorpher(property_names=('COUNT', 'CASE', 'GENDER'),
                                 suffix_map={('sng', 'nom', 'm'): 'der',
                                             ('plu', 'nom', 'm'): 'die',
                                             ('sng', 'acc', 'm'): 'den',
                                             ('plu', 'acc', 'm'): 'die',
                                             ('sng', 'nom', 'f'): 'die',
                                             ('plu', 'nom', 'f'): 'die',
                                             ('sng', 'acc', 'f'): 'die',
                                             ('plu', 'acc', 'f'): 'die',
                                             ('sng', 'nom', 'n'): 'das',
                                             ('plu', 'nom', 'n'): 'die',
                                             ('sng', 'acc', 'n'): 'das',
                                             ('plu', 'acc', 'n'): 'die'
                                             })
        vbox.delegate('dt', MorphologyVoicebox(None, [dt_morph]))
        return vbox

register_voicebox_theme("deutsch", GermanTheme)

register_voicebox_theme("goose", MotherGooseTheme)
register_voicebox_theme("english", MotherGooseTheme)
register_voicebox_theme("japanese", JapaneseTheme)
