##
# util.py
# Miscellaneous utility functions.
##


from testperanto.globals import COMPOUND_SEP


def compound(symbols):
    """Converts a TreeNode label (i.e. a tuple of strings) into a compound symbol.

    Parameters
    ----------
    symbols : tuple[str]
        The TreeNode label

    Returns
    -------
    str
        The compound symbol
    """

    return COMPOUND_SEP.join([str(s) for s in symbols])


def zvar(i):
    """Creates a canonical representation of the ith z-variable.

    Parameters
    ----------
    i : int
        Index of the z-variable

    Returns
    -------
    str
        The canonical representation of variables z_i
    """
    return '$z{}'.format(i)


def is_state(label):
    """Returns whether a TreeNode label (i.e. tuple) represents a transducer state.

    Parameters
    ----------
    label : tuple
        TreeNode label

    Returns
    -------
    bool
        True iff the input string is the canonical representation of some transducer state
    """

    try:
        return label[0][:2] == '$q'
    except Exception:
        return False


def stream_ngrams(lines, ngram_order, tokenize=lambda line: line.split()):
    """Converts a stream of lines into a stream of ngrams.

    Numbers are converted to a canonical generic token.

    Parameters
    ----------
    lines : generator[str]
        the input stream of lines
    ngram_order : int
        desired n
    tokenize : function
        function to split the line into tokens

    Returns
    -------
    generator[str]
        n-grams from the input stream of lines
    """

    def is_number(s):
        try:
            float(s)
            return True
        except ValueError:
            return False

    def replace_numbers(word):
        if is_number(word):
            return "*NUM*"
        else:
            return word

    for line in lines:
        iwords = tokenize(line)
        words = [replace_numbers(wd) for wd in iwords]
        for i in range(0, len(words) - ngram_order + 1):
            ngram = ' '.join(words[i:i+ngram_order])
            yield ngram


def stream_lines(filename):
    """Streams the lines from an input file.

    Parameters
    ----------
    filename : str
        Input file

    Returns
    -------
    generator[str]
        Stream of lines from the input file.
    """

    with open(filename, 'r') as reader:
        for line in reader:
            yield line.strip()
