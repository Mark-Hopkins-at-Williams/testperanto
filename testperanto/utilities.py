##
# utilities.py
# Methods for running the Testperanto pipeline.
# $Author: mhopkins $
# $Revision: 34246 $
# $Date: 2012-05-23 18:26:36 -0700 (Wed, 23 May 2012) $
##

import pickle
import time
import sys

from testperanto import trees
from testperanto import vocabulary
from testperanto import macros
from testperanto import voicebox


def init_macro_from_yaml(macro_file, yaml_file):
    """Initializes a tree transducer macro from a YAML configuration and pickles it to the specified file."""
    yaml_handle = open(yaml_file, 'rb')
    transducer_macro = macros.construct_tree_transducer_from_yaml( yaml_handle )
    yaml_handle.close()
    pickle.dump(transducer_macro, open(macro_file, 'wb'))


def init_macro_from_xml(macro_file, xml_file):
    """Initializes a tree transducer macro from an XML configuration and pickles it to the specified file."""
    transducer_macro = macros.construct_tree_transducer_from_xml( xml_file )
    pickle.dump(transducer_macro, open(macro_file, 'wb'))


def init_voicebox(voicebox_file, vb_name):
    """Initializes a voicebox based on a preconfigured word style."""
    vocab = vocabulary.TpoVocab()
    tok_parser = vocabulary.TpoVocabTokenParser(vocab)
    vb_factory = voicebox.VoiceboxFactory()
    vbox = vb_factory.create_voicebox(vb_name)
    pickle.dump(vbox, open(voicebox_file, 'wb'))



def generate_sentences(macro_file_sequence, voicebox_file, num_to_generate, automaton_start_state = '$qstart', outstream=sys.stdout, outfile=None):
    """Generate sentences using the specified sequence of transducer macros.
    
    The first macro must be an automaton macro.
    """
    if outfile != None:
        outstream = open(outfile, 'w')
    macros = [pickle.load(open(macro_file, 'rb')) for macro_file in macro_file_sequence]
    vbox = pickle.load(open(voicebox_file, 'rb'))
    start_time = time.time()
    for sentence_index in range(num_to_generate):
        in_tree_str = str(automaton_start_state)
        for macro in macros:
            in_tree = trees2.construct_node_based_tree_from_string(in_tree_str, macro.tok_parser)
            out_tree = macro.transduce(in_tree)
            in_tree_str = '($qstart ' + out_tree.to_string(macro.tok_parser)  + ')'
        vocab = vocabulary.TpoVocab()  # TODO: bundle these into vbox
        tok_parser = vocabulary.TpoVocabTokenParser(vocab)
        in_tree = trees2.construct_node_based_tree_from_string(in_tree_str, tok_parser) # should be vbox's tok parser
        outstream.write( vbox.express(in_tree, vocab, tok_parser, '') + '\n' ) # should be vbox's tok parser
        if sentence_index>0 and sentence_index%1000 == 0:
            display_generation_rate(sentence_index, time.time()-start_time)
    for macro_index in range(len(macro_file_sequence)):
        pickle.dump(macros[macro_index], open(macro_file_sequence[macro_index], 'wb'))
    pickle.dump(vbox, open(voicebox_file, 'wb'))    
    if outfile != None:
        outstream.close()

def display_generation_rate(generated_so_far, seconds_elapsed):
    generation_rate = (generated_so_far/seconds_elapsed) * 3600
    sys.stderr.write('Generated ' + str(generated_so_far) + ' items (generation rate: ' + str(int(generation_rate)) + ' per hour)\n')
    sys.stderr.flush()
    

def run_automaton(macro_file, num_trees_to_generate, automaton_start_state = '$qstart', outstream=sys.stdout, outfile=None):
    """Generate trees from the specified tree automaton.
    
    The trees are printed to stdout unless outfile is specified. Logging info printed to stderr.
    """
    if outfile != None:
        outstream = open(outfile, 'w')
    automaton = pickle.load(open(macro_file, 'rb')) # Load the macro from file.
    in_tree_str = automaton_start_state
    in_tree = trees2.construct_node_based_tree_from_string(in_tree_str, automaton.tok_parser)
    start_time = time.time()
    for i in range(num_trees_to_generate):
        if i>0 and i%1000 == 0:
            seconds_elapsed = time.time()-start_time
            generation_rate = (i/seconds_elapsed) * 3600
            sys.stderr.write('Generated ' + str(i) + ' trees (generation rate: ' + str(int(generation_rate)) + ' per hour)\n')
            sys.stderr.flush()
        out_tree = automaton.transduce(in_tree)
        outstream.write(out_tree.to_string(automaton.tok_parser) + '\n')
    pickle.dump(automaton, open(macro_file, 'wb')) # Update the macro file with its new state.
    if outfile != None:
        outstream.close()


def run_transducer(macro_file, num_trees_to_generate=1, start_state='$qstart', instream=sys.stdin, outstream=sys.stdout, infile=None, outfile=None):
    """Generate output trees from the input tree stream, using the specified transducer macro.
    
    You can generate multiple output trees from a single input stream by changing the
    value of num_trees_to_generate.
    
    Input trees are read from stdin unless infile is specified.
    Output trees are written to stdout unless outfile is specified.
    Logging info is written to stderr.

    """
    if infile != None:
        instream = open(infile, 'r')
    if outfile != None:
        outstream = open(outfile, 'w')
    transducer = pickle.load(open(macro_file, 'rb'))
    num_output_trees_so_far = 0
    start_time = time.time()
    for in_tree_str in instream:
        in_tree_str = '(' + start_state + ' '+ in_tree_str.strip() + ')'
        in_tree = trees2.construct_node_based_tree_from_string(in_tree_str, transducer.tok_parser)
        for j in range(num_trees_to_generate):
            num_output_trees_so_far += 1
            if num_output_trees_so_far%1000 == 0:
                seconds_elapsed = time.time()-start_time
                if seconds_elapsed > 1:
                    generation_rate = (num_output_trees_so_far/seconds_elapsed) * 3600
                    sys.stderr.write('Generated ' + str(num_output_trees_so_far) + ' trees (generation rate: ' + str(int(generation_rate)) + ' per hour)\n')
                    sys.stderr.flush()
            out_tree = transducer.transduce(in_tree)
            outstream.write( out_tree.to_string(transducer.tok_parser) + '\n' )
        if num_trees_to_generate > 1:
            outstream.write( '>>> END REFS\n' )
    pickle.dump(transducer, open(macro_file, 'wb'))
    if infile != None:
        instream.close()
    if outfile != None:
        outstream.close()


def run_voicebox(voicebox_file, instream=sys.stdin, outstream=sys.stdout, infile=None, outfile=None):
    """Runs a voicebox on an input stream of parse trees."""
    if infile != None:
        instream = open(infile, 'r')
    if outfile != None:
        outstream = open(outfile, 'w')
    vbox = pickle.load(open(voicebox_file, 'rb'))
    start_time = time.time()
    vocab = vocabulary.TpoVocab()
    tok_parser = vocabulary.TpoVocabTokenParser(vocab)
    num_trees_so_far = 0
    for in_tree_str in instream:
        num_trees_so_far += 1
        if num_trees_so_far % 1000 == 0:
            seconds_elapsed = time.time()-start_time
            generation_rate = (num_trees_so_far/seconds_elapsed) * 3600
            sys.stderr.write('Generated ' + str(num_trees_so_far) + ' sentences (generation rate: ' + str(int(generation_rate)) + ' per hour)\n')
            sys.stderr.flush()

        in_tree_str = in_tree_str.strip()
        if in_tree_str.startswith('>>>'):
            outstream.write( in_tree_str + '\n' )
        else:
            in_tree = trees2.construct_node_based_tree_from_string(in_tree_str, tok_parser)
            outstream.write( vbox.express(in_tree, vocab, tok_parser,'').strip() + '\n' )
    pickle.dump(vbox, open(voicebox_file, 'wb'))
    if infile != None:
        instream.close()
    if outfile != None:
        outstream.close()

