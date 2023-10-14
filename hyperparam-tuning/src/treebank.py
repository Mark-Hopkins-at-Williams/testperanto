import os
import json
import random

SRC_PATH  = os.getcwd()
MAIN_PATH = os.path.dirname(SRC_PATH)
PER_PATH  = os.path.dirname(MAIN_PATH)
DATA_PATH = f"{MAIN_PATH}/data"

def read_conllu(filename=None):
    """
    Reads a CoNLL-U formatted file and returns sentences.
    Each sentence is a list of tokens, and each token is a dictionary.
    """
    if filename is None:
        filename = f"{DATA_PATH}/treebank.conllu"
    
    sentences = []
    with open(filename, 'r', encoding='utf-8') as f:
        sentence = []
        for line in f:
            line = line.strip()
            if line.startswith("#"):
                continue
            elif not line:
                if sentence:
                    sentences.append(sentence)
                    sentence = []
            else:
                fields = line.split("\t")
                token = {
                    'id':     fields[0],
                    'lemma':  fields[2].lower(),
                    'upos':   fields[3],
                    'head':   fields[6],
                    'deprel': fields[7]
                    }
                sentence.append(token)
        if sentence:  # if file doesn't end with a newline
            sentences.append(sentence)
    return sentences   

def scrape_svo(sentences=None):
    """
    Extracts (S,V,O) triples from sentences.
    """
    if sentences is None:
        sentences = read_conllu()
        
    triples = []
    for sentence in sentences:
        verbs = [token for token in sentence if token['upos'] == 'VERB']
        for verb in verbs:
            subjects = [token for token in sentence if token['head'] == verb['id'] and 'subj' in token['deprel']]
            objects = [token for token in sentence if token['head'] == verb['id'] and 'obj' in token['deprel']]
            for subj in subjects:
                for obj in objects:
                    triple = (
                        subj['lemma'], 
                        verb['lemma'], 
                        obj['lemma'], 
                        subj['upos'] == 'PRON',  
                        obj['upos'] == 'PRON'    
                    )
                    triples.append(triple)
    random.shuffle(triples) # shuffle once 
    return triples

def save_svo(triples=None, filename=None):
    if triples is None:
        triples = scrape_svo()
    if filename is None:
        filename = f"{DATA_PATH}/scraped_treebank.json"

    if os.path.exists(filename):
        # if you really need to redo this get rid of this if statement
        print("File already exists. Don't rerun if already running experiments to avoid bias with randomizing.")
        return None 

    # convert to dictionaries 
    json_data = [{
        "subject"   : item[0],
        "verb"      : item[1],
        'object'    : item[2],
        'subj_pron' : item[3],
        'obj_pron'  : item[4]
    } for item in triples]

    # Write the list of dictionaries to a JSON file
    with open(filename, 'w') as file:
        json.dump(json_data, file, indent=4)

class IdiotException(Exception):
    pass

class TripleStore:
    def __init__(self):
        self.data = []    
        self.load_from_file()
    
    def add_triple(self, s, v, o, subj_pron=False, obj_pron=False):
        self.data.append({
            'subject'  : s,
            'verb'     : v,
            'object'   : o,
            'subj_pron': subj_pron,
            'obj_pron' : obj_pron
            })
    
    def _retrieve(self, parts, pronoun_filter=None):
        filter_fn = self._get_filter_function(pronoun_filter)
        return [tuple(item[part] for part in parts if part) for item in self.data if filter_fn(item)]
    
    def _get_filter_function(self, pronoun_filter):
        if pronoun_filter == 'both':
            return lambda item: item['subj_pron'] and item['obj_pron']
        elif pronoun_filter == 'either':
            return lambda item: item['subj_pron'] or item['obj_pron']
        elif pronoun_filter == 'subject':
            return lambda item: item['subj_pron']
        elif pronoun_filter == 'object':
            return lambda item: item['obj_pron']
        else:
            return lambda item: True

    # Specific retrieval methods
    def get_triples(self, pronoun_filter=None):
        return self._retrieve(['subject', 'verb', 'object'], pronoun_filter)
    
    def get_sv_pairs(self, pronoun_filter=None):
        return self._retrieve(['subject', 'verb'], pronoun_filter)
    
    def get_vo_pairs(self, pronoun_filter=None):
        return self._retrieve(['verb', 'object'], pronoun_filter)
    
    def get_subjects(self, pronoun_filter=None):
        return self._retrieve(['subject'], pronoun_filter)
    
    def get_verbs(self, pronoun_filter=None):
        return self._retrieve(['verb'], pronoun_filter)
    
    def get_objects(self, pronoun_filter=None):
        return self._retrieve(['object'], pronoun_filter)
    
    def get_nouns(self, pronoun_filter=None):
        subjects = self._retrieve(['subject'], pronoun_filter)
        objects = self._retrieve(['object'], pronoun_filter)
        return list(set(subjects + objects))
    
    def get(self, distribution=None, pronoun_filter=None):
        if distribution == 'vb':
            return self.get_verbs(pronoun_filter=pronoun_filter)

        elif distribution == 'nn':
            return self.get_nouns(pronoun_filter=pronoun_filter)

        elif distribution == 'nn.arg0':
            return self.get_subjects(pronoun_filter=pronoun_filter)

        elif distribution == 'nn.arg1':
            return self.get_objects(pronoun_filter=pronoun_filter)

        elif distribution == "nn.arg0.$y0":
            return self.get_sv_pairs(pronoun_filter=pronoun_filter)
        
        elif distribution == 'nn.arg1.$y0':
            return self.get_vo_pairs(pronoun_filter=pronoun_filter)

        else:
            raise IdiotException(f"You stupid piece of crap this isn't a distribution get your head out of your ass.")
            
    def load_from_file(self, filename=None):
        if filename is None:
            filename = f"{DATA_PATH}/scraped_treebank.json"
        with open(filename, 'r') as f:
            self.data = json.load(f)


if __name__ == "__main__":
    store = TripleStore()
    triples = store.get_triples()
    print(len(triples))
    # file_path = "../../../../../../home/data/treebanks/UD_English-EWT/en_ewt-ud-train.conllu"
    # sentances = read_conllu(file_path)
    # tripples = scrape_svo(sentances)
    # save_svo(tripples)
    # store = TripleStore() 
    # results = {
    #     'triples'       : store.get_triples(),
    #     'sv_pairs'      : store.get_sv_pairs(),
    #     'sv_pairs_pron' : store.get_sv_pairs(pronoun_filter="both"),
    #     'triples_pron'  : store.get_triples(pronoun_filter="both")
    #     }

    # print("SVO Triples")
    # for triple in results['triples'][:10]:
    #     print(triple)

    # print("\n\nSV Pairs")
    # print('=' * 20)
    # for sv_pair in results['sv_pairs'][:10]:
    #     print(sv_pair)

    # print("\n\nSV Pairs (S pronoun)")
    # print('=' * 20)
    # for sv_pair in results['sv_pairs_pron'][:10]:
    #     print(sv_pair)


    # print("\n\nTriples (S & O pronouns)")
    # print('=' * 20)
    # for triple in results['triples_pron'][:10]:
    #     print(triple)

