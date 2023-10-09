import re 
import os 

SRC_PATH  = os.getcwd()
MAIN_PATH = os.path.dirname(SRC_PATH)
DATA_PATH = f"{MAIN_PATH}/data"
PER_PATH  = f"{DATA_PATH}/peranto_data"

def scrape_svo(file_path):
    """
    Extracts (S, V, O) triples from CONLLU file of labeled treebank data.
    (S, V, O) triples are Subject, Verb, Object appearances

    Input:
    -------
    filepath (str): path to CONLLU file.

    Output:
    -------
    List[Tuple[Str]]: (S, V, O) triples.
    """

    with open(file_path, 'r') as f:
        content = f.read()
    

    # splitting sentences based on the # sent_id pattern
    sentences = re.split(r'# sent_id = [^\n]+\n', content)[1:]
    triples = []
    count = 0
    for sentence in sentences:
        # Splitting lines and filtering out comment lines
        lines = [line for line in sentence.split('\n') if line and not line.startswith('#')]
        # Dictionary to store token details
        tokens = {}
        for line in lines:
            columns = line.split('\t')
            token_id = columns[0]
            lemma = columns[2]
            upos = columns[3]
            head = columns[6]
            relation = columns[7]
            tokens[token_id] = {'lemma': lemma, 'upos': upos, 'head': head, 'relation': relation}
        
        # Extracting (S, V, O) triples
        for token_id, details in tokens.items():
            if details['upos'] == 'VERB':  # if token is verb 
                subject = None
                obj = None
                a, b = True, True 
                for t_id, d in tokens.items():
                    if d['head'] == token_id: # if relates to verb
                        if 'nsubj' in d['relation']:
                            subject = d['lemma'] 
                            if int(t_id) > int(token_id): # subject after 
                                count += 1
                                a = False
                        elif 'obj' in d['relation']:
                            obj = d['lemma']
                            if int(t_id) < int(token_id): # object before 
                                count += 1
                                b = False
                if subject and obj:
                    verb = details['lemma']
                    if not b: 
                        print(sentence) # weird obj
                        print(f"========{(subject, verb, obj)}========")
                        print("\n\n\n")
                        b = True 
                    triples.append((subject, verb, obj))
    print(count)
    return triples

def write_data(svo_triples):
    """
    Writes svo_triples to a .txt file result.txt.

    Input:
    ------
    svo_triples (list(tuple(str))): list of svo triples
    
    Output:
    ------
    None (writes to a file)
    """
    content = "Subject\tVerb\tObject\n"  
    for (s,v,o) in svo_triples:
        content += f"{s}\t{v}\t{o}\n"
    
    # Write the content to the "treebank.txt" file
    result_path = f"{os.getcwd()}/treebank.txt"
    with open(result_path, 'w') as f:
        f.write(content)
    

def read_data(file_path=None):
    """
    Reads (S, V, O) triples from a given .txt file.
        If none then defaults to result.txt 

    Input:
    ------
    file_path (str): path to .txt file 
    
    Output:
    ------
    svo_triples (list(tuple(str))): list of svo triples
    """
    if file_path is None:
        file_path = f"{os.getcwd()}/treebank.txt"

    with open(file_path, 'r') as f:
        content = f.readlines()
    
    # skip the header and extract triples
    triples = [tuple(line.strip().split('\t')) for line in content[1:]]
    return triples

def is_equal(file_path1, file_path2):
    """
    Compares two .txt files containing (S, V, O) triples to check if they are equal.
    
    Input:
    ------
    file_path1 (str): path to .txt file 1
    file_path2 (str): path to .txt file 2 

    Output:
    ------
    (bool): true iff the two .txt fiels are equal 
    """
    triples1 = read_data(file_path1)
    triples2 = read_data(file_path2)
    
    return set(triples1) == set(triples2)
  
def main():
    print(f"Reading in data...")
    file_path   = f"{os.getcwd()}/data.conllu"
    svo_triples = scrape_svo(file_path)
    write_data(svo_triples)
    print(f"Wrote triples to .txt file")

if __name__ == "__main__":
    file_path   = f"{DATA_PATH}/data.conllu"
    scrape_svo(file_path)
    """
    file_path = f"{os.getcwd()}/treebank.txt"
    if os.path.exists(file_path):
        data = read_data(file_path) #5617
        print(len(data))
        print(len(list(set(data))))
    else:
        main()
    """