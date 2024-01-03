from config import *
from helper import format_number

import os 
import pandas as pd 
import tensorflow as tf 

def get_result_dir(config, corp_len, combo):
    form_len    = format_number(corp_len)
    folder_name = f"{'_'.join(combo)}_{form_len}"
    result_dir  = f"{config.RESULTS_PATH}/{folder_name}"  
    return result_dir

def get_translations(config, corp_len, combo, write=False):
    """
    Extract translations and optionally write to a file.
    Function to extract translations from a provided file and optionally write to a file.

    Returns:
    - List of tuples (src, tgt, hyp) 
    """
    result_dir = get_result_dir(config, corp_len, combo) 
    src_fpath  = f'{result_dir}/translations.src'
    ref_fpath  = f"{result_dir}/translations.ref"
    hyp_fpath  = f"{result_dir}/translations.hyp"
    src, tgt   = combo
    save_path  = f'{config.EXP_PATH}/example_outputs.txt'
    
    src_trans = []
    with open(src_fpath, 'r') as file:
        src_trans = [line.strip() for line in file.readlines()]

    ref_trans = []
    with open(ref_fpath, 'r') as file:
        ref_trans = [line.strip() for line in file.readlines()]

    hyp_trans = []
    with open(hyp_fpath, 'r') as file:
        hyp_trans = [line.strip() for line in file.readlines()] 

    translations = list(zip(src_trans, ref_trans, hyp_trans))

    if write and save_path:
        with open(save_path, 'w') as f:
            f.write(f"SRC: {src}, TGT: {tgt}, LEN: {corp_len}\n")
            for s, t, h in translations:
                f.write("===========================\n")
                f.write(f"{s}\n{t}\n{h}\n")

    return translations

def get_accuracy(translations, config, combo):
    """
    function to handle unknown tokens in translations.

    Parameters:
    - translations (list): List of tuples containing (src, tgt, hyp) sentences.
    - word_order (str): A permutation of "SVO" indicating the order of subject, verb, and object.
    - articles (bool): Indicates if articles are present (True) or not (False).

    Returns:
    - A dictionary mapping each subset of "SVO" to its normalized count of matches between tgt and hyp.
    """
    articles   = config.articles
    word_order = combo[1] #SVO 

    # Dictionary to store counts of matches for each subset of "SVO"
    accuracy_counts = {"S": 0, "V": 0, "O": 0, "SV": 0, "VO": 0, "SO": 0, "SVO": 0}

    for _, tgt, hyp in translations:
        if articles:
            tgt = tgt.replace('the', '')
            hyp = hyp.replace("the", '')

        # Replace unknown tokens with 'UNK'
        # sometimes this bugs if unk is the last token 
        # or if the words don't exactly match up 
        tgt = tgt.replace("<<unk>> ", "UNK")
        hyp = hyp.replace("<<unk>> ", "UNK")

        tgt_words = tgt.split()
        hyp_words = hyp.split()

        # Ensure both sentences have exactly 3 words
        if len(tgt_words) != 3 or len(hyp_words) != 3:
            continue

        # Map each word in tgt and hyp to S, V, O based on word_order
        tgt_mapped = {word_order[i]: tgt_words[i] for i in range(3)}
        hyp_mapped = {word_order[i]: hyp_words[i] for i in range(3)}

        # Check for matches in each subset and increment counts
        for subset in accuracy_counts.keys():
            if all(tgt_mapped.get(pos) == hyp_mapped.get(pos) for pos in subset):
                accuracy_counts[subset] += 1

    total_sentences = len(translations)
    # Normalize the counts by the total number of sentences
    normalized_accuracy = {subset: count / total_sentences for subset, count in accuracy_counts.items()}

    return normalized_accuracy


def add_data(config, corp_len, combo):
    result_dir = get_result_dir(config, corp_len, combo) 
    score_fpath = f"{result_dir}/scores"
    tb_fpath    = f"{result_dir}/tensorboard_logs"

    try:
        with open(score_fpath) as f:
            contents = eval(f.read())
            bleu = contents[0]['score']
            chrF2 = contents[1]['score']

        num_steps = None
        for section in ['train']: # for now just getting num_steps 'train_inner', 'valid']:
            tb_fpath = f"{tb_fpath}/{section}"
            for event_file in os.listdir(tb_fpath):
                event_path = os.path.join(tb_fpath, event_file)
                last = list(tf.compat.v1.train.summary_iterator(event_path))[-1]
                num_steps = last.step

        translations       = get_translations(config, corp_len, combo)
        S,V,O,SV,VO,SO,SVO = list(get_accuracy(translations, config, combo).values())


        return [corp_len, combo[0], combo[1], bleu, chrF2, num_steps, S, V, O, SV, VO, SO, SVO]

    except Exception:
        # hasn't been processed yet
        return None 

def create_df(config):
    ### creates a results.csv based on config experiment results
    all_combos = config.combos + [combo[::-1] for combo in config.combos] 
    data = [add_data(config, corp_len, combo)
            for corp_len in config.corp_lens 
            for combo in all_combos]

    # filter out unprocessed results
    data = [d for d in data if d is not None]
    df = pd.DataFrame(data, columns=[
        'corpus_length', 
        'src', 
        'tgt', 
        'bleu', 
        'chrF2', 
        'num_steps',
        's_acc',
        'v_acc',
        'o_acc',
        'sv_acc',
        'vo_acc',
        'so_acc',
        'svo_acc'
        ])
    csv_path = f"{config.EXP_PATH}/results.csv"
    df.to_csv(csv_path, index=False)

if __name__ == "__main__":
    config = SVOConfig()
    create_df(config)

