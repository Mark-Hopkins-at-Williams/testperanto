import os
import tensorflow as tf
import pandas as pd

from globals import * 

def update_csv():
    """
    Updates results.csv with post-training info, namely num_steps, bleu, chrF2. This is done
    directly by scraping the results data. Currently num_steps is just the number in train.
    """
    csv_file_path  = f"{DATA_PATH}/results.csv"
    df             = pd.read_csv(csv_file_path)
    rows_to_update = df[df['bleu'].isna()]

    for index, row in rows_to_update.iterrows():
        result_dir  = f"{RESULTS_PATH}/{row['work_dir']}"
        score_fpath = f"{result_dir}/scores"
        tb_fpath    = f"{result_dir}/tensorboard_logs"

        try:
            ### update bleu/chrf2
            with open(score_fpath) as f:
                contents = eval(f.read())
                df.at[index, 'bleu']  = contents[0]['score']
                df.at[index, 'chrF2'] = contents[1]['score']

            ### update num_steps
            num_steps = None
            for section in ['train']:  
                tb_section_fpath = f"{tb_fpath}/{section}"
                for event_file in os.listdir(tb_section_fpath):
                    event_path = os.path.join(tb_section_fpath, event_file)
                    last = list(tf.compat.v1.train.summary_iterator(event_path))[-1]
                    num_steps = last.step
            df.at[index, 'num_steps'] = num_steps

        except Exception:
            print(f"Haven't trained {row['work_dir']}...")

    df.to_csv(csv_file_path, index=False)

def create_svo_df():
    """
    Creates svo_perm specific dataframe, which includes accuracy. Specifically, let
    A subset {S,V,O}. Then Acc(A) is defined as the proportion of sentences where the ref
    and hyp (real translation/model translation) match for all a in A. This is used since
    for overly simple language bleu is not a good metric. For more info see my writeup in
    the LP Hardness .tex document

    In reality this process is flawed (though more or less correct). To try to match words 
    properly (subj with subj etc.) I get rid of 'the' and replace '<<unk>> ' with UNK. I then
    check if both hyp/ref have 3 words (this is why replacing <<unk>> is necessary as it 
    causes an extra space). If they both do then I assume the word order based on the language 
    and check that way.

    If they don't have 3 words, I assume the worst and say it's a wrong translation.
    """
    update_csv() 
    csv_fpath = f"{DATA_PATH}/results.csv"
    svo_fpath = f'{DATA_PATH}/svo_results.csv'
    df        = pd.read_csv(csv_fpath)
    subset_df = df[~df['bleu'].isna() & (df['exp_name'] == 'svo_perm')].copy()

    ### initialize new columns
    accuracy_types = ["s", "v", "o", "sv", "so", "vo", "svo"]
    for acc_type in accuracy_types:
        col_name            = f"{acc_type}_acc"
        subset_df[col_name] = None

    ### process each row in the subset
    for idx, row in subset_df.iterrows():
        result_dir = f'{RESULTS_PATH}/{row["work_dir"]}'
        ref_fpath  = f"{result_dir}/translations.ref"
        hyp_fpath  = f"{result_dir}/translations.hyp"

        with open(ref_fpath, 'r') as file:
            ref_trans = [line.strip() for line in file.readlines()]
        with open(hyp_fpath, 'r') as file:
            hyp_trans = [line.strip() for line in file.readlines()]

        translations    = list(zip(ref_trans, hyp_trans))
        accuracy_counts = {"S": 0, "V": 0, "O": 0, "SV": 0, "VO": 0, "SO": 0, "SVO": 0}

        ### extract the word order
        src_perm   = eval(row['tgt'])[0].split('.')[0]
        word_order = list(src_perm.upper())

        for tgt, hyp in translations:
            ### replace unknown tokens/the
            tgt = tgt.replace('the', '').replace("<<unk>> ", "UNK")
            hyp = hyp.replace('the', '').replace("<<unk>> ", "UNK")

            tgt_words = tgt.split()
            hyp_words = hyp.split()

            if len(tgt_words) != 3 or len(hyp_words) != 3:
                continue

            ### map words to S, V, and O
            tgt_mapped = {word_order[i]: tgt_words[i] for i in range(3)}
            hyp_mapped = {word_order[i]: hyp_words[i] for i in range(3)}

            for subset in accuracy_counts.keys():
                if all(tgt_mapped.get(pos) == hyp_mapped.get(pos) for pos in subset):
                    accuracy_counts[subset] += 1

        total_sentences = len(translations)
        normalized_accuracy = {subset: count / total_sentences for subset, count in accuracy_counts.items()}

        for acc_key, acc_value in normalized_accuracy.items():
            col_name = f"{acc_key.lower()}_acc"  # s_acc for example
            subset_df.at[idx, col_name] = acc_value

    subset_df.to_csv(svo_fpath, index=False)
    return subset_df

if __name__ == "__main__":
    create_svo_df()