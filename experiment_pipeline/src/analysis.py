import pandas as pd 
import os 
import tensorflow as tf

from config import *
from helper import format_number

def get_data(config, combo, corp_len):
    res = []
    form_len = format_number(corp_len)
    folder_name = f"{'_'.join(combo)}_{form_len}"
    result_dir = f"{config.RESULTS_PATH}/{folder_name}"
    scores = f"{result_dir}/scores"

    try:
        with open(scores) as f:
            contents = eval(f.read())
            bleu = contents[0]['score']
            chrF2 = contents[1]['score']

        tensorboard_path = f"{result_dir}/tensorboard_logs"
        num_steps = None
        for section in ['train']: # for now just getting num_steps 'train_inner', 'valid']:
            tb_path = f"{tensorboard_path}/{section}"
            for event_file in os.listdir(tb_path):
                event_path = os.path.join(tb_path, event_file)
                last = list(tf.compat.v1.train.summary_iterator(event_path))[-1]
                num_steps = last.step

        res.append([corp_len, combo[0], combo[1], bleu, chrF2, num_steps])
        return [corp_len, combo[0], combo[1], bleu, chrF2, num_steps]

    except Exception:
        # hasn't been processed yet 
        return None 

def create_df(config):
    all_combos = config.combos + [combo[::-1] for combo in config.combos] # bc src-tgt and tgt-src were trained
    data = [get_data(config, combo, corp_len) 
            for corp_len in config.corp_lens 
            for combo in config.combos]

    data = [d for d in data if d is not None] # filter out unprocessed results
    df = pd.DataFrame(data, columns=['corpus_length', 'src', 'tgt', 'bleu', 'chrF2', 'num_steps'])
    csv_path = f"{config.EXP_PATH}/results.csv"
    df.to_csv(csv_path, index=False)

if __name__ == "__main__":
    config = SVOConfig()
    create_df(config)

