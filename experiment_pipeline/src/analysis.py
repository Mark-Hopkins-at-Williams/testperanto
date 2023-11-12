import pandas as pd 
import os 
import tensorflow as tf

from config import Config
from helper import format_number

def get_data(config, combo, corp_len):
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

        return [corp_len, combo[0], combo[1], bleu, chrF2, num_steps]

    except Exception:
        # hasn't been processed yet 
        return None 

def create_df(config):
    data = [get_data(config, combo, corp_len) 
            for corp_len in config.corp_lens 
            for combo in config.combos]

    data = [d for d in data if d is not None] # filter out unprocessed results
    df = pd.DataFrame(data, columns=['corpus_length', 'src', 'tgt', 'bleu', 'chrF2', 'num_steps'])
    csv_path = f"{config.EXP_PATH}/results.csv"
    df.to_csv(csv_path, index=False)

if __name__ == "__main__":
    config = Config()
    create_df(config)

"""

    def plot_scores(self):
        scores = self.get_scores()
        plt.figure(figsize=(15,10))

        # Defining a list of distinct colors
        colors = ['#00BFFF', '#228B22', '#FF6347', '#7851A9', '#FFA500', '#008080', '#708090']
        color_cycle = itertools.cycle(colors)

        # Defining different line markers
        markers = ['o', 's', '^', 'x', '*', '+', 'd']
        marker_cycle = itertools.cycle(markers)

        handles, labels = [], []

        for name, points in scores.items():
            lengths, bleus = zip(*points)
            color = next(color_cycle)
            marker = next(marker_cycle)
            line, = plt.plot(lengths, bleus, label=name, color=color, marker=marker, linestyle='-')
            handles.append(line)
            labels.append((name, max(bleus)))

        labels, handles = zip(*sorted(zip(labels, handles), key=lambda x: x[0][1], reverse=True))
        labels = [label[0] for label in labels]

        plt.legend(handles, labels)

        plt.title('BLEU Scores vs. Corpus Lengths')
        plt.xlabel("Corpus Size")
        plt.ylabel("BLEU Score")

        plt.grid(True)
        plt.show()

        plot_path = f"{self.exp_path}/bleu.jpg"
        plt.savefig(plot_path)

if __name__ == '__main__':
    config = Config()
    trainer = Trainer(config)
    #trainer.create_train_script()
    trainer.plot_scores()


Analysis:

(1) BLEU vs. Corpus Size for all Training
(2) Create Dataframe (combination, bleu, corpus size, ...)
(3) Python function that automatically opens tensorboard to easily get that going
    You would need to go to terminal and call
    tensorboard --logdir {work_dir}/tensorboard_logs/

    where tensorboard {work_dir} = f"{self.results_path}/{'_'.join(combo)}_{form_len}" 
    (see model config function for what these all are)
(4) Collect data to understand what to set epochs/patience etc. 
(5) Or generally analyzing data from training at each individual model training level 
    i.e. how fast/slow did 1k train, when did it stop, etc.






"""