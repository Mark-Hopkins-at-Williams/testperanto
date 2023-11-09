from collections import defaultdict
import itertools 
import os 

import matplotlib.pyplot as plt 

from config import Config
from data_processor import DataProcessor
from data_generator import DataGenerator

class ModelConfig:
    ### this configures each model run- I think this allows for more flexibility down the road but for now doesn't do much
    def __init__(self, config, work_dir, data_dir, src, tgt, **kwargs):
        ### required 
        self.config = config
        self.work_dir = work_dir
        self.data_dir = data_dir
        self.src = src
        self.tgt = tgt
        
        ### optional (default vals specified)
        self.patience   = config.patience
        self.num_epochs = config.num_epochs
        
        ### changing optional if given in kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

class Trainer:

    def __init__(self, config: Config):
        self.config       = config
        self.per_tree     = config.peranto_tree
        self.num_trans    = config.num_trans
        self.corp_lens    = config.corp_lens
        self.train_path   = config.TRAIN_PATH
        self.results_path = config.RESULTS_PATH
        self.exp_path     = config.EXP_PATH
        self.appa_path    = config.APPA_PATH

        self.num_epochs   = config.num_epochs
        self.num_gpus     = config.num_gpus
        self.patience     = config.patience

    def format_number(self, num):
        if num >= 1000000:
            return f"{num/1000000:.1f}m"
        elif num >= 1000:
            return f"{num/1000:.1f}k"
        else:
            return str(num)

    def create_model_configs(self):
        model_configs = []
        num = len(self.per_tree.names)
        combinations = list(itertools.combinations(range(num), self.num_trans))
        identity     = list(zip(range(num), range(num))) #[(0,0), (1,1), ..., (5,5)]
        all_combos = combinations + identity 

        for corp_len in self.corp_lens:
            for idxs in all_combos:
                folder_name = f"{'_'.join([self.per_tree.names[i] for i in idxs])}_{self.format_number(corp_len)}" 
                data_dir = f"{self.train_path}/{folder_name}"
                work_dir = f"{self.results_path}/{folder_name}"

                ### calls are only things to still be trained so you can call this again if it screws up midway
                if not os.path.exists(work_dir):
                    src = self.per_tree.languages[idxs[0]] 
                    if idxs[0] == idxs[1]:
                        tgt = "da"
                    else:
                        tgt = self.per_tree.languages[idxs[1]] 
                
                #src = self.per_tree.languages[idxs[0]]
                #tgt = self.per_tree.languages[idxs[1]]

                    model_config = ModelConfig(
                        config   = self.config,
                        work_dir = work_dir,
                        data_dir = data_dir,
                        src      = src,
                        tgt      = tgt
                        )
                    model_configs.append(model_config)
        return model_configs

    def create_train_script(self):
        model_configs = self.create_model_configs()

        scripts = []
        for c in model_configs:
            source_file = f"""
SRC={c.src}
TGT={c.tgt}
MAX_EPOCHS={c.num_epochs}

mkdir {c.work_dir}
cp -R {c.data_dir} {c.work_dir}/data

SCRIPT_PATH={self.appa_path}
bash $SCRIPT_PATH/prepare-data.sh {c.work_dir} $SRC $TGT {self.appa_path}

TEXT={c.work_dir}/data-tokenized
BINARY_TEXT={c.work_dir}/data-bin

fairseq-preprocess --source-lang $SRC --target-lang $TGT \\
    --trainpref $TEXT/train --validpref $TEXT/dev --testpref $TEXT/test \\
    --destdir $BINARY_TEXT \\
    --scoring chrf \\
    --workers 20

CUDA_VISIBLE_DEVICES=0 fairseq-train \\
    $BINARY_TEXT \\
    --max-epoch $MAX_EPOCHS \\
    --arch transformer_iwslt_de_en --share-decoder-input-output-embed \\
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \\
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \\
    --dropout 0.3 --weight-decay 0.0001 \\
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \\
    --max-tokens 4096 \\
    --eval-bleu \\
    --eval-bleu-args '{{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}}' \\
    --eval-bleu-detok moses \\
    --eval-bleu-remove-bpe \\
    --eval-bleu-print-samples \\
    --scoring chrf \\
    --no-epoch-checkpoints \\
    --save-dir {c.work_dir}/checkpoints \\
    --skip-invalid-size-inputs-valid-test \\
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \\
    --patience {c.patience} \\
    --tensorboard-logdir {c.work_dir}/tensorboard_logs/

fairseq-generate $BINARY_TEXT \\
    --path {c.work_dir}/checkpoints/checkpoint_best.pt \\
    --batch-size 128 --beam 5 --remove-bpe > {c.work_dir}/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py {c.work_dir}/translations

sacrebleu {c.work_dir}/translations.ref -i {c.work_dir}/translations.hyp -m bleu chrf > {c.work_dir}/scores
"""
            scripts.append(source_file)
        
        ### split into list of self.num_gpus lists, each of which has a bunch of scripts
        scripts = [scripts[i :: self.num_gpus] for i in range(self.num_gpus)]

        ### turning into self.num_gpu scripts
        for i in range(self.num_gpus):

            shell_script = f'{self.exp_path}/train{i}.sh'

            with open(shell_script, 'w') as f: 
                f.write("""#!/bin/bash\n""")
                f.write(f"""
#SBATCH -c 8 # Request 8 CPU cores
#SBATCH -t 2-00:00 # Runtime in D-HH:MM
#SBATCH -p dl # Partition to submit to
#SBATCH --mem=2G # Request 2G of memory
#SBATCH -o {self.exp_path}/output{i}.out # File to which STDOUT will be written
#SBATCH -e {self.exp_path}/error{i}.err # File to which STDERR will be written
#SBATCH --gres=gpu:1 # Request 1 GPUs\n""")
                
                for script in scripts[i]:
                    f.write(script)
                    f.write("\n")

    def get_scores(self):
        num = len(self.per_tree.names)
        combinations = list(itertools.combinations(range(num), self.num_trans))
        identity     = list(zip(range(num), range(num))) #[(0,0), (1,1), ..., (5,5)]
        all_combos = combinations + identity 

        scores = defaultdict(list)
        for corp_len in self.corp_lens:
            for idxs in all_combos:
                names = '_'.join([self.per_tree.names[i] for i in idxs]) #SVO_OVS
                folder_name = f"{names}_{self.format_number(corp_len)}" 
                result_dir = f"{self.results_path}/{folder_name}"
                
                try:
                    with open(f'{result_dir}/scores') as f:
                        contents = eval(f.read())
                        bleu = contents[0]['score']
                        scores[names].append((corp_len, bleu))
                except:
                    pass 
        return scores 

    def plot_scores(self):
        scores = self.get_scores()
        plt.figure(figsize=(15,10))

        for name, points in scores.items():
            lengths, bleus = zip(*points)
            plt.plot(lengths, bleus, label=name, marker='o', linestyle='-')

        plt.legend()
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