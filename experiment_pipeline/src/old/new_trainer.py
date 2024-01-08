import os 

from new_config import *
from helper import format_number


class Trainer:
    """
    Trainer class takes all the processed dataset and trains models on them
    """
    def __init__(self, config: AbstractConfig):
        self.results_path = config.RESULTS_PATH
        self.exp_path     = config.EXP_PATH
        self.appa_path    = config.APPA_PATH
        self.train_path   = config.TRAIN_PATH

        self.config       = config
        self.datasets     = config.datasets
        self.models       = config.models
        self.num_gpus     = config.num_gpus

    def create_train_script(self):
        """
        Create num_gpus shell scripts to train. Uses the above function to create lst
        of model configs
        """
        scripts = []
        for dataset in self.datasets:
            for m in self.models:
                data_dir = f'{self.train_path}/{dataset.name}'
                work_dir = f"{self.results_path}/{dataset.name}"
            ### base training shell script
            source_file = f"""
SRC=src
TGT=tgt
MAX_EPOCHS={m.num_epochs}

mkdir {work_dir}
cp -R {data_dir} {work_dir}/data

SCRIPT_PATH={self.appa_path}
bash $SCRIPT_PATH/prepare-data.sh {work_dir} $SRC $TGT {self.appa_path}

TEXT={work_dir}/data-tokenized
BINARY_TEXT={work_dir}/data-bin

fairseq-preprocess --source-lang $SRC --target-lang $TGT \\
    --trainpref $TEXT/train --validpref $TEXT/dev --testpref $TEXT/test \\
    --destdir $BINARY_TEXT \\
    --scoring chrf \\
    --workers 20

CUDA_VISIBLE_DEVICES=0 fairseq-train \\
    $BINARY_TEXT \\
    --max-epoch $MAX_EPOCHS \\
    --arch {m.model} --share-decoder-input-output-embed \\
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
    --save-dir {work_dir}/checkpoints \\
    --skip-invalid-size-inputs-valid-test \\
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \\
    --patience {m.patience} \\
    --tensorboard-logdir {work_dir}/tensorboard_logs/

fairseq-generate $BINARY_TEXT \\
    --path {work_dir}/checkpoints/checkpoint_best.pt \\
    --batch-size 128 --beam 5 --remove-bpe > {work_dir}/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py {work_dir}/translations

sacrebleu {work_dir}/translations.ref -i {work_dir}/translations.hyp -m bleu chrf > {work_dir}/scores
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
#SBATCH -t 5-00:00 # Runtime in D-HH:MM
#SBATCH -p dl # Partition to submit to
#SBATCH --mem=2G # Request 2G of memory
#SBATCH -o {self.exp_path}/output{i}.out # File to which STDOUT will be written
#SBATCH -e {self.exp_path}/error{i}.err # File to which STDERR will be written
#SBATCH --gres=gpu:1 # Request 1 GPUs\n""")
                
                for script in scripts[i]:
                    f.write(script)
                    f.write("\n")

if __name__ == '__main__':
    config = SVOConfig()
    trainer = Trainer(config)
    trainer.create_train_script()
