import itertools 
import os 

from config import Config
from data_processor import DataProcessor
from data_generator import DataGenerator

class Trainer:

    def __init__(self, config: Config):
        self.per_tree = config.peranto_tree
        self.num_trans = config.num_trans
        self.corp_lens = config.corp_lens
        self.train_path = config.TRAIN_PATH
        self.results_path = config.RESULTS_PATH
        self.exp_path = config.EXP_PATH
        self.appa_path    = config.APPA_PATH
        self.num_epochs = config.num_epochs

    def format_number(self, num):
        if num >= 1000000:
            return f"{num/1000000:.1f}m"
        elif num >= 1000:
            return f"{num/1000:.1f}k"
        else:
            return str(num)

    def create_train_script_alternate(self, num_files = 1):
        configs = self.get_calls_alternate()

        scripts = []
        for config in configs:
            # TODO: add flag for early stopping
            work_dir, data_dir, src, tgt, max_epochs = config

            source_file = f"""
SRC={src}
TGT={tgt}
MAX_EPOCHS={max_epochs}

mkdir {work_dir}
cp -R {data_dir} {work_dir}/data

###########
# THIS PART OF THE CODE FIGURES OUT THE DIRECTORY OF THE SHELL SCRIPT.
# THIS IS SOMEWHAT COMPLICATED WHEN WE RUN THROUGH SLURM.
# check if script is started via SLURM or bash
# if with SLURM: there variable '$SLURM_JOB_ID' will exist
# `if [ -n $SLURM_JOB_ID ]` checks if $SLURM_JOB_ID is not an empty string
if [ -n $SLURM_JOB_ID ];  then
    # check the original location through scontrol and $SLURM_JOB_ID
    SCRIPT_COMMAND=$(scontrol show job $SLURM_JOBID | awk -F= '/Command=/{{print {data_dir}}}')
else
    # otherwise: started with bash. Get the real location.
    SCRIPT_COMMAND=$(realpath $0)
fi
SCRIPT_COMMAND_ARRAY=($SCRIPT_COMMAND)
SCRIPT_NAME=${{SCRIPT_COMMAND_ARRAY[0]}}
SCRIPT_PATH=$(dirname $SCRIPT_NAME)
#
###########

bash $SCRIPT_PATH/prepare-data.sh {work_dir} $SRC $TGT

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
    --save-dir {work_dir}/checkpoints \\
    --skip-invalid-size-inputs-valid-test \\
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric

fairseq-generate $BINARY_TEXT \\
    --path {work_dir}/checkpoints/checkpoint_best.pt \\
    --batch-size 128 --beam 5 --remove-bpe > {work_dir}/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py {work_dir}/translations

sacrebleu {work_dir}/translations.ref -i {work_dir}/translations.hyp -m bleu chrf > {work_dir}/scores
"""
            scripts.append(source_file)
        
        for i in range(0, num_files):

            shell_script = f'{self.exp_path}/train_alt_{i}.sh'

            with open(shell_script, 'w') as f: 
                f.write("""#!/bin/sh\n""")
                f.write(f"""
#SBATCH -c 8 # Request 8 CPU cores
#SBATCH -t 2-00:00 # Runtime in D-HH:MM
#SBATCH -p dl # Partition to submit to
#SBATCH --mem=2G # Request 2G of memory
#SBATCH -o {self.results_path}/output.out # File to which STDOUT will be written
#SBATCH -e {self.results_path}/error.err # File to which STDERR will be written
#SBATCH --gres=gpu:1 # Request 1 GPUs\n""")

                for script_num, script in enumerate(scripts):
                    if script_num % num_files == 0:
                        f.write(script)
                        f.write("\n")
            # function done 


    def create_train_script(self):
        shell_script = f'{self.exp_path}/train.sh'

        with open(shell_script, 'w') as f: 
            f.write("""#!/bin/sh\n""")
            f.write(f"""
#SBATCH -c 8 # Request 8 CPU cores
#SBATCH -t 2-00:00 # Runtime in D-HH:MM
#SBATCH -p dl # Partition to submit to
#SBATCH --mem=2G # Request 2G of memory
#SBATCH -o {self.results_path}/output.out # File to which STDOUT will be written
#SBATCH -e {self.results_path}/error.err # File to which STDERR will be written
#SBATCH --gres=gpu:2 # Request 2 GPUs\n""")
            
            calls = self.get_calls()

            # Initialize an empty list to store job IDs
            job_ids = []

            for call in calls:
                f.write("OUTPUT=$( " + call.strip() + " )\n")  # Get the output of sbatch command
                f.write("JOB_ID=$(echo $OUTPUT | awk '{print $NF}')\n")  # Extract the job ID
                job_ids.append("$JOB_ID")

                # If there are more than 1 job IDs stored, wait for the earliest job to complete
                if len(job_ids) > 1:
                    f.write(f"""srun -n 1 -c 1 --mem=1M sh -c "while squeue -j {job_ids.pop(0)} | grep -q ' R\| PD'; do sleep 60; done"\n""")
                        # Ensure all jobs complete at the end
            for job_id in job_ids:
                f.write(f"""srun -n 1 -c 1 --mem=1M sh -c "while squeue -j {job_id} | grep -q ' R\| PD'; do sleep 60; done"\n""")
            
            f.write("EOT\n")

    def get_calls(self):

        # all the calls generated here
        calls = []
        for corp_len in self.corp_lens:
            for idxs in itertools.combinations(range(len(self.per_tree.names)), self.num_trans):
                folder_name = f"{'_'.join([self.per_tree.names[i] for i in idxs])}_{self.format_number(corp_len)}" 
                data_dir = f"{self.train_path}/{folder_name}"
                work_dir = f"{self.results_path}/{folder_name}"

                ### calls are only things to still be trained so you can call this again if it screws up midway
                if not os.path.exists(work_dir):
                    src = self.per_tree.languages[idxs[0]] 
                    tgt = self.per_tree.languages[idxs[1]] 
                    
                    call = f"sbatch {self.appa_path}/train.sh {work_dir} {data_dir} {src} {tgt} {self.num_epochs}\n"
                    calls.append(call)
        return calls
    
    def get_calls_alternate(self):
        # all the calls generated here
        calls = []
        for corp_len in self.corp_lens:
            for idxs in itertools.combinations(range(len(self.per_tree.names)), self.num_trans):
                folder_name = f"{'_'.join([self.per_tree.names[i] for i in idxs])}_{self.format_number(corp_len)}" 
                data_dir = f"{self.train_path}/{folder_name}"
                work_dir = f"{self.results_path}/{folder_name}"

                ### calls are only things to still be trained so you can call this again if it screws up midway
                if not os.path.exists(work_dir):
                    src = self.per_tree.languages[idxs[0]] 
                    tgt = self.per_tree.languages[idxs[1]] 

                    call = (work_dir, data_dir, src, tgt, self.num_epochs)
                    calls.append(call)
        return calls


if __name__ == '__main__':
    config = Config()
    trainer = Trainer(config)
    trainer.create_train_script_alternate(1)