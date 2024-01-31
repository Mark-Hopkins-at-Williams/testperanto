import json 
import csv 

from globals import *
from data_splitter import Dataset

class Model:
    """
    Data class that specifies basic model configs. 

    Args:
        name (str, optional): model name
        kwargs can be num_epochs, patience, size (XS/S/M)

    Important Info:
        - Here XS, S, and M are analogous to the models in Shaham 2023 Causes and Cures 
        for Interference in Multilingual Translation (the names like transformer_iwslt_de_en
        can be found in the fairseq documentation)
        - The name has only ever been the size (XS) so far, but that's only because we haven't
        really changed anything else 
    """

    def __init__(self, name=None, **kwargs):
        self.name       = name
        self.num_epochs = NUM_EPOCHS
        self.patience   = PATIENCE
        self.size       = MODEL_SIZE

        size_map = {
            "XS" : 'transformer_iwslt_de_en',
            "S"  : 'transformer',
            'M'  : "transformer_vaswani_wmt_en_de_big"
            }

        for key, value in kwargs.items():
            setattr(self, key, value)

        self.model = size_map[self.size] 

class Trainer:
    """
    Trainer class. For each Dataset in datasets, trains one Model
    for each Model in models.

    Args:
        exp_name (str)          : experiment name
        datasets (list[Dataset]): list of Datasets to train on
        models (list[Model])    : list of Models to train

    Methods:
        update_metadata : adds untrained model data to results.csv
        create_sh_script: calls update_metadata and then creates NUM_GPUS train .sh scripts

    Important Info:
        - results.csv adds '' for any info that will be updated later (ex: bleu). These can only
        be added later (namely by calling results.py)
        - Before we were using 1000 epochs/30 patience, which will surely overtrain. It's now 
        set to 300 epochs, which may slightly undertrain but overall is fine.
        - Regardless training takes a long time- the switches experiment takes probably around a week 
        - It's likely worthwhile to look into ways to dynamically utilize the GPU best
        - The default is to train every model that hasn't been trained yet. So if you create a Trainer
        object and realize you don't want to train those models you need to manually delete those rows
        from results.csv
        - The base training script allows for more flexibility if needed 
        - There's potentially an issue with one training script taking way longer than the other. For example 
        say I did 2 Models, one with num_epochs = 1 and the other num_epochs = 10000. The current code will 
        split the two scripts in such a way that all of the 1 epoch are together. 

    Example:
        >>> datasets = fetch_data(['new_switches'])
        >>> models   = [Model('XS', size = "XS")]
        >>> trainer  = Trainer('new_switches', datasets, models)
        >>> trainer.create_train_script() 
    """

    def __init__(self, exp_name, datasets, models):
        self.exp_name = exp_name 
        self.datasets = datasets
        self.models   = models 

    def update_metadata(self):
        csv_file_path = f"{DATA_PATH}/results.csv"
    
        existing_data = {} # keep track of existing rows
        try:
            with open(csv_file_path, 'r', newline='') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    # specifically keep track of if they have been trained (bleu is not None)
                    existing_data[row['work_dir']] = row['bleu']
        except FileNotFoundError:
            pass

        with open(csv_file_path, 'a', newline='') as file:
            cols   = ['exp_name', 'work_dir', 'dataset_name', 'model_name', 'bleu', 'chrF2', 
                    'num_steps', 'src', 'tgt', 'corp_lens', 'model_arch', 'num_epochs', 'patience']
            writer = csv.DictWriter(file, fieldnames=cols)
    
            if not existing_data:
                writer.writeheader()

            to_train = []
            for dataset in self.datasets:
                for m in self.models:
                    add      = f"{m.name}_" if m.name is not None else ''
                    work_dir = f"{add}{dataset.name}" #ex: XS_SVO_SOV_4.0k
                    
                    # if row doesn't exist add to df 
                    if work_dir not in existing_data: 
                        to_train.append((dataset, m))
                        row = {
                            'exp_name': self.exp_name,
                            'work_dir'    : work_dir,
                            'dataset_name': dataset.name,
                            'model_name'  : m.name,
                            'bleu'        : '',
                            'chrF2'       : '',
                            'num_steps'   : '',
                            'src'         : dataset.src,
                            'tgt'         : dataset.tgt,
                            'corp_lens'   : dataset.corp_lens,
                            'model_arch'  : m.size,
                            'num_epochs'  : m.num_epochs,
                            'patience'    : m.patience
                            }
                        writer.writerow(row)
                        existing_data[work_dir] = None
                    
                    # but still train if haven't been trained yet 
                    elif not existing_data[work_dir]: 
                        to_train.append((dataset, m))

            return to_train
            
    def create_train_script(self):
        scripts = []
        to_train = self.update_metadata()
        
        if len(to_train) == 0:
            return None 
            
        for (dataset, m) in to_train:
            add = f"{m.name}_" if m.name is not None else ''
            data_dir = f'{DATASET_PATH}/{dataset.name}'
            work_dir = f"{RESULTS_PATH}/{add}{dataset.name}" #{RESULTS_PATH}/M_SVO_SOV_32k
            ### training script
            source_file = f"""
SRC=src
TGT=tgt
MAX_EPOCHS={m.num_epochs}

mkdir {work_dir}
cp -R {data_dir} {work_dir}/data

SCRIPT_PATH={APPA_PATH}
bash $SCRIPT_PATH/prepare-data.sh {work_dir} $SRC $TGT {APPA_PATH}

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
    
        ### split in NUM_GPUS scripts 
        scripts = [scripts[i :: NUM_GPUS] for i in range(NUM_GPUS)]

        ### create each of NUM_GPU scripts 
        for i in range(NUM_GPUS):

            shell_script = f'{RUN_PATH}/train{i}.sh'

            with open(shell_script, 'w') as f: 
                f.write("""#!/bin/bash\n""")
                f.write(f"""
#SBATCH -c 2 # Request 8 CPU cores
#SBATCH -t 10-00:00 # Runtime in D-HH:MM
#SBATCH -p dl # Partition to submit to
#SBATCH --mem=2G # Request 2G of memory
#SBATCH -o {RUN_PATH}/train{i}.out # File to which STDOUT will be written
#SBATCH -e {RUN_PATH}/train{i}.err # File to which STDERR will be written
#SBATCH --gres=gpu:1 # Request 1 GPUs\n""")
                
                for script in scripts[i]:
                    f.write(script)
                    f.write("\n")

def fetch_data(splitter_names=None, corp_lens=None):
    """
    Fetches list[datasets] from dataset_info.json by specifying
    the splitter(s) and the corp_len(s)

    Args:
        splitter_names (list[str], optional): splitters to fetch
        corp_lens      (list[int], optional): corpus lengths to fetch

    Example:
        >>> fetch_data(['new_switches']) gets all Datasets created by the 
        Splitter instance with name new_switches 
    """
    result    = []
    data_path = f"{DATA_PATH}/dataset_info.json"

    with open(data_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    if splitter_names is not None:
        for splitter_name in splitter_names:
            for dataset_dict in data[splitter_name]:
                corp_len = sum(list(dataset_dict.values())[0]['corp_lens'])
                if corp_lens is None or corp_len in corp_lens:
                    ### recreate Dataset object
                    dataset = Dataset(
                        list(dataset_dict.keys())[0],
                        list(dataset_dict.values())[0]['src'],
                        list(dataset_dict.values())[0]['tgt'],
                        list(dataset_dict.values())[0]['corp_lens']
                        )
                    result.append((corp_len, dataset))
    ### sort first by corp len and then by name 
    sorted_res = sorted(result, key=lambda x: (x[0], x[1].name))  
    return [dataset for _,dataset in sorted_res]

if __name__ == '__main__':
    datasets = fetch_data(['new_switches'])
    models   = [Model('XS', size = "XS")]
    trainer  = Trainer('new_switches', datasets, models)
    trainer.create_train_script()

