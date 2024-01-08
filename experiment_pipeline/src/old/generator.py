import yaml

from new_config import *
from helper import format_number

from abc import ABC, abstractmethod


class PathConfig:

    def __init__(self):
        self.SRC_PATH     = os.getcwd()                                     # path to src code 
        self.MAIN_PATH    = os.path.dirname(self.SRC_PATH)                  # main experiment_pipeline path
        self.PERANTO_PATH = os.path.dirname(self.MAIN_PATH)                 # testperanto main path
        self.DATA_PATH    = f"{self.MAIN_PATH}/experiment_data"             # data path
        self.EXP_PATH     = f"{self.DATA_PATH}/experiments/{self.exp_name}" # path for this experiment
        self.TRAIN_PATH   = f"{self.EXP_PATH}/data"                         # for train/test/dev sets
        self.RESULTS_PATH = f'{self.EXP_PATH}/results'                      # for model results
        self.APPA_PATH    = f"{self.PERANTO_PATH}/appa-mt/fairseq"          # for appa fairseq training
        self.JSON_PATH    = f"{self.DATA_PATH}/peranto_files"               # contains 3 folders w/ tp config 
        self.OUT_PATH     = f"{self.EXP_PATH}/output"                       # path for generated data 
        self.YAML_FPATH   = f"{self.EXP_PATH}/{self.exp_name}.yaml"         


class PerantoTree:
    def __init__(self,
            json_path,
            amr_files,
            middleman_files,
            language_files,
            names
            ):
        
        self.amr_paths    = [f'{json_path}/amr_files/{amr_file}.json' for amr_file in amr_files]
        self.middle_paths = [f'{json_path}/middleman_files/{mid_file}.json' for mid_file in middleman_files]
        self.lang_paths   = [f'{json_path}/language_files/{lang_file}.json' for lang_file in language_files]

        self.names        = names 

        self.file_order   = {i : name for i, name in enumerate(self.names)}

        self.data         = [self.amr_paths, self.middle_paths, self.lang_paths]
        self.num_paths  = len(self.amr_paths) * len(self.middle_paths) * len(self.lang_paths)


class Generator(ABC):

    def __init__(self, **kwargs):
        super().__init__()

        self.exp_name    = config.exp_name
        self.yaml_fpath  = config.YAML_FPATH
        self.sh_fpath    = config.SH_FPATH
        self.output_path = config.OUT_PATH
        self.tp_path     = config.PERANTO_PATH

        self.per_tree    = config.per_tree 
        self.max_len     = config.max_len
        self.file_order  = self.per_tree.file_order
        self.num_paths   = self.per_tree.num_paths

    def generate_yaml(self): 
        """Generates yaml file"""
        def format(paths):
            ### how yaml config must look 
            if len(paths) == 1:  
                return paths[0]
            
            return {
                "branch" : {idx + 1 : [path] for idx, path in enumerate(paths)}
                }
        
        per_tree = self.per_tree.data # [[amr_path1, ...], [middle_path1, ...], [lang_path1, ...]]

        data = [format(paths) for paths in per_tree]

        with open(self.yaml_fpath, "w") as yaml_file:
            yaml.dump(data, yaml_file, default_flow_style=False)

    def create_sh_script(self):
        """Creates sh script to run generate data assoc. w/ above yaml file"""
        with open(self.sh_fpath, 'w') as f:
            # Write the initial part of the script
            f.write("#!/bin/sh\n")

            # Initialize the slurm stuff 
            f.write(f"""
#SBATCH -c 1 # Request 1 CPU cores
#SBATCH -t 0-10:00 # Runtime in D-HH:MM
#SBATCH -p dl # Partition to submit to
#SBATCH --mem=2G # Request 2G of memory
#SBATCH -o {self.output_path}/output.out # File to which STDOUT will be written
#SBATCH -e {self.output_path}/error.err # File to which STDERR will be written
#SBATCH --gres=gpu:0 # Request 0 GPUs\n""")

            ### call parallel gen and generate data 
            max_len = self.max_len
            form_len = format_number(max_len)
            f.write(f"python {self.tp_path}/scripts/parallel_gen.py -c {self.yaml_fpath} -n {max_len} -o {self.output_path}/{self.exp_name}{form_len}\n") 

            for j in range(self.num_paths):
                name = self.file_order[j]
                output_name = f"{self.output_path}/{self.exp_name}"
                ### change name from default (e.g. name.1 to name.svo)
                f.write(f"mv {output_name}{form_len}.{j} {output_name}{form_len}.{name}\n")

    def generate(self):
        ### generates sh script to be called 
        self.generate_yaml()
        self.create_sh_script()

if __name__ == "__main__":
    config = SizeConfig()
    generator = DataGenerator(config)
    generator.generate()







