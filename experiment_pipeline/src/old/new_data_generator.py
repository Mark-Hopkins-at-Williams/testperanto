import yaml

from new_config import *
from helper import format_number

class DataGenerator:
    """
    Class for generating testperanto data for translations. 
    """

    def __init__(self, config: AbstractConfig):
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







